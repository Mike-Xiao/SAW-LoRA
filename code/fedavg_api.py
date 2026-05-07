import copy
import gc
import csv
import logging
import os
import re
import time
from decimal import Decimal, InvalidOperation

import numpy as np
import torch

from client import Client
from safety_eval import evaluate_all_clients_safety, evaluate_safety
from safety_data_loader import get_client_type


def _append_trace(trace_path, line: str):
    if not trace_path:
        return
    try:
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
    except Exception:
        pass


def _clip(s, n=240):
    s = s or ""
    if n is None or n <= 0:
        return s
    return s if len(s) <= n else (s[:n] + "...")


class FedAvgAPI:
    """FedAvg server for federated LoRA fine-tuning of LLMs."""

    def __init__(self, dataset, device, args, model_trainer,
                 safety_eval_data=None, safety_train_data=None,
                 tokenizer=None, task_eval_data=None):
        self.device = device
        self.args = args
        self.model_trainer = model_trainer
        self.safety_eval_data = safety_eval_data
        self.safety_train_data = safety_train_data
        self.tokenizer = tokenizer
        self.method = getattr(args, "method", "baseline")
        self.stage = getattr(args, "stage", "stage1_benign")
        self.task_eval_data = task_eval_data

        if len(dataset) == 3:
            client_data_num, client_train_data, test_loader = dataset
        elif len(dataset) >= 4:
            client_data_num, client_train_data, test_loader, d_task_eval = dataset[:4]
            if self.task_eval_data is None:
                self.task_eval_data = d_task_eval
        else:
            raise ValueError(f"Unexpected dataset container length: {len(dataset)}")

        self.test_loader = test_loader

        self.client_list = []
        logging.info("############ Setting up clients ############")
        for client_idx in range(args.client_num_in_total):
            client_safety = None
            if self.safety_train_data and client_idx in self.safety_train_data:
                client_safety = self.safety_train_data[client_idx]
            client = Client(
                client_idx=client_idx,
                local_training_data=client_train_data[client_idx],
                local_sample_number=client_data_num[client_idx],
                args=args, device=device, model_trainer=model_trainer,
                safety_train_data=client_safety,
            )
            self.client_list.append(client)
        logging.info("############ Clients ready ############")

        self.safety_judge_backend = None
        self.safety_judge_name = self._resolve_safety_judge_name()
        self.results = {"round": []}
        self.qa_trace_path = None
        self.safety_embed_base_params = None
        self.lora_gate_alpha_trace_path = None

    # ------------------------------------------------------------------
    def _banner(self, title, ch="*"):
        width = max(int(getattr(self.args, "log_banner_width", 90)), 40)
        line = ch * width
        logging.info(line)
        logging.info(title)
        logging.info(line)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self):
        start_time = time.time()
        self._banner(f"STAGE START | stage={self.stage} | method={self.method}", "#")
        self._validate_stage_runtime()

        init_lora_path = getattr(self.args, "init_lora_path", None)
        if self.stage == "stage3_personal":
            if init_lora_path:
                logging.warning(
                    "stage3_personal now trains from the base model; ignoring init_lora_path=%s",
                    init_lora_path,
                )
            logging.info("Stage3 starts from the base model with freshly initialized LoRA.")
        elif init_lora_path:
            if os.path.exists(init_lora_path):
                ckpt = torch.load(init_lora_path, map_location="cpu", weights_only=True)
                self.model_trainer.set_model_params(ckpt)
                logging.info(f"Loaded initial LoRA params from: {init_lora_path}")
            else:
                raise FileNotFoundError(f"init_lora_path not found: {init_lora_path}")

        if getattr(self.args, "eval_before_train", True):
            self._banner("PRE-TRAIN EVAL START (Round -1)", "=")
            self._evaluate(round_idx=-1)
            self._save_results()
            self._banner("PRE-TRAIN EVAL END (Round -1)", "=")

        if getattr(self.args, "eval_only", False):
            self._banner("EVAL-ONLY MODE: TRAINING SKIPPED", "#")
            return

        if self.method == "safelora":
            self._banner("INITIALISING SAFE LORA PROJECTOR", "+")
            self.model_trainer.init_safelora_projector(self.args)

        if self.method not in ("baseline", "safelora"):
            self._banner("INITIALISING SAFETY METHODS", "+")
            init_task_params = copy.deepcopy(self.model_trainer.get_model_params())

            safety_adapter_init = None
            if self.method == "local_safety_adapter":
                rank = getattr(self.args, "safety_adapter_rank", 4)
                self.model_trainer.add_safety_adapter(rank)
                safety_adapter_init = copy.deepcopy(
                    self.model_trainer.get_safety_adapter_params()
                )
            elif self.method == "safety_embed":
                self.safety_embed_base_params = self.model_trainer.prepare_safety_embed_base_params(
                    self.safety_train_data, self.args
                )

            for client in self.client_list:
                self.model_trainer.set_model_params(copy.deepcopy(init_task_params))
                if safety_adapter_init is not None:
                    self.model_trainer.set_safety_adapter_params(
                        copy.deepcopy(safety_adapter_init)
                    )
                if self.safety_embed_base_params is not None:
                    self.model_trainer.set_safety_embed_params(self.safety_embed_base_params)
                client.init_safety_method()
                if self.method == "safety_embed":
                    self.model_trainer.model.zero_grad(set_to_none=True)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            self.model_trainer.set_model_params(init_task_params)
            if safety_adapter_init is not None:
                self.model_trainer.set_safety_adapter_params(safety_adapter_init)
            if self.safety_embed_base_params is not None:
                self.model_trainer.set_safety_embed_params(self.safety_embed_base_params)

        eval_every = max(1, int(getattr(self.args, "eval_every_rounds", 1)))
        save_every = int(getattr(self.args, "save_every_rounds", 0))

        for round_idx in range(self.args.comm_round):
            round_start = time.time()
            self._banner(
                f"COMM ROUND {round_idx} START | elapsed={round_start - start_time:.0f}s", "*"
            )

            global_lora_params = self.model_trainer.get_model_params()
            client_indexes = self._client_sampling(round_idx)
            logging.info(f"Selected clients: {client_indexes}")

            lora_params_list = []
            for idx in client_indexes:
                client = self.client_list[idx]
                local_params = client.train(copy.deepcopy(global_lora_params))
                self._append_lora_gate_alpha_trace(round_idx, client)
                lora_params_list.append((client.get_sample_number(), local_params))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            global_lora_params = self._aggregate(lora_params_list)
            self.model_trainer.set_model_params(global_lora_params)

            should_eval = ((round_idx + 1) % eval_every == 0) or (round_idx == self.args.comm_round - 1)
            if should_eval:
                self._evaluate(round_idx)
                self._save_results()
            else:
                logging.info(f"Round {round_idx} | skip evaluation (eval_every_rounds={eval_every})")

            if save_every > 0 and (round_idx + 1) % save_every == 0:
                self._save_model(tag=f"round{round_idx}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            self._banner(f"COMM ROUND {round_idx} END", "*")

        self._save_model()
        self._banner("TRAINING FINISHED", "#")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validate_stage_runtime(self):
        stage = self.stage
        method = self.method
        if stage == "stage1_benign":
            if method != "baseline":
                logging.warning(f"[Stage1] Expected method=baseline, but got method={method}.")

        elif stage == "stage3_personal":
            if method not in ("local_safety_adapter", "local_safety_lora",
                              "ewc", "lwf", "replay", "safety_embed",
                              "safelora"):
                logging.warning(f"[Stage3] Expected personalized method, got {method}.")

        if method == "safelora":
            if not getattr(self.args, "safelora_base_model_path", None):
                raise ValueError("[SafeLoRA] --safelora_base_model_path is required.")
            if not getattr(self.args, "safelora_aligned_model_path", None):
                raise ValueError("[SafeLoRA] --safelora_aligned_model_path is required.")

    def _init_safety_judge_backend(self):
        judge_name = getattr(self.args, "safety_judge", "llamaguard").lower()

        if judge_name == "rule":
            logging.info("Safety judge backend: rule")
            return None, "rule"

        if self.tokenizer is None or self.safety_eval_data is None:
            logging.warning(f"{judge_name} requested but tokenizer/eval_data missing; fallback to rule.")
            return None, "rule"

        try:
            judge_model_name = getattr(self.args, "judge_model_name", "Llama-Guard-3-8B")
            kwargs = dict(
                judge_model_name=judge_model_name,
                device=self.device,
                max_input_len=int(getattr(self.args, "judge_max_input_len", 2048)),
                max_new_tokens=int(getattr(self.args, "judge_max_new_tokens", 8)),
                debug_samples=int(getattr(self.args, "judge_debug_samples", 0)),
                unknown_as_unsafe=bool(getattr(self.args, "judge_unknown_as_unsafe", 0)),
                judge_batch_size=int(getattr(self.args, "judge_batch_size", 8)),
            )
            from llamaguard_judge import LlamaGuardJudge
            backend = LlamaGuardJudge(**kwargs)
            logging.info(f"Safety judge backend: llamaguard ({judge_model_name})")
            return backend, "llamaguard"

        except Exception as e:
            logging.warning(f"Failed to init judge backend ({judge_name}), fallback to rule: {e}")
            return None, "rule"

    def _resolve_safety_judge_name(self):
        judge_name = getattr(self.args, "safety_judge", "llamaguard").lower()
        if judge_name == "rule":
            logging.info("Safety judge backend: rule")
            return "rule"
        if self.tokenizer is None or self.safety_eval_data is None:
            logging.warning(f"{judge_name} requested but tokenizer/eval_data missing; fallback to rule.")
            return "rule"
        logging.info(f"Safety judge backend: {judge_name} (lazy load)")
        return judge_name

    def _get_safety_judge_backend(self):
        if self.safety_judge_name == "rule":
            return None
        if self.safety_judge_backend is None:
            self.safety_judge_backend, self.safety_judge_name = self._init_safety_judge_backend()
        return self.safety_judge_backend

    def _release_safety_judge_backend(self):
        if self.safety_judge_backend is None:
            return
        try:
            model = getattr(self.safety_judge_backend, "model", None)
            if model is not None:
                model.to("cpu")
        except Exception:
            pass
        self.safety_judge_backend = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _client_sampling(self, round_idx):
        total = self.args.client_num_in_total
        per_round = self.args.client_num_per_round
        if total == per_round:
            return list(range(total))
        num = min(per_round, total)
        rng = np.random.RandomState(round_idx)
        return rng.choice(range(total), num, replace=False).tolist()

    def _aggregate(self, lora_params_list):
        if not lora_params_list:
            logging.warning("Aggregate called with empty client updates.")
            return self.model_trainer.get_model_params()

        total_samples = sum(max(n, 0) for n, _ in lora_params_list)
        if total_samples <= 0:
            logging.warning("All client sample counts are zero. Using first client params.")
            return copy.deepcopy(lora_params_list[0][1])

        _, first_params = lora_params_list[0]
        averaged = {}
        for key in first_params.keys():
            averaged[key] = torch.zeros_like(first_params[key], dtype=torch.float32)
            for sample_num, local_params in lora_params_list:
                if sample_num <= 0:
                    continue
                w = sample_num / total_samples
                averaged[key] += local_params[key].float() * w
            averaged[key] = averaged[key].to(first_params[key].dtype)
        return averaged

    def _get_lora_gate_alpha_trace_path(self):
        if not bool(getattr(self.args, "lora_download_gate", 0)):
            return None
        if not bool(getattr(self.args, "lora_gate_alpha_dump", 1)):
            return None
        if self.lora_gate_alpha_trace_path:
            return self.lora_gate_alpha_trace_path

        os.makedirs(self.args.output_dir, exist_ok=True)
        base_name = self._get_result_filename()
        if base_name.endswith(".txt"):
            base_name = base_name[:-4]
        self.lora_gate_alpha_trace_path = os.path.join(
            self.args.output_dir, f"{base_name}_alpha_trace.csv"
        )

        if not os.path.exists(self.lora_gate_alpha_trace_path):
            with open(self.lora_gate_alpha_trace_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "round",
                    "client_id",
                    "client_type",
                    "method",
                    "dataset",
                    "model_name",
                    "group",
                    "alpha",
                    "utility_gain",
                    "safety_risk",
                    "score",
                    "param_count",
                ])
        logging.info(f"LoRA gate alpha trace: {self.lora_gate_alpha_trace_path}")
        return self.lora_gate_alpha_trace_path

    def _append_lora_gate_alpha_trace(self, round_idx, client):
        path = self._get_lora_gate_alpha_trace_path()
        if not path:
            return

        metrics = client.get_last_lora_gate_metrics()
        if not metrics or not metrics.get("enabled", False):
            return

        group_alpha = metrics.get("group_alpha", {})
        if not group_alpha:
            return

        dataset_cfg = getattr(self.args, "dataset", "")
        if isinstance(dataset_cfg, (list, tuple)):
            dataset_name = "+".join(str(x) for x in dataset_cfg)
        else:
            dataset_name = str(dataset_cfg)

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for group, vals in sorted(group_alpha.items()):
                writer.writerow([
                    round_idx,
                    client.client_idx,
                    get_client_type(client.client_idx),
                    self.method,
                    dataset_name,
                    getattr(self.args, "model_name", ""),
                    group,
                    vals.get("alpha", ""),
                    vals.get("utility_gain", ""),
                    vals.get("safety_risk", ""),
                    vals.get("score", ""),
                    vals.get("param_count", ""),
                ])

    # ------------------------------------------------------------------
    # Task generation accuracy
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_text(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    @staticmethod
    def _normalize_num_str(s: str):
        s = (s or "").strip().replace(",", "")
        try:
            d = Decimal(s).normalize()
            v = format(d, "f")
            if "." in v:
                v = v.rstrip("0").rstrip(".")
            return v
        except (InvalidOperation, ValueError):
            return None

    @classmethod
    def _extract_gsm8k_gold(cls, answer: str):
        m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer or "")
        if m:
            return cls._normalize_num_str(m.group(1))
        nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", answer or "")
        return cls._normalize_num_str(nums[-1]) if nums else None

    @classmethod
    def _extract_last_number(cls, text: str):
        nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text or "")
        return cls._normalize_num_str(nums[-1]) if nums else None

    @staticmethod
    def _truncate_at_new_turn(text: str) -> str:
        for marker in ["\nUser:", "\nHuman:", "\n\nUser:", "\n\nHuman:"]:
            idx = text.find(marker)
            if idx > 0:
                text = text[:idx]
        return text.strip()

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        return re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()

    @staticmethod
    def _extract_option_letter(text: str):
        if not text:
            return None
        m = re.search(r"\b([A-E])\b", text.upper())
        return m.group(1) if m else None

    def _build_prompt_for_generation_eval(self, question: str, dataset_name: str = ""):
        messages = [{"role": "user", "content": question}]
        for kwargs in [{"enable_thinking": False}, {}]:
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **kwargs
                )
            except TypeError:
                continue
            except Exception:
                break
        return f"User: {question}\nAssistant: "

    def _generate_texts_for_task_eval_batch(self, items):
        if not items:
            return []

        prompts = [
            self._build_prompt_for_generation_eval(
                item.get("question", ""), item.get("dataset", "")
            )
            for item in items
        ]

        max_new_tokens = int(getattr(self.args, "task_gen_max_new_tokens", 128))

        old_side = getattr(self.tokenizer, "padding_side", "right")
        inputs, output_ids, gen_ids = None, None, None

        try:
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.args.max_seq_length
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model_trainer.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    max_length=None,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            gen_ids = output_ids[:, input_len:]
            raw_preds = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            return [self._strip_thinking_tags(self._truncate_at_new_turn(p)) for p in raw_preds]

        except RuntimeError as e:
            logging.warning(f"task generation batch eval OOM/runtime failed: {type(e).__name__}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return [""] * len(items)

        except Exception as e:
            logging.warning(f"task generation batch eval failed: {type(e).__name__}: {e}")
            return [""] * len(items)

        finally:
            self.tokenizer.padding_side = old_side
            if gen_ids is not None:
                del gen_ids
            if output_ids is not None:
                del output_ids
            if inputs is not None:
                del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _is_correct_generation(self, item, pred_text: str):
        dname = str(item.get("dataset", "")).lower()
        gold = str(item.get("answer", ""))

        if "gsm8k" in dname:
            g = self._extract_gsm8k_gold(gold)
            p = self._extract_gsm8k_gold(pred_text)
            if p is None:
                p = self._extract_last_number(pred_text)
            return g is not None and p is not None and g == p

        if dname in ("arc_challenge", "piqa") or "mmlu" in dname:
            g = self._extract_option_letter(gold)
            p = self._extract_option_letter(pred_text)
            return g is not None and p is not None and g == p

        return self._normalize_text(pred_text) == self._normalize_text(gold)

    def _prepare_task_eval_trace(self, round_idx):
        if not bool(getattr(self.args, "task_eval_dump", 1)):
            return None
        try:
            os.makedirs(self.args.output_dir, exist_ok=True)
            p = os.path.join(self.args.output_dir, f"task_eval_round_{round_idx}.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(f"[TaskEvalTrace] round={round_idx}\n")
                f.write(self._build_run_header())
            return p
        except Exception:
            return None

    def _evaluate_task_generation_accuracy(self, round_idx=None, qa_trace_path=None):
        if not self.task_eval_data or self.tokenizer is None:
            return None

        items = list(self.task_eval_data)
        target_ds = str(getattr(self.args, "task_gen_eval_target_dataset", "gsm8k")).lower()
        if target_ds != "all":
            filt = [x for x in items if str(x.get("dataset", "")).lower() == target_ds]
            if filt:
                items = filt
            else:
                logging.warning(
                    f"No task eval items found for target dataset '{target_ds}'. "
                    "Falling back to all available eval items."
                )
                target_ds = "all"

        max_n = getattr(self.args, "max_task_eval_gen_samples", 50)
        if max_n is not None and max_n < len(items):
            items = items[:max_n]

        if not items:
            return {"task_gen_acc": 0.0, "task_gen_correct": 0, "task_gen_total": 0, "task_gen_dataset": target_ds}

        trace_path = self._prepare_task_eval_trace(round_idx)
        max_c = int(getattr(self.args, "eval_dump_max_chars", 240))
        qa_max_c = int(getattr(self.args, "qa_dump_max_chars", 0))

        self.model_trainer.model.eval()
        correct, total = 0, 0
        bs = max(1, int(getattr(self.args, "task_gen_eval_batch_size", 8)))

        for s in range(0, len(items), bs):
            chunk = items[s:s + bs]
            preds = self._generate_texts_for_task_eval_batch(chunk)

            for j, (item, pred) in enumerate(zip(chunk, preds)):
                ok = self._is_correct_generation(item, pred)
                correct += int(ok)
                total += 1

                _append_trace(
                    trace_path,
                    f"[{s+j}] ds={item.get('dataset','')} ok={ok} "
                    f"gold={repr(_clip(str(item.get('answer','')), max_c))} "
                    f"pred={repr(_clip(pred, max_c))}"
                )
                _append_trace(
                    qa_trace_path,
                    "\n".join([
                        f"[TASK_QA][round={round_idx}][idx={s+j}] "
                        f"ds={item.get('dataset','')} ok={ok}",
                        "Q:",
                        _clip(str(item.get("question", "")), qa_max_c),
                        "A:",
                        _clip(pred, qa_max_c),
                        "GOLD:",
                        _clip(str(item.get("answer", "")), qa_max_c),
                        "-" * 80,
                    ])
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result = {
            "task_gen_acc": (correct / total) if total > 0 else 0.0,
            "task_gen_correct": correct,
            "task_gen_total": total,
            "task_gen_dataset": target_ds,
        }
        _append_trace(trace_path, f"[Summary] acc={result['task_gen_acc']:.4f} ({correct}/{total})")
        if trace_path:
            logging.info(f"Task eval trace: {trace_path}")
        return result

    # ------------------------------------------------------------------
    # Safety evaluation helpers
    # ------------------------------------------------------------------
    def _prepare_safety_trace_file(self, round_idx):
        if not bool(getattr(self.args, "safety_judge_dump", 1)):
            return None
        try:
            os.makedirs(self.args.output_dir, exist_ok=True)
            p = os.path.join(self.args.output_dir, f"safety_judge_round_{round_idx}.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(f"[SafetyJudgeTrace] round={round_idx}, judge={self.safety_judge_name}\n")
                f.write(self._build_run_header())
            return p
        except Exception:
            return None

    def _prepare_qa_trace_file(self):
        if not bool(getattr(self.args, "qa_dump", 1)):
            return None
        if self.qa_trace_path:
            return self.qa_trace_path
        try:
            os.makedirs(self.args.output_dir, exist_ok=True)
            filename = f"{self.args.method}_{self._get_result_filename()}"
            self.qa_trace_path = os.path.join(self.args.output_dir, filename)
            with open(self.qa_trace_path, "w", encoding="utf-8") as f:
                f.write(
                    f"[QATrace] stage={self.stage}, method={self.method}, "
                    f"judge={self.safety_judge_name}\n"
                )
                f.write(self._build_run_header())
            logging.info(f"QA trace: {self.qa_trace_path}")
            return self.qa_trace_path
        except Exception as e:
            logging.warning(f"prepare QA trace failed: {type(e).__name__}: {e}")
            return None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate(self, round_idx):
        def _std_ignore_none(vals):
            vals = [x for x in vals if isinstance(x, (int, float))]
            return float(np.std(vals)) if vals else None

        def _pick(d, *keys):
            for k in keys:
                if k in d:
                    return d[k]
            return None

        qa_trace_path = self._prepare_qa_trace_file()
        _append_trace(qa_trace_path, f"\n{'=' * 30} ROUND {round_idx} {'=' * 30}")

        if self.method == "safety_embed" and self.safety_embed_base_params is not None:
            self.model_trainer.set_safety_embed_params(self.safety_embed_base_params)

        metrics = self.model_trainer.test(self.test_loader, self.device, self.args)
        logging.info(
            f"Round {round_idx} | task_loss={metrics['test_loss']:.4f} "
            f"task_ppl={metrics['test_perplexity']:.2f} "
            f"task_token_acc={metrics.get('test_token_acc', 0.0):.4f}"
        )

        row = {
            "task_loss": metrics["test_loss"],
            "task_ppl": metrics["test_perplexity"],
            "task_token_acc": metrics.get("test_token_acc", 0.0),
        }

        if getattr(self.args, "task_gen_eval", True):
            gen_metrics = self._evaluate_task_generation_accuracy(
                round_idx=round_idx,
                qa_trace_path=qa_trace_path,
            )
            if gen_metrics is not None:
                row.update(gen_metrics)
                logging.info(
                    f"Round {round_idx} | task_gen_acc={gen_metrics['task_gen_acc']:.4f} "
                    f"({gen_metrics['task_gen_correct']}/{gen_metrics['task_gen_total']}) "
                    f"on {gen_metrics.get('task_gen_dataset', 'all')}"
                )

        trace_path = self._prepare_safety_trace_file(round_idx)
        dump_max = int(getattr(self.args, "eval_dump_max_chars", 240))

        if getattr(self.args, "safety_eval", True) and self.safety_eval_data and self.tokenizer:
            logging.info(f"--- Safety Evaluation (Round {round_idx}, judge={self.safety_judge_name}) ---")

            detail_mode = getattr(self.args, "safety_eval_detail_mode", "none")
            max_chars = int(getattr(self.args, "safety_eval_print_max_chars", 120))
            gen_bs = int(getattr(self.args, "safety_gen_batch_size", 8))
            judge_bs = int(getattr(self.args, "judge_batch_size", 8))
            qa_max_c = int(getattr(self.args, "qa_dump_max_chars", 0))

            need_per_client = self.method in (
                "local_safety_lora", "local_safety_adapter", "safety_embed"
            )
            judge_backend = self._get_safety_judge_backend()
            try:
                if need_per_client:
                    safety_results = self._evaluate_per_client_merged(
                        detail_mode, max_chars, gen_bs, judge_bs, trace_path, dump_max,
                        qa_trace_path, qa_max_c, judge_backend=judge_backend,
                    )
                else:
                    safety_results = evaluate_all_clients_safety(
                        self.model_trainer.model, self.tokenizer,
                        self.safety_eval_data, self.device,
                        judge_backend=judge_backend,
                        detail_mode=detail_mode, max_print_chars=max_chars,
                        gen_batch_size=gen_bs, judge_batch_size=judge_bs,
                        trace_path=trace_path, trace_max_chars=dump_max,
                        qa_trace_path=qa_trace_path, qa_trace_max_chars=qa_max_c,
                    )
            finally:
                self._release_safety_judge_backend()

            has_general = any(
                isinstance(k, int) and safety_results[k].get("general_total", 0) > 0
                for k in safety_results
            )

            if has_general:
                row["avg_general_client_asr"] = _pick(safety_results, "avg_general_client_asr", "avg_general_asr")
                row["avg_general_benign_accept"] = _pick(safety_results, "avg_general_benign_accept")
                row["avg_general_correctness"] = _pick(safety_results, "avg_general_correctness")

            row["avg_personal_client_asr"] = _pick(safety_results, "avg_personal_client_asr", "avg_personal_asr")
            row["avg_personal_benign_accept"] = _pick(safety_results, "avg_personal_benign_accept")
            row["avg_personal_correctness"] = _pick(safety_results, "avg_personal_correctness")

            row["avg_overall_client_asr"] = _pick(safety_results, "avg_overall_client_asr", "avg_overall_asr")
            row["avg_overall_benign_accept"] = _pick(safety_results, "avg_overall_benign_accept")
            row["avg_overall_correctness"] = _pick(safety_results, "avg_overall_correctness")

            if has_general:
                row["avg_general_asr"] = row["avg_general_client_asr"]
            row["avg_personal_asr"] = row["avg_personal_client_asr"]
            row["avg_overall_asr"] = row["avg_overall_client_asr"]

            client_asr_vals = []
            type_bucket = {}
            client_ids = sorted(k for k in safety_results if isinstance(k, int))

            for cid in client_ids:
                ctype = get_client_type(cid)
                r = safety_results[cid]

                g_asr = _pick(r, "general_client_asr", "general_asr")
                g_ba = _pick(r, "general_benign_accept")
                g_corr = _pick(r, "general_correctness")

                p_asr = _pick(r, "personal_client_asr", "personal_asr")
                p_ba = _pick(r, "personal_benign_accept")
                p_corr = _pick(r, "personal_correctness")

                o_asr = _pick(r, "overall_client_asr", "overall_asr")
                o_ba = _pick(r, "overall_benign_accept")
                o_corr = _pick(r, "overall_correctness")

                if has_general:
                    row[f"c{cid}({ctype})_gen_client_asr"] = g_asr
                    row[f"c{cid}({ctype})_gen_benign_accept"] = g_ba
                    row[f"c{cid}({ctype})_gen_correctness"] = g_corr

                row[f"c{cid}({ctype})_per_client_asr"] = p_asr
                row[f"c{cid}({ctype})_per_benign_accept"] = p_ba
                row[f"c{cid}({ctype})_per_correctness"] = p_corr

                row[f"c{cid}({ctype})_client_asr"] = o_asr
                row[f"c{cid}({ctype})_benign_accept"] = o_ba
                row[f"c{cid}({ctype})_correctness"] = o_corr

                if isinstance(o_asr, (int, float)):
                    client_asr_vals.append(o_asr)
                if isinstance(p_asr, (int, float)):
                    type_bucket.setdefault(ctype, []).append(p_asr)

            row["client_asr_worst"] = max(client_asr_vals) if client_asr_vals else None
            row["client_asr_std"] = _std_ignore_none(client_asr_vals)

            row["personal_worst_asr"] = row["client_asr_worst"]
            row["personal_std_asr"] = row["client_asr_std"]

            for ctype, vals in type_bucket.items():
                row[f"type_{ctype}_personal_asr"] = sum(vals) / len(vals)

        self.results["round"].append({"round_idx": round_idx, **row})

    def _evaluate_per_client_merged(self, detail_mode, max_chars,
                                    gen_bs, judge_bs, trace_path, dump_max,
                                    qa_trace_path=None, qa_max_chars=0,
                                    judge_backend=None):
        def _mean_ignore_none(vals):
            vals = [x for x in vals if isinstance(x, (int, float))]
            return (sum(vals) / len(vals)) if vals else None

        def _pick(d, *keys):
            for k in keys:
                if k in d:
                    return d[k]
            return None

        global_task_params = self.model_trainer.get_model_params()
        results = {}

        g_asr, g_ba, g_corr = [], [], []
        p_asr, p_ba, p_corr = [], [], []
        o_asr, o_ba, o_corr = [], [], []

        client_ids = sorted(self.safety_eval_data.keys())
        for cid in client_ids:
            client = self.client_list[cid]
            adapters_merged = False
            if self.method == "local_safety_lora":
                self.model_trainer.set_model_params(
                    client.get_merged_lora_params(global_task_params))
            elif self.method == "safety_embed":
                self.model_trainer.set_model_params(global_task_params)
                if client.safety_embed_params is not None:
                    self.model_trainer.set_safety_embed_params(client.safety_embed_params)
            elif self.method == "local_safety_adapter":
                self.model_trainer.enable_task_adapter_only()
                self.model_trainer.set_model_params(global_task_params)
                if client.safety_adapter_params is not None:
                    self.model_trainer.set_safety_adapter_params(client.safety_adapter_params)
                    self.model_trainer.enable_both_adapters()
                    adapters_merged = True

            try:
                r = evaluate_safety(
                    self.model_trainer.model, self.tokenizer,
                    self.safety_eval_data[cid], self.device,
                    judge_backend=judge_backend,
                    client_id=cid, detail_mode=detail_mode,
                    max_print_chars=max_chars,
                    gen_batch_size=gen_bs, judge_batch_size=judge_bs,
                    trace_path=trace_path, trace_max_chars=dump_max,
                    qa_trace_path=qa_trace_path, qa_trace_max_chars=qa_max_chars,
                )
            finally:
                if adapters_merged:
                    self.model_trainer.enable_task_adapter_only()

            results[cid] = r

            g_asr.append(_pick(r, "general_client_asr", "general_asr"))
            g_ba.append(_pick(r, "general_benign_accept"))
            g_corr.append(_pick(r, "general_correctness"))

            p_asr.append(_pick(r, "personal_client_asr", "personal_asr"))
            p_ba.append(_pick(r, "personal_benign_accept"))
            p_corr.append(_pick(r, "personal_correctness"))

            o_asr.append(_pick(r, "overall_client_asr", "overall_asr"))
            o_ba.append(_pick(r, "overall_benign_accept"))
            o_corr.append(_pick(r, "overall_correctness"))

            ctype = get_client_type(cid)
            logging.info(
                f"  Client {cid}({ctype}) | "
                f"overall_client_asr={_pick(r, 'overall_client_asr', 'overall_asr')} "
                f"overall_benign_accept={_pick(r, 'overall_benign_accept')} "
                f"overall_correctness={_pick(r, 'overall_correctness')}"
            )

        self.model_trainer.set_model_params(global_task_params)
        if self.method == "safety_embed" and self.safety_embed_base_params is not None:
            self.model_trainer.set_safety_embed_params(self.safety_embed_base_params)

        results["avg_general_client_asr"] = _mean_ignore_none(g_asr)
        results["avg_general_benign_accept"] = _mean_ignore_none(g_ba)
        results["avg_general_correctness"] = _mean_ignore_none(g_corr)

        results["avg_personal_client_asr"] = _mean_ignore_none(p_asr)
        results["avg_personal_benign_accept"] = _mean_ignore_none(p_ba)
        results["avg_personal_correctness"] = _mean_ignore_none(p_corr)

        results["avg_overall_client_asr"] = _mean_ignore_none(o_asr)
        results["avg_overall_benign_accept"] = _mean_ignore_none(o_ba)
        results["avg_overall_correctness"] = _mean_ignore_none(o_corr)

        results["avg_general_asr"] = results["avg_general_client_asr"]
        results["avg_personal_asr"] = results["avg_personal_client_asr"]
        results["avg_overall_asr"] = results["avg_overall_client_asr"]

        return results

    # ------------------------------------------------------------------
    # Results I/O
    # ------------------------------------------------------------------
    def _build_run_header(self):
        a = self.args
        dataset_cfg = getattr(a, "dataset", "")
        if isinstance(dataset_cfg, (list, tuple)):
            dataset_name = "+".join(str(x) for x in dataset_cfg)
        else:
            dataset_name = str(dataset_cfg)

        lines = [
            f"method={getattr(a, 'method', self.method)}",
            f"model_name={getattr(a, 'model_name', '')}",
            f"stage={getattr(a, 'stage', self.stage)}",
            f"dataset={dataset_name}",
            f"judge={self.safety_judge_name}",
            f"lora_download_gate={int(bool(getattr(a, 'lora_download_gate', 0)))}",
        ]
        if bool(getattr(a, "lora_download_gate", 0)):
            lines.extend([
                f"lora_gate_anchor_batches={getattr(a, 'lora_gate_anchor_batches', 1)}",
                f"lora_gate_mu={getattr(a, 'lora_gate_mu', 5.0)}",
                f"lora_gate_lambda={getattr(a, 'lora_gate_lambda', 5.0)}",
                f"lora_gate_group_mode={getattr(a, 'lora_gate_group_mode', 'module_path')}",
            ])
        return "\n".join(lines) + "\n\n"

    def _get_result_filename(self):
        a = self.args
        dataset_cfg = getattr(a, "dataset", "gsm8k")
        if isinstance(dataset_cfg, (list, tuple)):
            dataset_name = "+".join(str(x) for x in dataset_cfg)
        else:
            dataset_name = str(dataset_cfg)
        dataset_name = dataset_name.replace("/", "-").replace(" ", "")

        stage = getattr(a, "stage", "stage")
        name = (f"{stage}_{a.method}_{dataset_name}_c{a.client_num_in_total}"
                f"_r{a.comm_round}_lr{a.lr}_rank{a.lora_rank}"
                f"_safetylr{a.safety_lr}"
                f"_judge{getattr(a, 'safety_judge', 'rule')}")

        if a.method == "ewc":
            name += f"_lambda{a.ewc_lambda}"
        elif a.method == "lwf":
            name += f"_alpha{a.lwf_alpha}_T{a.lwf_temperature}"
        elif a.method == "replay":
            name += f"_ratio{a.replay_ratio}"
        elif a.method == "safety_embed":
            name += f"_{getattr(a, 'safety_embed_layers', 'layernorm')}"
            if getattr(a, "safety_embed_layers", "layernorm") in ("token_embed", "token_embed_norm"):
                name += f"_target{getattr(a, 'safety_embed_target_params', 3800000)}"
        elif a.method == "local_safety_adapter":
            name += f"_sarank{getattr(a, 'safety_adapter_rank', 4)}"
        elif a.method == "safelora":
            if getattr(a, "safelora_select_layers_type", "number") == "number":
                name += f"_proj{getattr(a, 'safelora_num_proj_layers', 10)}"
            else:
                name += f"_thrs{getattr(a, 'safelora_threshold', 0.5)}"

        if bool(getattr(a, "lora_download_gate", 0)):
            name += (
                f"_loragate_mu{getattr(a, 'lora_gate_mu', 5.0)}"
                f"_lambda{getattr(a, 'lora_gate_lambda', 5.0)}"
                f"_anchor{getattr(a, 'lora_gate_anchor_batches', 1)}"
                f"_{getattr(a, 'lora_gate_group_mode', 'module_path')}"
            )

        return name + ".txt"

    def _save_results(self):
        def _fmt(v):
            return "N/A" if v is None else f"{v:.4f}"

        os.makedirs(self.args.output_dir, exist_ok=True)
        path = os.path.join(self.args.output_dir, self._get_result_filename())

        with open(path, "w", encoding="utf-8") as f:
            f.write("[ResultSummary]\n")
            f.write(self._build_run_header())
            for row in self.results["round"]:
                ridx = row["round_idx"]
                f.write(f"Round {ridx}:\n")
                f.write(
                    f"  task_ppl={row['task_ppl']:.4f}, "
                    f"task_loss={row['task_loss']:.4f}, "
                    f"task_token_acc={row.get('task_token_acc', 0.0):.4f}\n"
                )

                if "task_gen_acc" in row:
                    f.write(
                        f"  task_gen_acc={row['task_gen_acc']:.4f}, "
                        f"task_gen_correct={int(row.get('task_gen_correct', 0))}, "
                        f"task_gen_total={int(row.get('task_gen_total', 0))}, "
                        f"task_gen_dataset={row.get('task_gen_dataset', 'all')}\n"
                    )

                if "avg_overall_client_asr" in row:
                    if "avg_general_client_asr" in row:
                        f.write(
                            f"  AVG_GENERAL: "
                            f"client_asr={_fmt(row.get('avg_general_client_asr'))}, "
                            f"benign_accept={_fmt(row.get('avg_general_benign_accept'))}, "
                            f"correctness={_fmt(row.get('avg_general_correctness'))}\n"
                        )

                    f.write(
                        f"  AVG_PERSONAL: "
                        f"client_asr={_fmt(row.get('avg_personal_client_asr'))}, "
                        f"benign_accept={_fmt(row.get('avg_personal_benign_accept'))}, "
                        f"correctness={_fmt(row.get('avg_personal_correctness'))}\n"
                    )

                    f.write(
                        f"  AVG_OVERALL: "
                        f"client_asr={_fmt(row.get('avg_overall_client_asr'))}, "
                        f"benign_accept={_fmt(row.get('avg_overall_benign_accept'))}, "
                        f"correctness={_fmt(row.get('avg_overall_correctness'))}\n"
                    )

                    f.write(
                        f"  client_asr_worst={_fmt(row.get('client_asr_worst'))}, "
                        f"client_asr_std={_fmt(row.get('client_asr_std'))}\n"
                    )

                # type-level
                for key in sorted(row.keys()):
                    if isinstance(key, str) and key.startswith("type_") and key.endswith("_personal_asr"):
                        f.write(f"  {key}={_fmt(row[key])}\n")

                per_client_keys = sorted(
                    k for k in row.keys()
                    if isinstance(k, str) and k.startswith("c") and (
                        k.endswith("_client_asr")
                        or k.endswith("_benign_accept")
                        or k.endswith("_correctness")
                    )
                )
                for k in per_client_keys:
                    f.write(f"  {k}={_fmt(row[k])}\n")

                f.write("\n")

    def _save_model(self, tag=None):
        os.makedirs(self.args.output_dir, exist_ok=True)
        save_model_name = getattr(self.args, "save_model_name", "")
        if tag:
            filename = f"{self.stage}_{self.method}_lora_{tag}.pt"
        elif save_model_name:
            filename = save_model_name
        else:
            filename = f"{self.stage}_{self.method}_lora_params.pt"
        save_path = os.path.join(self.args.output_dir, filename)
        torch.save(self.model_trainer.get_model_params(), save_path)
        logging.info(f"LoRA params saved to {save_path}")
