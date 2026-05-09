import os
import argparse
import logging
import random

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, TaskType

from data_loader import load_task_data
from safety_data_loader import load_safety_data
from my_model_trainer_lm import MyModelTrainer
from fedavg_api import FedAvgAPI
from modelscope_utils import resolve_modelscope_path


def add_args(parser):
    # ---- Model ----
    parser.add_argument(
        "--model_name", type=str,
        default="Qwen/Qwen3-8B",
        help="ModelScope model id or local path"
    )
    parser.add_argument(
        "--judge_model_name", type=str,
        default="/Llama-Guard-3-8B",
        help="ModelScope local judge model dir or model id"
    )
    parser.add_argument(
        "--modelscope_cache_root", type=str,
        default="~/.cache/modelscope",
        help="ModelScope cache root"
    )
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument(
        "--dataset", type=str, nargs="+", default=["dolly_15k"],
        help="Task dataset(s): gsm8k / dolly_15k / arc_challenge / piqa / mbpp_plus / all"
    )

    # ---- LoRA ----
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules", type=str, nargs="+",
        default=["q_proj", "v_proj"]
    )

    # ---- Training ----
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=5,
                        help="Local training epochs per round")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.1)

    # ---- Federated ----
    parser.add_argument("--client_num_in_total", type=int, default=5)
    parser.add_argument("--client_num_per_round", type=int, default=3)
    parser.add_argument("--comm_round", type=int, default=20)

    # ---- System ----
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./output")

    # ---- Stage control (stage1 + stage3 only) ----
    parser.add_argument(
        "--stage", type=str, default="stage3_personal",
        choices=["stage1_benign", "stage3_personal"],
        help="Experiment stage"
    )
    parser.add_argument("--eval_only", action="store_true",
                        help="Only run evaluation (pre-eval), skip training")
    parser.add_argument("--save_model_name", type=str, default="",
                        help="Custom output model filename")
    parser.add_argument("--init_lora_path", type=str, default=None,
                        help="Legacy option; ignored in stage3_personal, which now trains from the base model.")

    pre_eval_group = parser.add_mutually_exclusive_group()
    pre_eval_group.add_argument("--eval_before_train", dest="eval_before_train",
                                action="store_true",
                                help="Run round=-1 evaluation before training")
    pre_eval_group.add_argument("--no_eval_before_train", dest="eval_before_train",
                                action="store_false",
                                help="Disable pre-training evaluation")
    parser.set_defaults(eval_before_train=True)

    # ---- Data quantity ----
    parser.add_argument("--max_task_train_samples", type=int, default=2500)
    parser.add_argument("--max_task_test_samples", type=int, default=200)
    parser.add_argument("--max_personal_safety_train", type=int, default=300)
    parser.add_argument("--max_personal_safety_eval", type=int, default=150)

    # ---- Personalized safety focus ----
    skip_gen_group = parser.add_mutually_exclusive_group()
    skip_gen_group.add_argument("--skip_general_safety", dest="skip_general_safety",
                                action="store_true",
                                help="Skip general safety data; only use personalized safety")
    skip_gen_group.add_argument("--no_skip_general_safety", dest="skip_general_safety",
                                action="store_false")
    parser.set_defaults(skip_general_safety=True)
    parser.add_argument("--max_general_safety_train", type=int, default=300)
    parser.add_argument("--max_general_safety_eval", type=int, default=150)

    # ---- Task generation accuracy eval ----
    task_eval_group = parser.add_mutually_exclusive_group()
    task_eval_group.add_argument("--task_gen_eval", dest="task_gen_eval", action="store_true")
    task_eval_group.add_argument("--no_task_gen_eval", dest="task_gen_eval", action="store_false")
    parser.set_defaults(task_gen_eval=True)

    parser.add_argument("--max_task_eval_gen_samples", type=int, default=500)
    parser.add_argument("--task_gen_eval_target_dataset", type=str, default="mmlu",
                        help="gsm8k / mmlu / dolly_15k / arc_challenge / piqa / mbpp_plus / all")
    parser.add_argument("--task_gen_eval_batch_size", type=int, default=4)
    parser.add_argument("--task_gen_max_new_tokens", type=int, default=128)

    # ---- Safety evaluation ----
    
    safety_group = parser.add_mutually_exclusive_group()
    safety_group.add_argument("--safety_eval", dest="safety_eval", action="store_true")
    safety_group.add_argument("--no_safety_eval", dest="safety_eval", action="store_false")
    parser.set_defaults(safety_eval=True)

    # ---- Safety judge ----
    parser.add_argument("--safety_judge", type=str, default="llamaguard",
                        choices=["rule", "llamaguard"],
                        help="Safety judge backend")
    parser.add_argument("--judge_max_input_len", type=int, default=2048)
    parser.add_argument("--judge_max_new_tokens", type=int, default=8)
    parser.add_argument("--judge_batch_size", type=int, default=4)
    parser.add_argument("--judge_debug_samples", type=int, default=0)
    parser.add_argument("--judge_unknown_as_unsafe", type=int, default=0)

    # ---- Method (stage3 methods) ----
    parser.add_argument(
        "--method", type=str, default="baseline",
        choices=["baseline", "local_safety_lora",
                 "ewc", "lwf", "replay", "safety_embed",
                 "local_safety_adapter", "safelora"]
    )
    parser.add_argument("--safety_lr", type=float, default=1e-4)
    parser.add_argument("--safety_epochs", type=int, default=1)
    parser.add_argument("--ewc_lambda", type=float, default=100.0)
    parser.add_argument("--lwf_alpha", type=float, default=1.0)
    parser.add_argument("--lwf_temperature", type=float, default=2.0)
    parser.add_argument("--replay_ratio", type=float, default=0.3)
    parser.add_argument("--safety_embed_layers", type=str, default="token_embed_norm",
                        choices=["layernorm", "bias", "both",
                                 "token_embed", "token_embed_norm"])
    parser.add_argument("--safety_embed_target_params", type=int, default=3800000,
                        help="Auto-select safety token embedding rows up to this effective parameter budget.")
    parser.add_argument("--safety_embed_token_count", type=int, default=0,
                        help="Explicit number of safety token embedding rows; 0=auto from target params.")
    parser.add_argument("--safety_adapter_rank", type=int, default=4)
    parser.add_argument("--safelora_base_model_path", type=str, default=None,
                        help="Unaligned/base model path used by SafeLoRA.")
    parser.add_argument("--safelora_aligned_model_path", type=str, default=None,
                        help="Aligned/chat model path used by SafeLoRA; defaults to --model_name.")
    parser.add_argument("--safelora_select_layers_type", type=str, default="number",
                        choices=["threshold", "number"])
    parser.add_argument("--safelora_threshold", type=float, default=0.5)
    parser.add_argument("--safelora_num_proj_layers", type=int, default=10)
    parser.add_argument("--safelora_device", type=str, default="cpu",
                        help="Device for SafeLoRA projection matrix construction/application.")

    # ---- Safety-aware LoRA download gate ----
    parser.add_argument("--lora_download_gate", type=int, default=0,
                        help="1=gate server LoRA downloads using utility/safety anchors; 0=standard FedAvg download.")
    parser.add_argument("--lora_gate_anchor_batches", type=int, default=1,
                        help="Number of local utility/safety anchor batches used for the two gate backward passes.")
    parser.add_argument("--lora_gate_mu", type=float, default=5.0,
                        help="Weight for normalized utility gain in the LoRA download gate.")
    parser.add_argument("--lora_gate_lambda", type=float, default=5.0,
                        help="Weight for normalized safety risk in the LoRA download gate.")
    parser.add_argument("--lora_gate_alpha_min", type=float, default=0.0,
                        help="Lower bound for per-module download gate alpha.")
    parser.add_argument("--lora_gate_alpha_max", type=float, default=1.0,
                        help="Upper bound for per-module download gate alpha.")
    parser.add_argument("--lora_gate_group_mode", type=str, default="module_path",
                        choices=["module_path", "module_type"],
                        help="module_path gates each layer/module separately; module_type shares gates across q_proj/v_proj/etc.")
    parser.add_argument("--lora_gate_eps", type=float, default=1e-8,
                        help="Numerical epsilon for normalized gate inner products.")
    parser.add_argument("--lora_gate_log_modules", type=int, default=0,
                        help="1=log the lowest-alpha LoRA gate groups each client round.")
    parser.add_argument("--lora_gate_alpha_dump", type=int, default=1,
                        help="1=write per-round/client/module LoRA gate alpha CSV when lora_download_gate is enabled.")

    # ---- Eval dump / Logging ----
    parser.add_argument("--task_eval_dump", type=int, default=1,
                        help="1=write per-sample task eval trace file, 0=disable")
    parser.add_argument("--safety_judge_dump", type=int, default=1,
                        help="1=write per-sample safety judge trace file, 0=disable")
    parser.add_argument("--qa_dump", type=int, default=1,
                        help="1=write task/safety question-answer trace file, 0=disable")
    parser.add_argument("--qa_dump_max_chars", type=int, default=0,
                        help="Max chars for each question/answer in QA trace; 0=unlimited")
    parser.add_argument("--eval_dump_max_chars", type=int, default=240,
                        help="Max chars for prompt/response in dump files")

    parser.add_argument("--safety_eval_detail_mode", type=str, default="none",
                        choices=["none", "unsafe", "all"])
    parser.add_argument("--safety_eval_print_max_chars", type=int, default=120)
    parser.add_argument("--safety_gen_batch_size", type=int, default=4)

    parser.add_argument("--eval_every_rounds", type=int, default=5,
                        help="Run evaluation every N rounds (always eval on last round).")
    parser.add_argument("--save_every_rounds", type=int, default=5,
                        help="Save LoRA checkpoint every N rounds (0=only save at end).")
    parser.add_argument("--log_banner_width", type=int, default=90)

    return parser


def resolve_dataset_arg(dataset_arg):
    if isinstance(dataset_arg, list):
        if len(dataset_arg) == 1:
            return dataset_arg[0]
        return dataset_arg
    return dataset_arg


def validate_stage_method(args):
    if args.stage == "stage1_benign":
        if args.method != "baseline":
            logging.warning("[stage1_benign] usually use method=baseline.")
    elif args.stage == "stage3_personal":
        if args.method not in ("local_safety_adapter", "local_safety_lora",
                               "ewc", "lwf", "replay", "safety_embed",
                               "safelora"):
            logging.warning("[stage3_personal] expected a personalized method.")
        if not args.init_lora_path:
            logging.info("[stage3_personal] --init_lora_path omitted; training from the base model LoRA init.")
    if args.method == "safelora" and not args.safelora_base_model_path:
        raise ValueError("[safelora] --safelora_base_model_path is required.")


def load_tokenizer_with_fallback(model_dir):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except Exception as e:
        logging.warning(f"AutoTokenizer (fast) failed: {type(e).__name__}: {e}")

    return AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, use_fast=False)


def load_causallm_with_fallback(model_dir, torch_dtype):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = add_args(argparse.ArgumentParser(
        description="FedLoRA - Federated LoRA Fine-tuning"))
    args = parser.parse_args()
    validate_stage_method(args)

    # Resolve ModelScope local usable dirs
    args.model_name = resolve_modelscope_path(
        args.model_name, cache_root=args.modelscope_cache_root
    )
    if args.method == "safelora":
        args.safelora_base_model_path = resolve_modelscope_path(
            args.safelora_base_model_path, cache_root=args.modelscope_cache_root
        )
        if args.safelora_aligned_model_path:
            args.safelora_aligned_model_path = resolve_modelscope_path(
                args.safelora_aligned_model_path, cache_root=args.modelscope_cache_root
            )
        else:
            args.safelora_aligned_model_path = args.model_name
    if str(args.safety_judge).lower() == "llamaguard":
        args.judge_model_name = resolve_modelscope_path(
            args.judge_model_name, cache_root=args.modelscope_cache_root
        )

    logger.info(f"[Resolved] base model dir: {args.model_name}")
    if args.method == "safelora":
        logger.info(f"[Resolved] SafeLoRA base model dir: {args.safelora_base_model_path}")
        logger.info(f"[Resolved] SafeLoRA aligned model dir: {args.safelora_aligned_model_path}")
    logger.info(f"[Resolved] judge model dir: {args.judge_model_name}")
    logger.info(args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Tokenizer ----
    tokenizer = load_tokenizer_with_fallback(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Task data ----
    dataset_cfg = resolve_dataset_arg(args.dataset)
    args.dataset = dataset_cfg

    client_data_num, client_train_data, test_loader, task_eval_data = load_task_data(
        dataset_cfg, args.client_num_in_total, args.batch_size, tokenizer,
        args.max_seq_length,
        max_train_samples=args.max_task_train_samples,
        max_test_samples=args.max_task_test_samples,
        return_eval_items=True,
    )
    dataset = [client_data_num, client_train_data, test_loader]

    # ---- Safety data ----
    safety_eval_data = None
    safety_train_data = None

    need_safety = (
        args.safety_eval
        or args.method not in ("baseline",)
        or args.stage == "stage3_personal"
    )

    if need_safety:
        skip_gen = getattr(args, "skip_general_safety", True)
        logging.info(
            f"Loading safety data ({'personalized only' if skip_gen else 'general + personalized'})..."
        )
        safety_train_data, safety_eval_data = load_safety_data(
            args.client_num_in_total, args.batch_size, tokenizer,
            args.max_seq_length,
            max_general_safety_train=args.max_general_safety_train,
            max_general_safety_eval=args.max_general_safety_eval,
            max_personal_safety_train=args.max_personal_safety_train,
            max_personal_safety_eval=args.max_personal_safety_eval,
            skip_general=skip_gen,
        )

    # ---- Base model ----
    logger.info(f"Loading base model: {args.model_name}")
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    model = load_causallm_with_fallback(args.model_name, torch_dtype=dtype)

    # ---- LoRA ----
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Train / Eval ----
    model_trainer = MyModelTrainer(model)
    fedavg = FedAvgAPI(
        dataset, device, args, model_trainer,
        safety_eval_data=safety_eval_data,
        safety_train_data=safety_train_data,
        tokenizer=tokenizer,
        task_eval_data=task_eval_data,
    )
    fedavg.train()
