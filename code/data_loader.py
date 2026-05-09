import logging
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter, defaultdict


# ============================================================================
# Generic instruction-tuning dataset
# ============================================================================
class InstructDataset(Dataset):
    """Instruction-tuning dataset for causal LM.

    Each sample is a (question, answer) pair formatted with the model's chat
    template. Only assistant response tokens contribute to the loss.
    """

    def __init__(self, data, tokenizer, max_seq_length=512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = self._preprocess(data)

    def _build_prompt_text(self, question):
        prompt_messages = [{"role": "user", "content": question}]
        try:
            return self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return f"User: {question}\nAssistant: "

    def _encode_instruction_pair(self, question, answer):
        """Encode prompt and answer separately so labels cannot drift.

        Some chat templates, including Qwen3 with thinking disabled, do not
        make the full user+assistant template a byte-for-byte continuation of
        the generation prompt template. Building labels from two independently
        rendered templates can therefore expose prompt/special tokens to the
        loss and make perplexity meaningless. Here the model input is explicitly
        prompt tokens followed by answer tokens, and only answer tokens are
        labelled.
        """
        prompt_text = self._build_prompt_text(question)
        answer_text = str(answer)
        eos = self.tokenizer.eos_token or ""
        if eos and not answer_text.endswith(eos):
            answer_text = answer_text + eos

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        answer_ids = self.tokenizer(
            answer_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        if not answer_ids:
            answer_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []

        max_len = self.max_seq_length
        if len(answer_ids) >= max_len:
            answer_ids = answer_ids[:max_len]
            prompt_ids = []
        else:
            prompt_budget = max_len - len(answer_ids)
            if len(prompt_ids) > prompt_budget:
                prompt_ids = prompt_ids[-prompt_budget:]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + list(answer_ids)
        attention_mask = [1] * len(input_ids)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len
            labels += [-100] * pad_len

        import torch
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _preprocess(self, raw_data):
        processed = []
        for item in raw_data:
            question = item["question"]
            answer = item["answer"]
            processed.append(self._encode_instruction_pair(question, answer))
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================================
# Dataset-specific raw data loaders
# ============================================================================
def _load_gsm8k_raw():
    ds = load_dataset("openai/gsm8k", "main")
    train = [{"question": r["question"], "answer": r["answer"]} for r in ds["train"]]
    test = [{"question": r["question"], "answer": r["answer"]} for r in ds["test"]]
    return train, test


def _load_arc_challenge_raw():
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")

    def _format(row):
        choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        options = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
        q = f"{row['question']}\n{options}"
        a = row["answerKey"]
        return {"question": q, "answer": a}

    train = [_format(r) for r in ds["train"]]
    test = [_format(r) for r in ds["test"]]
    return train, test


def _load_mbpp_plus_raw():
    ds = load_dataset("google-research-datasets/mbpp", "sanitized")

    def _format(row):
        return {"question": row["prompt"], "answer": row["code"]}

    train_items, test_items = [], []
    if "train" in ds:
        train_items = [_format(r) for r in ds["train"]]
    if "test" in ds:
        test_items = [_format(r) for r in ds["test"]]
    if "prompt" in ds:
        train_items += [_format(r) for r in ds["prompt"]]
    return train_items, test_items


def _load_piqa_raw():
    ds = load_dataset("piqa")

    def _format(row):
        q = (
            f"{row['goal']}\n"
            f"A. {row['sol1']}\n"
            f"B. {row['sol2']}\n"
            f"Please answer with A or B."
        )
        a = "A" if int(row["label"]) == 0 else "B"
        return {"question": q, "answer": a}

    train = [_format(r) for r in ds["train"]]
    test = [_format(r) for r in ds["validation"]]
    return train, test


def _format_options(choices, labels=None):
    if labels is None:
        labels = [chr(ord("A") + i) for i in range(len(choices))]
    return "\n".join(f"{label}. {text}" for label, text in zip(labels, choices))


def _answer_to_letter(answer):
    if isinstance(answer, int):
        return chr(ord("A") + answer)
    s = str(answer).strip()
    if s.isdigit():
        return chr(ord("A") + int(s))
    return s[:1].upper() if s else ""


def _format_mmlu_row(row):
    choices = row.get("choices", [])
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    options = _format_options(choices, labels)
    subject = row.get("subject", row.get("category", "mmlu"))
    question = row.get("question", "")
    q = f"{question}\n{options}\nPlease answer with the correct option letter."
    return {
        "question": q,
        "answer": _answer_to_letter(row.get("answer", "")),
        "category": str(subject),
        "dataset": "mmlu",
    }


def _load_mmlu_eval_raw():
    """Load MMLU as a fallback evaluation set when a dataset has no test split."""
    last_err = None
    for spec in [("cais/mmlu", "all"), ("lukaemon/mmlu", "all")]:
        try:
            ds = load_dataset(*spec)
            for split in ["test", "validation", "dev"]:
                if split in ds:
                    items = [_format_mmlu_row(r) for r in ds[split]]
                    items = [x for x in items if x["question"] and x["answer"]]
                    if items:
                        logging.info(
                            f"MMLU fallback eval: loaded {len(items)} items from {spec[0]}/{spec[1]}:{split}"
                        )
                        return items
        except Exception as e:
            last_err = e
            logging.warning(f"Failed to load MMLU fallback {spec}: {type(e).__name__}: {e}")
    raise RuntimeError(f"Failed to load MMLU fallback eval data: {last_err}")


def _format_dolly_row(row):
    instruction = (row.get("instruction") or "").strip()
    context = (row.get("context") or "").strip()
    response = (row.get("response") or "").strip()
    category = (row.get("category") or "unknown").strip()

    if context:
        question = f"{instruction}\n\nContext:\n{context}"
    else:
        question = instruction

    return {
        "question": question,
        "answer": response,
        "category": category,
    }


def _balanced_sample_by_category(items, max_n, category_key="category"):
    if max_n is None or max_n >= len(items):
        return list(items)
    if max_n <= 0:
        return []

    groups = defaultdict(list)
    for item in items:
        groups[str(item.get(category_key, "unknown"))].append(item)

    if len(groups) <= 1:
        return list(items[:max_n])

    if max_n < len(groups):
        logging.warning(
            f"max_n={max_n} is smaller than category count={len(groups)}; "
            "not every category can be represented."
        )

    for bucket in groups.values():
        np.random.shuffle(bucket)

    selected = []
    keys = sorted(groups.keys())
    while len(selected) < max_n:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < max_n:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def _log_category_counts(name, items):
    counts = Counter(str(x.get("category", "unknown")) for x in items)
    if counts:
        logging.info(f"{name} categories: {dict(sorted(counts.items()))}")


def _load_dolly_15k_raw():
    ds = load_dataset("databricks/databricks-dolly-15k")

    train_split = "train" if "train" in ds else list(ds.keys())[0]
    train = [_format_dolly_row(r) for r in ds[train_split]]
    train = [x for x in train if x["question"] and x["answer"]]
    _log_category_counts("Dolly-15k train", train)

    eval_split = None
    for split in ["test", "validation", "eval"]:
        if split in ds:
            eval_split = split
            break

    if eval_split is not None:
        test = [_format_dolly_row(r) for r in ds[eval_split]]
        test = [x for x in test if x["question"] and x["answer"]]
        logging.info(f"Dolly-15k using built-in {eval_split} split for evaluation.")
    else:
        logging.info("Dolly-15k has no built-in test/validation split; using MMLU for evaluation.")
        test = _load_mmlu_eval_raw()

    return train, test


DATASET_LOADERS = {
    "gsm8k": _load_gsm8k_raw,
    "arc_challenge": _load_arc_challenge_raw,
    "mbpp_plus": _load_mbpp_plus_raw,
    "piqa": _load_piqa_raw,
    "dolly_15k": _load_dolly_15k_raw,
}

DATASET_ALIASES = {
    "databricks_dolly_15k": "dolly_15k",
    "databricks-dolly-15k": "dolly_15k",
    "dolly-15k": "dolly_15k",
}


def _normalize_dataset_names(dataset_name):
    if isinstance(dataset_name, (list, tuple)):
        dataset_names = [str(x).strip().lower() for x in dataset_name if str(x).strip()]
    elif isinstance(dataset_name, str):
        ds = dataset_name.strip().lower()
        if ds.lower() == "all":
            dataset_names = list(DATASET_LOADERS.keys())
        elif "," in ds:
            dataset_names = [x.strip() for x in ds.split(",") if x.strip()]
        else:
            dataset_names = [ds]
    else:
        raise ValueError(f"Unsupported dataset_name type: {type(dataset_name)}")

    if not dataset_names:
        raise ValueError("No task dataset specified.")

    dataset_names = [DATASET_ALIASES.get(d, d) for d in dataset_names]
    dataset_names = list(dict.fromkeys(dataset_names))

    unknown = [d for d in dataset_names if d not in DATASET_LOADERS]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}. Supported: {list(DATASET_LOADERS.keys())}")
    return dataset_names


# ============================================================================
# Main entry point
# ============================================================================
def load_task_data(dataset_name, client_num, batch_size, tokenizer,
                   max_seq_length=512, max_train_samples=None,
                   max_test_samples=None, return_eval_items=False):
    """Load one or multiple task datasets and partition across clients.

    Returns:
      - client_data_num
      - client_train_data
      - test_loader
      - (optional) raw test items with 'dataset' field for generation eval
    """
    if client_num <= 0:
        raise ValueError(f"client_num must be > 0, got {client_num}")

    dataset_names = _normalize_dataset_names(dataset_name)

    logging.info(f"Loading task datasets: {dataset_names}")
    train_data, test_data = [], []
    for dname in dataset_names:
        tr, te = DATASET_LOADERS[dname]()
        tr = [{**x, "dataset": x.get("dataset", dname)} for x in tr]
        te = [{**x, "dataset": x.get("dataset", dname)} for x in te]
        train_data.extend(tr)
        test_data.extend(te)
        logging.info(f"  - {dname}: {len(tr)} train, {len(te)} test")

    if max_train_samples is not None and max_train_samples < len(train_data):
        if any("category" in x for x in train_data):
            train_data = _balanced_sample_by_category(train_data, max_train_samples)
            _log_category_counts("Sampled task train", train_data)
        else:
            train_data = train_data[:max_train_samples]
    if max_test_samples is not None and max_test_samples < len(test_data):
        if any("category" in x for x in test_data):
            test_data = _balanced_sample_by_category(test_data, max_test_samples)
            _log_category_counts("Sampled task test", test_data)
        else:
            test_data = test_data[:max_test_samples]

    logging.info(f"Merged task data: {len(train_data)} train, {len(test_data)} test")

    np.random.shuffle(train_data)

    client_data_num = {}
    client_train_data = {}

    all_indices = np.arange(len(train_data))
    split_indices = np.array_split(all_indices, client_num)

    for i, idx_arr in enumerate(split_indices):
        idx_list = idx_arr.tolist()
        local_raw = [train_data[j] for j in idx_list]
        client_dataset = InstructDataset(local_raw, tokenizer, max_seq_length)
        shuffle_flag = len(client_dataset) > 0
        client_train_data[i] = DataLoader(
            client_dataset, batch_size=batch_size, shuffle=shuffle_flag, drop_last=False
        )
        client_data_num[i] = len(local_raw)
        logging.info(f"  Client {i}: {client_data_num[i]} samples")

    test_dataset = InstructDataset(test_data, tokenizer, max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if return_eval_items:
        return client_data_num, client_train_data, test_loader, test_data
    return client_data_num, client_train_data, test_loader


# Backward compatibility alias
def load_gsm8k(client_num, batch_size, tokenizer, max_seq_length=512,
               max_train_samples=None, max_test_samples=None, return_eval_items=False):
    return load_task_data(
        "gsm8k", client_num, batch_size, tokenizer,
        max_seq_length, max_train_samples, max_test_samples,
        return_eval_items=return_eval_items
    )
