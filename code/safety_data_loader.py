"""Safety data loader with client-type mapping for new 5-category safety schema.

Expected personalized files (under safety_data/personalized):
  - enterprise_general_safety.json
  - financial_safety.json
  - legal_compliance.json
  - regulated_healthcare.json
  - youth_safety.json

Each JSON can contain:
  - train: list[dict]
  - test / eval / eva: list[dict]
"""

import json
import logging
import os
import torch
from torch.utils.data import Dataset, DataLoader

# ---- New client type system (your updated 5 categories) ----
CLIENT_TYPES = [
    "enterprise_general_safety",
    "financial_safety",
    "legal_compliance",
    "regulated_healthcare",
    "youth_safety",
]


def get_client_type(client_idx):
    """Map client index to a type name (round-robin)."""
    return CLIENT_TYPES[client_idx % len(CLIENT_TYPES)]


class SafetyDataset(Dataset):
    """Safety alignment dataset for causal LM instruction tuning."""

    def __init__(self, data, tokenizer, max_seq_length=512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = self._preprocess(data)

    def _build_prompt_text(self, prompt):
        prompt_messages = [{"role": "user", "content": prompt}]
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
            return f"User: {prompt}\nAssistant: "

    def _encode_pair(self, prompt, response):
        prompt_text = self._build_prompt_text(prompt)
        response_text = str(response)
        eos = self.tokenizer.eos_token or ""
        if eos and not response_text.endswith(eos):
            response_text = response_text + eos

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        response_ids = self.tokenizer(
            response_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        if not response_ids:
            response_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []

        max_len = self.max_seq_length
        if len(response_ids) >= max_len:
            response_ids = response_ids[:max_len]
            prompt_ids = []
        else:
            prompt_budget = max_len - len(response_ids)
            if len(prompt_ids) > prompt_budget:
                prompt_ids = prompt_ids[-prompt_budget:]

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + list(response_ids)
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

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _preprocess(self, raw_data):
        processed = []

        for item in raw_data:
            prompt = item.get("prompt", "")

            response = item.get("response", None)
            if response is None:
                response = item.get("target_response", None)
            if response is None:
                response = item.get("expected_behavior", "")

            if not prompt:
                continue

            processed.append(self._encode_pair(prompt, response))

        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _truncate(lst, max_n):
    if max_n is not None and max_n < len(lst):
        return lst[:max_n]
    return lst


def _get_eval_items(raw):
    """Return the available evaluation split from supported field names."""
    if isinstance(raw.get("eval", None), list):
        return raw["eval"]
    if isinstance(raw.get("eva", None), list):
        return raw["eva"]
    if isinstance(raw.get("test", None), list):
        return raw["test"]
    return []


def _load_general_dir(general_dir, max_train=None, max_eval=None):
    """Load all general safety files without assuming fixed category names."""
    train_items, eval_items = [], []

    if not os.path.isdir(general_dir):
        return train_items, eval_items

    files = sorted([x for x in os.listdir(general_dir) if x.endswith(".json")])
    for fn in files:
        fpath = os.path.join(general_dir, fn)
        try:
            raw = _load_json(fpath)
            train_items.extend(raw.get("train", []))
            eval_items.extend(_get_eval_items(raw))
        except Exception as e:
            logging.warning(f"Failed to load general file {fpath}: {type(e).__name__}: {e}")

    train_items = _truncate(train_items, max_train)
    eval_items = _truncate(eval_items, max_eval)
    return train_items, eval_items


def load_safety_data(
    client_num, batch_size, tokenizer, max_seq_length=512,
    safety_data_dir=None,
    max_general_safety_train=None,
    max_general_safety_eval=None,
    max_personal_safety_train=None,
    max_personal_safety_eval=None,
    skip_general=False,
):
    """Load general + personalized safety data for every client.

    Returns:
        client_safety_train_data: dict client_idx -> DataLoader
        client_safety_eval_data: dict client_idx -> {"general": [...], "personalized": [...]}
    """
    if safety_data_dir is None:
        safety_data_dir = os.path.join(os.path.dirname(__file__), "safety_data")

    general_dir = os.path.join(safety_data_dir, "general")
    personal_dir = os.path.join(safety_data_dir, "personalized")

    # 1) General safety
    general_train_items = []
    general_eval_items = []
    if skip_general:
        logging.info("General safety: SKIPPED (--skip_general_safety)")
    else:
        general_train_items, general_eval_items = _load_general_dir(
            general_dir,
            max_train=max_general_safety_train,
            max_eval=max_general_safety_eval
        )
        logging.info(
            f"General safety: {len(general_train_items)} train, {len(general_eval_items)} eval"
        )

    # 2) Personalized safety per type
    personal_cache = {}  # type_name -> (train_items, eval_items)
    for tname in CLIENT_TYPES:
        fpath = os.path.join(personal_dir, f"{tname}.json")
        if not os.path.exists(fpath):
            logging.warning(f"Personalized safety file not found: {fpath}")
            personal_cache[tname] = ([], [])
            continue

        try:
            raw = _load_json(fpath)
            t = _truncate(raw.get("train", []), max_personal_safety_train)
            e = _truncate(_get_eval_items(raw), max_personal_safety_eval)
            personal_cache[tname] = (t, e)
            logging.info(f"  Personalized [{tname}]: {len(t)} train, {len(e)} eval")
        except Exception as e:
            logging.warning(f"Failed to load personalized file {fpath}: {type(e).__name__}: {e}")
            personal_cache[tname] = ([], [])

    # 3) Assemble per-client
    client_safety_train_data = {}
    client_safety_eval_data = {}

    for client_idx in range(client_num):
        ctype = get_client_type(client_idx)
        p_train, p_eval = personal_cache.get(ctype, ([], []))

        combined_train = general_train_items + p_train
        if combined_train:
            ds = SafetyDataset(combined_train, tokenizer, max_seq_length)
            client_safety_train_data[client_idx] = DataLoader(
                ds, batch_size=batch_size, shuffle=True, drop_last=False
            )

        client_safety_eval_data[client_idx] = {
            "general": list(general_eval_items),
            "personalized": list(p_eval),
        }

        logging.info(
            f"  Client {client_idx} ({ctype}): "
            f"{len(combined_train)} safety train, "
            f"{len(general_eval_items)}g+{len(p_eval)}p safety eval"
        )

    return client_safety_train_data, client_safety_eval_data
