import os
import glob
import logging
from typing import Optional, List, Tuple

DEFAULT_MS_CACHE = ".cache/modelscope/hub"


def _has_file(d: str, name: str) -> bool:
    return os.path.exists(os.path.join(d, name))


def _has_hf_weights(d: str) -> bool:
    # single-file
    if _has_file(d, "model.safetensors") or _has_file(d, "pytorch_model.bin"):
        return True
    # index-file (sharded)
    if _has_file(d, "model.safetensors.index.json") or _has_file(d, "pytorch_model.bin.index.json"):
        return True
    # shards
    if glob.glob(os.path.join(d, "*.safetensors")):
        return True
    if glob.glob(os.path.join(d, "pytorch_model-*.bin")):
        return True
    return False


def _has_config(d: str) -> bool:
    return _has_file(d, "config.json")


def _has_tokenizer(d: str) -> bool:
    return (
        _has_file(d, "tokenizer_config.json")
        or _has_file(d, "tokenizer.json")
        or _has_file(d, "tokenizer.model")
    )


def _has_raw_llama_weights(d: str) -> bool:
    # Meta raw format (not directly HF-loadable)
    return bool(glob.glob(os.path.join(d, "consolidated.*.pth"))) and _has_file(d, "params.json")


def _score_dir(d: str, prefer_snapshots: bool = True) -> int:
    score = 0
    if _has_config(d):
        score += 10
    if _has_tokenizer(d):
        score += 3
    if _has_hf_weights(d):
        score += 20

    # prefer snapshots subtree (usually real payload)
    if prefer_snapshots and "/snapshots/" in d.replace("\\", "/"):
        score += 5

    # deeper path often more specific than repo root
    depth = d.count(os.sep)
    score += min(depth, 20)
    return score


def _walk_candidate_dirs(root: str) -> Tuple[List[str], List[str]]:
    hf_dirs, raw_dirs = [], []
    for cur, _, _ in os.walk(root):
        if _has_config(cur):
            if _has_hf_weights(cur):
                hf_dirs.append(cur)
            elif _has_raw_llama_weights(cur):
                raw_dirs.append(cur)
    return hf_dirs, raw_dirs


def _pick_best(dirs: List[str]) -> Optional[str]:
    if not dirs:
        return None
    scored = [(_score_dir(d), os.path.getmtime(d), d) for d in dirs]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


def _resolve_from_repo_root(repo_root: str) -> str:
    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"repo_root not found: {repo_root}")

    # 1) Prefer snapshots subtree first
    snapshots_dir = os.path.join(repo_root, "snapshots")
    if os.path.isdir(snapshots_dir):
        hf_dirs, raw_dirs = _walk_candidate_dirs(snapshots_dir)
        best_hf = _pick_best(hf_dirs)
        if best_hf:
            return best_hf
        if raw_dirs:
            raise RuntimeError(
                f"Found raw llama weights under snapshots (consolidated.*.pth), "
                f"but no HF weights in: {repo_root}\n"
                f"Please convert to HF format first."
            )

    # 2) Fallback to whole repo
    hf_dirs, raw_dirs = _walk_candidate_dirs(repo_root)
    best_hf = _pick_best(hf_dirs)
    if best_hf:
        return best_hf

    if raw_dirs:
        raise RuntimeError(
            f"Found raw llama weights (consolidated.*.pth) but no HF weights in: {repo_root}\n"
            f"Please convert to HF format first."
        )

    raise FileNotFoundError(
        f"No HF-loadable model directory found under: {repo_root}\n"
        f"Need config.json + model.safetensors / pytorch_model*.bin (or index json)."
    )


def resolve_modelscope_path(model_name_or_path: str, cache_root: str = DEFAULT_MS_CACHE) -> str:
    s = (model_name_or_path or "").strip()
    if not s:
        raise ValueError("Empty model_name_or_path")

    # A) direct local path
    if os.path.isdir(s):
        resolved = _resolve_from_repo_root(s)
        logging.info(f"[ModelScopeResolve] local: {s} -> {resolved}")
        return resolved

    # B) model id -> cache lookup
    cache_root = os.path.expanduser(cache_root)
    repo_dir = os.path.join(cache_root, "models", *s.split("/"))
    if os.path.isdir(repo_dir):
        resolved = _resolve_from_repo_root(repo_dir)
        logging.info(f"[ModelScopeResolve] cache: {s} -> {resolved}")
        return resolved

    raise FileNotFoundError(
        f"Cannot resolve ModelScope model '{s}'. Tried: {repo_dir}. "
        f"Please pass local path or run: modelscope download --model {s}"
    )