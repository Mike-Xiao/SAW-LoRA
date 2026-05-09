import gc
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


@dataclass
class SafeLoRAProjectionStats:
    projected_layers: int
    candidate_layers: int
    cosine_threshold: float
    mean_pdst: Optional[float]


class SafeLoRAProjector:
    """SafeLoRA projection for PEFT LoRA adapters.

    SafeLoRA projects selected LoRA_B matrices onto the alignment subspace
    induced by W_aligned - W_base for the corresponding backbone layer. This
    class adapts the reference implementation to the existing federated setup:
    projection matrices are built once and then applied to each client's local
    LoRA update after local task training.
    """

    def __init__(
        self,
        base_model_path: str,
        aligned_model_path: str,
        select_layers_type: str = "number",
        threshold: float = 0.5,
        num_proj_layers: int = 10,
        device: str = "cpu",
    ):
        self.base_model_path = base_model_path
        self.aligned_model_path = aligned_model_path
        self.select_layers_type = select_layers_type
        self.threshold = threshold
        self.num_proj_layers = num_proj_layers
        self.device = torch.device(device)
        self.project_matrices: Dict[str, torch.Tensor] = {}
        self._initialized = False

    def initialize(self, peft_model: torch.nn.Module):
        """Build projection matrices keyed by LoRA_B parameter name."""
        if self._initialized:
            return

        logging.info(
            "SafeLoRA init: loading base/aligned models on CPU for projection matrices."
        )
        base_model = self._load_reference_model(self.base_model_path)
        aligned_model = self._load_reference_model(self.aligned_model_path)

        base_params = dict(base_model.named_parameters())
        aligned_params = dict(aligned_model.named_parameters())

        missing = []
        for lora_b_name, weight_name in self._iter_lora_b_to_weight_names(peft_model):
            base_weight = self._lookup_param(base_params, weight_name)
            aligned_weight = self._lookup_param(aligned_params, weight_name)
            if base_weight is None or aligned_weight is None:
                missing.append((lora_b_name, weight_name))
                continue
            if base_weight.shape != aligned_weight.shape:
                raise ValueError(
                    "SafeLoRA base/aligned shape mismatch for "
                    f"{weight_name}: {tuple(base_weight.shape)} vs {tuple(aligned_weight.shape)}"
                )

            delta = (aligned_weight.detach().float() - base_weight.detach().float()).to(
                self.device
            )
            norm = torch.norm(delta)
            if norm.item() == 0.0:
                logging.warning(
                    f"SafeLoRA skip zero alignment delta for {weight_name}."
                )
                continue
            project_matrix = torch.mm(delta, delta.t()) / norm
            self.project_matrices[lora_b_name] = project_matrix.detach().cpu()

        if missing:
            preview = ", ".join(f"{a}->{b}" for a, b in missing[:3])
            logging.warning(
                "SafeLoRA could not match %d LoRA layers to base/aligned weights. "
                "First missing: %s",
                len(missing),
                preview,
            )

        del base_params, aligned_params, base_model, aligned_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._initialized = True
        logging.info(
            "SafeLoRA init done: %d projection matrices cached.",
            len(self.project_matrices),
        )
        if not self.project_matrices:
            raise ValueError(
                "SafeLoRA initialized zero projection matrices. Check target modules "
                "and base/aligned model paths."
            )

    def project_model(self, peft_model: torch.nn.Module) -> SafeLoRAProjectionStats:
        """Project selected LoRA_B weights in-place."""
        if not self._initialized:
            raise RuntimeError("SafeLoRAProjector.initialize() must be called first.")

        params = dict(peft_model.named_parameters())
        candidates = []
        for lora_b_name, project_matrix in self.project_matrices.items():
            if lora_b_name not in params:
                continue
            lora_a_name = self._to_lora_a_name(lora_b_name)
            if lora_a_name not in params:
                logging.warning(f"SafeLoRA missing paired LoRA_A for {lora_b_name}.")
                continue

            lora_b = params[lora_b_name].detach().float()
            lora_a = params[lora_a_name].detach().float()
            p = project_matrix.to(lora_b.device)

            projected_b = torch.mm(p, lora_b)
            projected_delta = torch.mm(projected_b, lora_a)
            original_delta = torch.mm(lora_b, lora_a)
            cos = F.cosine_similarity(
                projected_delta.reshape(1, -1),
                original_delta.reshape(1, -1),
            ).item()
            pdst = 1.0 / (
                1.0 + torch.norm(projected_delta.reshape(1, -1) - original_delta.reshape(1, -1)).item()
            )
            candidates.append((lora_b_name, projected_b, round(float(cos), 5), pdst))

        selected_names, cosine_threshold = self._select_projection_layers(candidates)

        projected = 0
        with torch.no_grad():
            for lora_b_name, projected_b, _, _ in candidates:
                if lora_b_name in selected_names:
                    params[lora_b_name].data.copy_(
                        projected_b.to(
                            device=params[lora_b_name].device,
                            dtype=params[lora_b_name].dtype,
                        )
                    )
                    projected += 1

        mean_pdst = float(np.mean([x[3] for x in candidates])) if candidates else None
        logging.info(
            "SafeLoRA projected %d/%d LoRA layers (threshold=%s, Pdst=%s).",
            projected,
            len(candidates),
            cosine_threshold,
            "N/A" if mean_pdst is None else f"{mean_pdst:.4f}",
        )
        return SafeLoRAProjectionStats(
            projected_layers=projected,
            candidate_layers=len(candidates),
            cosine_threshold=cosine_threshold,
            mean_pdst=mean_pdst,
        )

    def _select_projection_layers(
        self, candidates: List[Tuple[str, torch.Tensor, float, float]]
    ) -> Tuple[set, float]:
        if not candidates:
            return set(), float(self.threshold)

        if self.select_layers_type == "threshold":
            thrs = float(self.threshold)
        elif self.select_layers_type == "number":
            k = min(max(int(self.num_proj_layers), 0), len(candidates))
            if k == 0:
                return set(), float("-inf")
            cosines = np.sort([c[2] for c in candidates])
            thrs = float(cosines[k - 1])
        else:
            raise ValueError(
                "safelora_select_layers_type must be 'threshold' or 'number'."
            )

        return {name for name, _, cos, _ in candidates if cos <= thrs}, thrs

    @staticmethod
    def _load_reference_model(model_path: str) -> torch.nn.Module:
        kwargs = dict(
            return_dict=True,
            load_in_8bit=False,
            device_map="cpu",
            low_cpu_mem_usage=True,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        try:
            return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        except (ImportError, ValueError) as e:
            logging.warning(
                "SafeLoRA reference load with device_map/low_cpu_mem_usage failed "
                "(%s: %s); retrying with default CPU loading.",
                type(e).__name__,
                e,
            )
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                return_dict=True,
                load_in_8bit=False,
                torch_dtype="auto",
                trust_remote_code=True,
            )

    @staticmethod
    def _iter_lora_b_to_weight_names(peft_model: torch.nn.Module) -> Iterable[Tuple[str, str]]:
        for name, _ in peft_model.named_parameters():
            if ".lora_B." not in name or not name.endswith(".weight"):
                continue
            if ".safety_adapter." in name:
                continue
            yield name, SafeLoRAProjector._to_base_weight_name(name)

    @staticmethod
    def _to_lora_a_name(lora_b_name: str) -> str:
        return lora_b_name.replace(".lora_B.", ".lora_A.")

    @staticmethod
    def _to_base_weight_name(lora_b_name: str) -> str:
        name = lora_b_name
        if name.startswith("base_model.model."):
            name = name[len("base_model.model."):]
        name = name.split(".lora_B.", 1)[0] + ".weight"
        return name

    @staticmethod
    def _lookup_param(params: Dict[str, torch.nn.Parameter], name: str):
        if name in params:
            return params[name]
        suffix = "." + name
        for key, value in params.items():
            if key.endswith(suffix):
                return value
        return None
