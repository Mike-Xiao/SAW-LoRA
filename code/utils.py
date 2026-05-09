import torch


def get_lora_param_names(model):
    """Return names of all trainable (LoRA) parameters."""
    return [name for name, p in model.named_parameters() if p.requires_grad]


def count_parameters(model):
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def aggregate_lora_params(lora_params_list):
    """Weighted average of LoRA parameters.

    Args:
        lora_params_list: list of (sample_num, lora_state_dict) tuples

    Returns:
        Averaged LoRA state dict.
    """
    total_samples = sum(n for n, _ in lora_params_list)
    if total_samples <= 0:
        # Fallback: equal weight when all counts are zero
        total_samples = len(lora_params_list)
        lora_params_list = [(1, p) for _, p in lora_params_list]
    _, first = lora_params_list[0]
    averaged = {}

    for key in first.keys():
        averaged[key] = torch.zeros_like(first[key], dtype=torch.float32)
        for n, params in lora_params_list:
            averaged[key] += params[key].float() * (n / total_samples)
        averaged[key] = averaged[key].to(first[key].dtype)

    return averaged
