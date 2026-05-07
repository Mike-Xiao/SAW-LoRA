import copy
import gc
import logging
import torch
import torch.nn.functional as F
from model_trainer import ModelTrainer
from safelora_utils import SafeLoRAProjector


class SafetyEmbeddingDelta(torch.nn.Module):
    """Trainable deltas for a selected subset of input-token embeddings."""

    def __init__(self, base_embedding, token_ids=None):
        super().__init__()
        self.base_embedding = base_embedding
        for p in self.base_embedding.parameters():
            p.requires_grad = False

        self.num_embeddings = base_embedding.num_embeddings
        self.embedding_dim = base_embedding.embedding_dim
        self.padding_idx = getattr(base_embedding, "padding_idx", None)
        self.register_buffer(
            "token_ids", torch.empty(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "index_map", torch.full((self.num_embeddings,), -1, dtype=torch.long),
            persistent=False,
        )
        self.delta_embedding = torch.nn.Embedding(0, self.embedding_dim)
        self.configure_token_ids(token_ids or [], reset=True)

    @property
    def weight(self):
        return self.base_embedding.weight

    def configure_token_ids(self, token_ids, reset=False):
        if token_ids is None:
            token_ids = []
        token_ids = [int(x) for x in token_ids if 0 <= int(x) < self.num_embeddings]
        token_ids = list(dict.fromkeys(token_ids))

        device = self.base_embedding.weight.device
        old_ids = self.token_ids.detach().cpu().tolist()
        same_ids = old_ids == token_ids
        if same_ids:
            if reset:
                with torch.no_grad():
                    self.delta_embedding.weight.zero_()
            return

        self.token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        self.index_map = torch.full(
            (self.num_embeddings,), -1, dtype=torch.long, device=device
        )
        if token_ids:
            self.index_map[self.token_ids] = torch.arange(
                len(token_ids), dtype=torch.long, device=device
            )

        self.delta_embedding = torch.nn.Embedding(len(token_ids), self.embedding_dim)
        self.delta_embedding.to(device=device, dtype=self.base_embedding.weight.dtype)
        torch.nn.init.zeros_(self.delta_embedding.weight)
        self.delta_embedding.weight.requires_grad = True

    def set_active(self, active: bool):
        self.delta_embedding.weight.requires_grad = bool(active)

    def forward(self, input_ids):
        base = self.base_embedding(input_ids)
        if self.token_ids.numel() == 0:
            return base

        mapped = self.index_map[input_ids]
        mask = mapped >= 0
        safe_mapped = mapped.clamp_min(0)
        delta = self.delta_embedding(safe_mapped).to(base.dtype)
        delta = delta * mask.unsqueeze(-1).to(delta.dtype)
        return base + delta


class MyModelTrainer(ModelTrainer):
    """Trainer for LLM with LoRA adapters in federated setting.

    Key design: only LoRA (trainable) parameters are communicated between
    server and clients. The frozen base model stays in place.

    Supports multiple training modes for safety-preserving methods:
      - Standard task training
      - EWC (Elastic Weight Consolidation)
      - LwF (Learning without Forgetting)
      - Replay (experience replay with safety data)
      - Safety-only training (for server_align / local_safety_lora)
      - Safety embedding (backbone layer fine-tuning)
    """

    def get_model_params(self):
        """Return task LoRA parameters on CPU (exclude local safety adapter params)."""
        lora_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and ("safety_adapter" not in name):
                lora_params[name] = param.data.clone().cpu()
        return lora_params

    def set_model_params(self, model_parameters):
        """Load LoRA parameters into the model."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in model_parameters:
                    param.data.copy_(model_parameters[name].to(param.device))

    @staticmethod
    def _lora_gate_group_name(name, group_mode="module_path"):
        base = name
        for marker in (".lora_A.", ".lora_B."):
            if marker in base:
                base = base.split(marker, 1)[0]
                break
        if group_mode == "module_type":
            return base.split(".")[-1]
        return base

    @staticmethod
    def _first_n_batches(data_loader, max_batches):
        if data_loader is None or max_batches <= 0:
            return []
        batches = []
        for idx, batch in enumerate(data_loader):
            if idx >= max_batches:
                break
            batches.append(batch)
        return batches

    def compute_lora_anchor_grads(self, data_loader, device, args, param_keys, max_batches=1):
        """Compute gradients for selected LoRA params on a small anchor set."""
        batches = self._first_n_batches(data_loader, max_batches)
        if not batches:
            return {}

        model = self.model
        model.to(device)
        model.train()

        key_set = set(param_keys)
        orig_grad = {name: p.requires_grad for name, p in model.named_parameters()}
        for name, param in model.named_parameters():
            param.requires_grad = name in key_set

        model.zero_grad(set_to_none=True)
        total_loss = None
        loss_count = 0
        for batch in batches:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / len(batches)
            total_loss = loss if total_loss is None else total_loss + loss
            loss_count += 1

        if total_loss is not None:
            total_loss.backward()

        grads = {}
        for name, param in model.named_parameters():
            if name in key_set and param.grad is not None:
                grads[name] = param.grad.detach().cpu().clone()

        model.zero_grad(set_to_none=True)
        for name, param in model.named_parameters():
            param.requires_grad = orig_grad.get(name, False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return grads

    def safety_aware_lora_download(
        self,
        global_params,
        local_params,
        utility_anchor,
        safety_anchor,
        device,
        args,
        safety_grad_params=None,
    ):
        """Gate the server LoRA download direction per LoRA module.

        Delta is W_g - W_i. Gradients are measured around the client state W_i
        on utility/safety anchors. Each module receives alpha * Delta, where
        alpha = sigmoid(mu * utility_gain - lambda * safety_risk).
        """
        if not global_params or not local_params:
            return copy.deepcopy(global_params), {}

        keys = [k for k in global_params.keys() if k in local_params]
        if not keys:
            return copy.deepcopy(global_params), {}

        anchor_batches = int(getattr(args, "lora_gate_anchor_batches", 1))
        if anchor_batches <= 0 or utility_anchor is None or safety_anchor is None:
            return copy.deepcopy(global_params), {"enabled": False, "reason": "missing_anchor"}

        self.set_model_params(local_params)
        utility_grads = self.compute_lora_anchor_grads(
            utility_anchor, device, args, keys, max_batches=anchor_batches
        )

        if safety_grad_params is not None:
            self.set_model_params(safety_grad_params)
        else:
            self.set_model_params(local_params)
        safety_grads = self.compute_lora_anchor_grads(
            safety_anchor, device, args, keys, max_batches=anchor_batches
        )

        self.set_model_params(local_params)

        if not utility_grads and not safety_grads:
            return copy.deepcopy(global_params), {"enabled": False, "reason": "empty_grads"}

        group_mode = getattr(args, "lora_gate_group_mode", "module_path")
        mu = float(getattr(args, "lora_gate_mu", 5.0))
        lam = float(getattr(args, "lora_gate_lambda", 5.0))
        eps = float(getattr(args, "lora_gate_eps", 1e-8))
        alpha_min = float(getattr(args, "lora_gate_alpha_min", 0.0))
        alpha_max = float(getattr(args, "lora_gate_alpha_max", 1.0))

        group_stats = {}
        for key in keys:
            delta = (global_params[key].float() - local_params[key].float()).cpu()
            if delta.numel() == 0:
                continue
            group = self._lora_gate_group_name(key, group_mode)
            stat = group_stats.setdefault(
                group,
                {
                    "u_dot": 0.0,
                    "u_norm_sq": 0.0,
                    "s_dot": 0.0,
                    "s_norm_sq": 0.0,
                    "d_norm_sq": 0.0,
                    "param_count": 0,
                    "keys": [],
                },
            )
            gu = utility_grads.get(key)
            gs = safety_grads.get(key)
            if gu is not None:
                gu = gu.float().cpu()
                stat["u_dot"] += torch.sum(gu * delta).item()
                stat["u_norm_sq"] += torch.sum(gu * gu).item()
            if gs is not None:
                gs = gs.float().cpu()
                stat["s_dot"] += torch.sum(gs * delta).item()
                stat["s_norm_sq"] += torch.sum(gs * gs).item()
            stat["d_norm_sq"] += torch.sum(delta * delta).item()
            stat["param_count"] += delta.numel()
            stat["keys"].append(key)

        group_alpha = {}
        for group, stat in group_stats.items():
            d_norm = stat["d_norm_sq"] ** 0.5
            u_norm = stat["u_norm_sq"] ** 0.5
            s_norm = stat["s_norm_sq"] ** 0.5

            utility_gain = -stat["u_dot"] / max(u_norm * d_norm, eps)
            safety_risk = max(0.0, stat["s_dot"] / max(s_norm * d_norm, eps))
            score = mu * utility_gain - lam * safety_risk
            alpha = torch.sigmoid(torch.tensor(score)).item()
            alpha = min(max(alpha, alpha_min), alpha_max)
            group_alpha[group] = {
                "alpha": alpha,
                "utility_gain": utility_gain,
                "safety_risk": safety_risk,
                "score": score,
                "param_count": stat["param_count"],
            }

        gated = {}
        for key in global_params:
            if key not in local_params:
                gated[key] = global_params[key].detach().cpu().clone()
                continue
            group = self._lora_gate_group_name(key, group_mode)
            alpha = group_alpha.get(group, {}).get("alpha", 1.0)
            delta = global_params[key].float().cpu() - local_params[key].float().cpu()
            gated[key] = (local_params[key].float().cpu() + alpha * delta).to(global_params[key].dtype)

        alphas = [v["alpha"] for v in group_alpha.values()]
        metrics = {
            "enabled": True,
            "groups": len(group_alpha),
            "alpha_mean": float(sum(alphas) / len(alphas)) if alphas else 1.0,
            "alpha_min": float(min(alphas)) if alphas else 1.0,
            "alpha_max": float(max(alphas)) if alphas else 1.0,
            "group_alpha": group_alpha,
        }
        if bool(getattr(args, "lora_gate_log_modules", 0)):
            top = sorted(group_alpha.items(), key=lambda x: x[1]["alpha"])[:8]
            metrics["lowest_alpha_groups"] = [
                (
                    name,
                    round(vals["alpha"], 4),
                    round(vals["utility_gain"], 4),
                    round(vals["safety_risk"], 4),
                )
                for name, vals in top
            ]
        return gated, metrics

    # ------------------------------------------------------------------
    # SafeLoRA: post-training LoRA_B projection
    # ------------------------------------------------------------------
    def init_safelora_projector(self, args):
        """Initialize and cache SafeLoRA projection matrices."""
        if hasattr(self, "safelora_projector") and self.safelora_projector is not None:
            return

        base_path = getattr(args, "safelora_base_model_path", None)
        aligned_path = getattr(args, "safelora_aligned_model_path", None)
        if not base_path or not aligned_path:
            raise ValueError(
                "SafeLoRA requires --safelora_base_model_path and "
                "--safelora_aligned_model_path."
            )

        self.safelora_projector = SafeLoRAProjector(
            base_model_path=base_path,
            aligned_model_path=aligned_path,
            select_layers_type=getattr(args, "safelora_select_layers_type", "number"),
            threshold=getattr(args, "safelora_threshold", 0.5),
            num_proj_layers=getattr(args, "safelora_num_proj_layers", 10),
            device=getattr(args, "safelora_device", "cpu"),
        )
        self.safelora_projector.initialize(self.model)

    def apply_safelora_projection(self, args):
        """Apply SafeLoRA projection to the current model in-place."""
        if not hasattr(self, "safelora_projector") or self.safelora_projector is None:
            self.init_safelora_projector(args)
        return self.safelora_projector.project_model(self.model)

    def train(self, train_data, device, args):
        model = self.model
        model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=getattr(args, "weight_decay", 0.01),
        )

        epoch_losses = []
        for epoch in range(args.epochs):
            batch_losses = []
            optimizer.zero_grad()

            for step, batch in enumerate(train_data):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()
                batch_losses.append(outputs.loss.item())

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            # Handle remaining gradients at end of epoch
            if len(batch_losses) % args.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
            epoch_losses.append(avg_loss)
            logging.info(
                f"Client {self.id} | Epoch {epoch} | Loss: {avg_loss:.4f}"
            )

        return epoch_losses

    # ------------------------------------------------------------------
    # Safety-only training (server_align / local_safety_lora)
    # ------------------------------------------------------------------
    def train_safety(self, safety_data, device, args):
        """Train LoRA on safety data only."""
        model = self.model
        model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.safety_lr,
            weight_decay=getattr(args, "weight_decay", 0.01),
        )

        for epoch in range(args.safety_epochs):
            batch_losses = []
            optimizer.zero_grad()
            for step, batch in enumerate(safety_data):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                batch_losses.append(loss.item())

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            avg = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
            logging.info(
                f"Client {self.id} | Safety Epoch {epoch} | Loss: {avg:.4f}"
            )

    # ------------------------------------------------------------------
    # EWC: Elastic Weight Consolidation
    # ------------------------------------------------------------------
    def compute_fisher(self, safety_data, device, args):
        """Compute diagonal Fisher information on safety data.

        Returns dict mapping param_name -> Fisher diagonal (on CPU).
        """
        model = self.model
        model.to(device)
        model.train()

        fisher = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        n_samples = 0
        for batch in safety_data:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            outputs.loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)
            n_samples += 1

        for name in fisher:
            fisher[name] = (fisher[name] / max(n_samples, 1)).cpu()

        model.zero_grad()
        logging.info(f"Client {self.id} | Fisher computed over {n_samples} batches")
        return fisher

    def train_with_ewc(self, train_data, device, args, fisher_dict, ref_params):
        """Task training with EWC regularisation to preserve safety."""
        if not fisher_dict or ref_params is None:
            logging.warning(f"Client {self.id} | EWC state missing, fallback to standard train.")
            return self.train(train_data, device, args)

        model = self.model
        model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=getattr(args, "weight_decay", 0.01),
        )

        ewc_lambda = args.ewc_lambda

        epoch_losses = []
        for epoch in range(args.epochs):
            batch_losses = []
            optimizer.zero_grad()

            for step, batch in enumerate(train_data):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                task_loss = outputs.loss / args.gradient_accumulation_steps

                ewc_loss = torch.tensor(0.0, device=device)
                for name, param in model.named_parameters():
                    if param.requires_grad and name in fisher_dict and name in ref_params:
                        fi = fisher_dict[name].to(device=device, dtype=param.dtype)
                        ref = ref_params[name].to(device=device, dtype=param.dtype)
                        ewc_loss += (fi * (param - ref).pow(2)).sum()
                ewc_loss = (ewc_lambda / 2.0) * ewc_loss / args.gradient_accumulation_steps

                loss = task_loss + ewc_loss
                loss.backward()
                batch_losses.append(outputs.loss.item())

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            if len(batch_losses) % args.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            avg = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
            epoch_losses.append(avg)
            logging.info(
                f"Client {self.id} | EWC Epoch {epoch} | TaskLoss: {avg:.4f}"
            )
        return epoch_losses

    # ------------------------------------------------------------------
    # LwF: Learning without Forgetting
    # ------------------------------------------------------------------
    def compute_soft_labels(self, safety_data, device):
        """Compute teacher logits on safety data for LwF distillation.

        Returns list of (input_ids, attention_mask, teacher_logits) tuples on CPU.
        """
        model = self.model
        model.to(device)
        model.eval()

        soft_labels = []
        with torch.no_grad():
            for batch in safety_data:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                soft_labels.append({
                    "input_ids": input_ids.cpu(),
                    "attention_mask": attention_mask.cpu(),
                    "logits": outputs.logits.cpu(),
                })
        logging.info(
            f"Client {self.id} | Soft labels computed: {len(soft_labels)} batches"
        )
        return soft_labels

    def train_with_lwf(self, train_data, device, args, soft_labels):
        """Task training with LwF knowledge distillation to preserve safety."""
        if not soft_labels:
            logging.warning(f"Client {self.id} | Empty soft labels, fallback to standard train.")
            return self.train(train_data, device, args)

        model = self.model
        model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=getattr(args, "weight_decay", 0.01),
        )

        alpha = args.lwf_alpha
        temperature = args.lwf_temperature

        epoch_losses = []
        for epoch in range(args.epochs):
            batch_losses = []
            optimizer.zero_grad()

            sl_iter = iter(soft_labels)

            for step, batch in enumerate(train_data):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                task_loss = outputs.loss / args.gradient_accumulation_steps

                # LwF distillation loss on safety data
                try:
                    sl_batch = next(sl_iter)
                except StopIteration:
                    sl_iter = iter(soft_labels)
                    sl_batch = next(sl_iter)

                sl_ids = sl_batch["input_ids"].to(device)
                sl_mask = sl_batch["attention_mask"].to(device)
                teacher_logits = sl_batch["logits"].to(device)

                student_outputs = model(
                    input_ids=sl_ids,
                    attention_mask=sl_mask,
                )
                student_logits = student_outputs.logits

                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
                student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
                distill_loss = F.kl_div(
                    student_log_probs, teacher_probs, reduction="batchmean"
                ) * (temperature ** 2)
                distill_loss = alpha * distill_loss / args.gradient_accumulation_steps

                loss = task_loss + distill_loss
                loss.backward()
                batch_losses.append(outputs.loss.item())

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            if len(batch_losses) % args.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            avg = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
            epoch_losses.append(avg)
            logging.info(
                f"Client {self.id} | LwF Epoch {epoch} | TaskLoss: {avg:.4f}"
            )
        return epoch_losses

    # ------------------------------------------------------------------
    # Replay: experience replay mixing task + safety data
    # ------------------------------------------------------------------
    def train_with_replay(self, train_data, safety_data, device, args):
        """Task training interleaved with safety data replay."""
        has_safety = safety_data is not None
        if has_safety and hasattr(safety_data, "__len__"):
            has_safety = len(safety_data) > 0
        if not has_safety:
            logging.warning(f"Client {self.id} | Replay safety data empty, fallback to standard train.")
            return self.train(train_data, device, args)

        model = self.model
        model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=getattr(args, "weight_decay", 0.01),
        )

        replay_ratio = args.replay_ratio

        epoch_losses = []
        for epoch in range(args.epochs):
            batch_losses = []
            optimizer.zero_grad()

            safety_iter = iter(safety_data)

            for step, batch in enumerate(train_data):
                # Task batch
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                task_loss = outputs.loss * (1.0 - replay_ratio)
                batch_losses.append(outputs.loss.item())

                # Safety replay batch
                try:
                    s_batch = next(safety_iter)
                except StopIteration:
                    safety_iter = iter(safety_data)
                    try:
                        s_batch = next(safety_iter)
                    except StopIteration:
                        s_batch = None

                if s_batch is not None:
                    s_ids = s_batch["input_ids"].to(device)
                    s_mask = s_batch["attention_mask"].to(device)
                    s_labels = s_batch["labels"].to(device)

                    s_outputs = model(
                        input_ids=s_ids,
                        attention_mask=s_mask,
                        labels=s_labels,
                    )
                    safety_loss = s_outputs.loss * replay_ratio
                    loss = (task_loss + safety_loss) / args.gradient_accumulation_steps
                else:
                    # Safety loader unexpectedly empty during iteration
                    loss = task_loss / args.gradient_accumulation_steps

                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            if len(batch_losses) % args.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            avg = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
            epoch_losses.append(avg)
            logging.info(
                f"Client {self.id} | Replay Epoch {epoch} | TaskLoss: {avg:.4f}"
            )
        return epoch_losses

    # ------------------------------------------------------------------
    # Safety Embedding: train safety-token embedding deltas and small backbone params
    # ------------------------------------------------------------------
    @staticmethod
    def _is_norm_name(name):
        nl = name.lower()
        return (
            "layernorm" in nl
            or ".norm" in nl
            or nl.endswith("norm.weight")
            or nl.endswith("norm.bias")
        )

    @staticmethod
    def _uses_token_safety_embed(layer_type):
        return layer_type in ("token_embed", "token_embed_norm")

    @staticmethod
    def _uses_norm_safety_embed(layer_type):
        return layer_type in ("layernorm", "both", "token_embed_norm")

    @staticmethod
    def _uses_bias_safety_embed(layer_type):
        return layer_type in ("bias", "both")

    def _ensure_safety_embedding_delta(self, token_ids=None, reset=False):
        emb = self.model.get_input_embeddings()
        if isinstance(emb, SafetyEmbeddingDelta):
            wrapper = emb
        else:
            wrapper = SafetyEmbeddingDelta(emb, [])
            try:
                self.model.set_input_embeddings(wrapper)
            except Exception:
                base_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else None
                if base_model is None or not hasattr(base_model, "set_input_embeddings"):
                    raise
                base_model.set_input_embeddings(wrapper)
        wrapper.configure_token_ids(token_ids or [], reset=reset)
        return wrapper

    def _matches_safety_embed_param(self, name, layer_type, include_token=True):
        nl = name.lower()
        if "lora" in nl or "safety_adapter" in nl or "base_embedding" in nl:
            return False
        if include_token and "delta_embedding" in nl:
            return self._uses_token_safety_embed(layer_type)
        if self._uses_norm_safety_embed(layer_type) and self._is_norm_name(name):
            return True
        if self._uses_bias_safety_embed(layer_type) and name.endswith(".bias"):
            return True
        return False

    def _count_safety_embed_base_params(self, layer_type):
        total = 0
        for name, param in self.model.named_parameters():
            if self._matches_safety_embed_param(name, layer_type, include_token=False):
                total += param.numel()
        return total

    @staticmethod
    def _iter_safety_batches(safety_data):
        if safety_data is None:
            return
        if isinstance(safety_data, dict):
            iterables = safety_data.values()
        else:
            iterables = [safety_data]
        for iterable in iterables:
            if iterable is None:
                continue
            for batch in iterable:
                yield batch

    def _select_safety_token_ids(self, safety_data, args, layer_type):
        if not self._uses_token_safety_embed(layer_type):
            return []

        emb = self.model.get_input_embeddings()
        vocab_size = emb.num_embeddings
        hidden_size = emb.embedding_dim
        counts = torch.zeros(vocab_size, dtype=torch.long)

        for batch in self._iter_safety_batches(safety_data):
            input_ids = batch["input_ids"].detach().cpu()
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).detach().cpu()
            valid_ids = input_ids[attention_mask.bool()]
            if valid_ids.numel() > 0:
                valid_ids = valid_ids.clamp(0, vocab_size - 1)
                counts += torch.bincount(valid_ids, minlength=vocab_size)

        if counts.numel() > 0:
            counts[0] = 0
        pad_id = getattr(getattr(self.model, "config", None), "pad_token_id", None)
        if pad_id is not None and 0 <= int(pad_id) < vocab_size:
            counts[int(pad_id)] = 0

        explicit_count = int(getattr(args, "safety_embed_token_count", 0))
        if explicit_count > 0:
            token_count = explicit_count
        else:
            target_params = int(getattr(args, "safety_embed_target_params", 3_800_000))
            base_params = self._count_safety_embed_base_params(layer_type)
            remaining = max(target_params - base_params, hidden_size)
            token_count = max(1, remaining // max(hidden_size, 1))

        available = int((counts > 0).sum().item())
        token_count = min(token_count, available)
        if token_count <= 0:
            return []
        token_ids = torch.topk(counts, k=token_count).indices.tolist()
        return [int(x) for x in token_ids if counts[int(x)].item() > 0]

    def prepare_safety_embed_base_params(self, safety_train_data, args):
        """Create a stable safety_embed layout shared by all clients."""
        layer_type = getattr(args, "safety_embed_layers", "token_embed_norm")
        token_ids = self._select_safety_token_ids(safety_train_data, args, layer_type)
        setattr(args, "_safety_embed_fixed_token_ids", token_ids)
        if self._uses_token_safety_embed(layer_type):
            self._ensure_safety_embedding_delta(token_ids, reset=True)
        params = self.get_safety_embed_params(layer_type)
        logging.info(
            f"SafetyEmbed global token layout prepared: mode={layer_type}, "
            f"token_rows={len(token_ids)}, param_tensors={len(params)}"
        )
        return params

    def train_safety_embed(self, safety_data, device, args):
        """Freeze LoRA, train specified safety embedding/backbone params.

        After training, restore original requires_grad state.
        """
        model = self.model
        model.to(device)

        layer_type = getattr(args, "safety_embed_layers", "token_embed_norm")
        token_ids = getattr(args, "_safety_embed_fixed_token_ids", None)
        if token_ids is None:
            token_ids = self._select_safety_token_ids(safety_data, args, layer_type)
        if self._uses_token_safety_embed(layer_type):
            self._ensure_safety_embedding_delta(token_ids, reset=True)

        # Save original requires_grad state
        orig_grad = {n: p.requires_grad for n, p in model.named_parameters()}

        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze target backbone layers
        unfrozen = []
        trainable_params = 0
        for name, param in model.named_parameters():
            if self._matches_safety_embed_param(name, layer_type):
                param.requires_grad = True
                unfrozen.append(name)
                trainable_params += param.numel()

        logging.info(
            f"Client {self.id} | SafetyEmbed mode={layer_type} | "
            f"unfrozen_tensors={len(unfrozen)} trainable_params={trainable_params:,} "
            f"token_rows={len(token_ids)}"
        )

        if not unfrozen:
            logging.warning("No backbone params unfrozen — skipping safety embed")
            for name, param in model.named_parameters():
                param.requires_grad = orig_grad.get(name, False)
            return

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.safety_lr,
            weight_decay=getattr(args, "weight_decay", 0.01),
        )

        model.train()
        for epoch in range(args.safety_epochs):
            batch_losses = []
            for batch in safety_data:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                batch_losses.append(outputs.loss.item())

            avg = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
            logging.info(
                f"Client {self.id} | SafetyEmbed Epoch {epoch} | Loss: {avg:.4f}"
            )

        # Restore original requires_grad state; safety_embed params are local state
        # and should stay frozen during later task-LoRA training.
        for name, param in model.named_parameters():
            if self._matches_safety_embed_param(name, layer_type):
                param.requires_grad = False
            else:
                param.requires_grad = orig_grad.get(name, False)

        model.zero_grad(set_to_none=True)
        del optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_safety_embed_params(self, layer_type="token_embed_norm"):
        """Extract safety-embedding parameters and token-row metadata."""
        params = {}

        emb = self.model.get_input_embeddings()
        if self._uses_token_safety_embed(layer_type):
            if isinstance(emb, SafetyEmbeddingDelta):
                params["__safety_embed_token_ids__"] = emb.token_ids.detach().cpu().clone()
                params["__safety_embed_delta_weight__"] = (
                    emb.delta_embedding.weight.data.detach().cpu().clone()
                )
            else:
                params["__safety_embed_token_ids__"] = torch.empty(0, dtype=torch.long)
                params["__safety_embed_delta_weight__"] = torch.empty(
                    0, emb.embedding_dim, dtype=emb.weight.dtype
                )

        for name, param in self.model.named_parameters():
            if self._matches_safety_embed_param(name, layer_type, include_token=False):
                params[name] = param.data.clone().cpu()
        return params

    def set_safety_embed_params(self, params):
        """Load safety-embedding parameters and token-row metadata."""
        if params is None:
            return

        token_ids = params.get("__safety_embed_token_ids__")
        delta_weight = params.get("__safety_embed_delta_weight__")
        if token_ids is not None:
            wrapper = self._ensure_safety_embedding_delta(token_ids.tolist(), reset=True)
            if delta_weight is not None and wrapper.delta_embedding.weight.shape == delta_weight.shape:
                with torch.no_grad():
                    wrapper.delta_embedding.weight.data.copy_(
                        delta_weight.to(wrapper.delta_embedding.weight.device)
                    )
            wrapper.set_active(False)

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.data.copy_(params[name].to(param.device))

    # ------------------------------------------------------------------
    # Local Safety Adapter: lightweight adapter in backbone (PEFT multi-adapter)
    # ------------------------------------------------------------------
    def add_safety_adapter(self, safety_adapter_rank=4):
        """Add a second LoRA adapter named 'safety_adapter' to the model (idempotent)."""
        from peft import LoraConfig, TaskType

        # Avoid adding duplicated adapter
        if hasattr(self.model, "peft_config") and "safety_adapter" in self.model.peft_config:
            logging.info("Safety adapter already exists, skip add_adapter.")
            self.model.set_adapter("default")
            return

        safety_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=safety_adapter_rank,
            lora_alpha=safety_adapter_rank * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        self.model.add_adapter("safety_adapter", safety_config)
        # Switch back to default (task) adapter
        self.model.set_adapter("default")
        logging.info(f"Safety adapter added (rank={safety_adapter_rank})")

    def get_safety_adapter_params(self):
        """Extract safety_adapter parameters on CPU."""
        params = {}
        for name, param in self.model.named_parameters():
            if "safety_adapter" in name:
                params[name] = param.data.clone().cpu()
        return params

    def set_safety_adapter_params(self, params):
        """Load safety_adapter parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.data.copy_(params[name].to(param.device))

    def train_safety_adapter(self, safety_data, device, args):
        """Train only the safety_adapter on safety data."""
        model = self.model
        model.to(device)
        model.set_adapter("safety_adapter")
        model.train()

        optimizer = torch.optim.AdamW(
            [p for n, p in model.named_parameters()
             if "safety_adapter" in n and p.requires_grad],
            lr=args.safety_lr,
            weight_decay=getattr(args, "weight_decay", 0.01),
        )

        for epoch in range(args.safety_epochs):
            batch_losses = []
            for batch in safety_data:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask, labels=labels)
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_losses.append(outputs.loss.item())

            avg = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
            logging.info(
                f"Client {self.id} | SafetyAdapter Epoch {epoch} | Loss: {avg:.4f}")

        # Switch back to task adapter
        model.set_adapter("default")

    def enable_both_adapters(self):
        """Activate both task LoRA and safety_adapter for inference.

        Merges safety_adapter delta (B @ A * scaling) into the base weights
        so both adapters are effective. Works with any PEFT version.
        Call enable_task_adapter_only() to undo afterwards.
        """
        if getattr(self, "_safety_adapter_merged", False):
            return

        self.model.set_adapter("default")
        merged_any = False

        for _, module in self.model.named_modules():
            lora_A = getattr(module, "lora_A", None)
            lora_B = getattr(module, "lora_B", None)
            if not isinstance(lora_A, torch.nn.ModuleDict):
                continue
            if "safety_adapter" not in lora_A:
                continue

            A = lora_A["safety_adapter"].weight
            B = lora_B["safety_adapter"].weight
            scaling = 1.0
            if hasattr(module, "scaling") and isinstance(module.scaling, dict):
                scaling = module.scaling.get("safety_adapter", 1.0)

            delta = (B @ A) * scaling

            base = module.get_base_layer() if hasattr(module, "get_base_layer") else module
            with torch.no_grad():
                base.weight.data += delta.to(base.weight.dtype)
            merged_any = True

        self._safety_adapter_merged = merged_any

    def enable_task_adapter_only(self):
        """Activate only the task LoRA adapter. Undo safety_adapter merge if needed."""
        if getattr(self, "_safety_adapter_merged", False):
            for _, module in self.model.named_modules():
                lora_A = getattr(module, "lora_A", None)
                lora_B = getattr(module, "lora_B", None)
                if not isinstance(lora_A, torch.nn.ModuleDict):
                    continue
                if "safety_adapter" not in lora_A:
                    continue

                A = lora_A["safety_adapter"].weight
                B = lora_B["safety_adapter"].weight
                scaling = 1.0
                if hasattr(module, "scaling") and isinstance(module.scaling, dict):
                    scaling = module.scaling.get("safety_adapter", 1.0)

                delta = (B @ A) * scaling

                base = module.get_base_layer() if hasattr(module, "get_base_layer") else module
                with torch.no_grad():
                    base.weight.data -= delta.to(base.weight.dtype)

            self._safety_adapter_merged = False

        self.model.set_adapter("default")

    def test(self, test_data, device, args):
        model = self.model
        model.to(device)
        model.eval()

        total_loss = 0.0
        total_tokens = 0
        total_samples = 0

        # token-level accuracy (shifted, consistent with causal LM loss)
        total_correct = 0
        total_eval_tokens = 0

        with torch.no_grad():
            for batch in test_data:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                # shifted valid tokens
                shift_labels = labels[:, 1:].contiguous()
                valid_mask = (shift_labels != -100)
                valid_tokens = valid_mask.sum().item()

                if valid_tokens > 0:
                    total_loss += outputs.loss.item() * valid_tokens
                    total_tokens += valid_tokens

                    shift_logits = outputs.logits[:, :-1, :].contiguous()
                    pred_ids = shift_logits.argmax(dim=-1)
                    total_correct += ((pred_ids == shift_labels) & valid_mask).sum().item()
                    total_eval_tokens += valid_tokens

                total_samples += input_ids.size(0)

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        if total_tokens > 0:
            # Keep logging readable if a broken dataset/model path produces an
            # extreme loss. The raw loss is still returned for diagnosis.
            ppl = torch.exp(torch.tensor(min(avg_loss, 50.0))).item()
        else:
            ppl = float("inf")
        token_acc = (total_correct / total_eval_tokens) if total_eval_tokens > 0 else 0.0

        metrics = {
            "test_loss": avg_loss,
            "test_perplexity": ppl,
            "test_token_acc": token_acc,
            "test_total": total_samples,
        }
        return metrics

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ):
        return False
