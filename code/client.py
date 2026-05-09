import copy
import logging
import torch


class Client:
    """Federated client for LLM LoRA training.

    All clients share a single model_trainer instance (single-machine simulation).
    Before each client trains, global LoRA params are loaded in; after training,
    local LoRA params are extracted and returned for aggregation.

    Supports multiple safety-preserving methods via the `method` argument.
    """

    def __init__(
        self, client_idx, local_training_data, local_sample_number,
        args, device, model_trainer, safety_train_data=None,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_sample_number = local_sample_number
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.method = getattr(args, "method", "baseline")
        self.safety_train_data = safety_train_data

        # Per-client state for specific methods
        self.safety_lora_params = None   # local_safety_lora
        self.fisher_dict = None          # ewc
        self.ref_params = None           # ewc
        self.soft_labels = None          # lwf
        self.safety_embed_params = None  # safety_embed
        self.safety_adapter_params = None  # local_safety_adapter
        self.local_lora_params = None    # last client-side W_i for gated download
        self.utility_anchor_batches = None
        self.safety_anchor_batches = None
        self.last_lora_gate_metrics = None

        logging.info(f"Client {client_idx}: {local_sample_number} samples")

    def get_sample_number(self):
        return self.local_sample_number

    def get_last_lora_gate_metrics(self):
        return self.last_lora_gate_metrics

    # ------------------------------------------------------------------
    # Initialisation (called once before federated training starts)
    # ------------------------------------------------------------------
    def init_safety_method(self):
        """One-time setup before federated rounds begin."""
        if self.safety_train_data is None:
            return

        mt = self.model_trainer
        mt.id = self.client_idx

        if self.method == "ewc":
            # Train on safety data first, then compute Fisher + save ref
            logging.info(f"Client {self.client_idx} | EWC init: safety pre-training")
            mt.train_safety(self.safety_train_data, self.device, self.args)
            self.ref_params = mt.get_model_params()
            self.fisher_dict = mt.compute_fisher(
                self.safety_train_data, self.device, self.args
            )

        elif self.method == "lwf":
            # Train on safety data first, then compute soft labels
            logging.info(f"Client {self.client_idx} | LwF init: safety pre-training")
            mt.train_safety(self.safety_train_data, self.device, self.args)
            self.soft_labels = mt.compute_soft_labels(
                self.safety_train_data, self.device
            )

        elif self.method == "local_safety_lora":
            # Train safety from current task-tuned model, store only the delta
            init_params = copy.deepcopy(mt.get_model_params())
            mt.train_safety(self.safety_train_data, self.device, self.args)
            after_params = mt.get_model_params()
            self.safety_lora_params = {
                k: after_params[k] - init_params[k] for k in init_params
            }
            # Restore model to init state
            mt.set_model_params(init_params)
            logging.info(
                f"Client {self.client_idx} | LocalSafetyLoRA init: "
                f"safety delta computed"
                f"safety params cloned"
            )

        elif self.method == "safety_embed":
            # Train backbone layers on safety data
            logging.info(
                f"Client {self.client_idx} | SafetyEmbed init: "
                f"backbone pre-training"
            )
            mt.train_safety_embed(self.safety_train_data, self.device, self.args)
            layer_type = getattr(self.args, "safety_embed_layers", "token_embed_norm")
            self.safety_embed_params = mt.get_safety_embed_params(layer_type)

        elif self.method == "local_safety_adapter":
            # Add safety adapter and train it on safety data
            logging.info(
                f"Client {self.client_idx} | LocalSafetyAdapter init: "
                f"training safety adapter"
            )
            rank = getattr(self.args, "safety_adapter_rank", 4)
            mt.add_safety_adapter(rank)
            mt.train_safety_adapter(self.safety_train_data, self.device, self.args)
            self.safety_adapter_params = mt.get_safety_adapter_params()

        elif self.method == "safelora":
            # SafeLoRA has no per-client safety pre-training state. Its
            # projection matrices are initialized once at the server/trainer.
            logging.info(f"Client {self.client_idx} | SafeLoRA init: no local state")

    # ------------------------------------------------------------------
    # Main training entry point (called each round)
    # ------------------------------------------------------------------
    def train(self, lora_params):
        """Load global LoRA params, train locally, return updated LoRA params."""
        self.model_trainer.id = self.client_idx

        if self.method == "baseline":
            updated = self._train_baseline(lora_params)
        elif self.method == "ewc":
            updated = self._train_ewc(lora_params)
        elif self.method == "lwf":
            updated = self._train_lwf(lora_params)
        elif self.method == "replay":
            updated = self._train_replay(lora_params)
        elif self.method == "local_safety_lora":
            updated = self._train_local_safety_lora(lora_params)
        elif self.method == "safety_embed":
            updated = self._train_safety_embed(lora_params)
        elif self.method == "local_safety_adapter":
            updated = self._train_local_safety_adapter(lora_params)
        elif self.method == "safelora":
            updated = self._train_safelora(lora_params)
        else:
            updated = self._train_baseline(lora_params)

        if updated is not None:
            self.local_lora_params = copy.deepcopy(updated)
        return updated

    def _has_safety_anchor(self):
        if self.safety_train_data is None:
            return False
        if hasattr(self.safety_train_data, "__len__"):
            return len(self.safety_train_data) > 0
        return True

    def _use_lora_download_gate(self):
        if not bool(getattr(self.args, "lora_download_gate", 0)):
            return False
        if self.method == "safety_embed":
            return False
        return self._has_safety_anchor()

    def _build_safety_grad_params(self, local_params):
        if self.method != "local_safety_lora" or self.safety_lora_params is None:
            return None
        merged = {}
        for key, val in local_params.items():
            safety_delta = self.safety_lora_params.get(key)
            if safety_delta is None:
                merged[key] = val
            else:
                merged[key] = (val.float() + safety_delta.float()).to(val.dtype)
        return merged

    @staticmethod
    def _clone_batch_to_cpu(batch):
        cloned = {}
        for key, val in batch.items():
            if torch.is_tensor(val):
                cloned[key] = val.detach().cpu().clone()
            else:
                cloned[key] = val
        return cloned

    def _capture_anchor_batches(self, data_loader, cache_attr):
        cached = getattr(self, cache_attr)
        if cached is not None:
            return cached

        max_batches = max(1, int(getattr(self.args, "lora_gate_anchor_batches", 1)))
        batches = []
        if data_loader is not None:
            for idx, batch in enumerate(data_loader):
                if idx >= max_batches:
                    break
                batches.append(self._clone_batch_to_cpu(batch))
        setattr(self, cache_attr, batches)
        return batches

    def _receive_lora_params(self, global_lora_params):
        """Receive server LoRA params, optionally with safety-aware module gates."""
        mt = self.model_trainer

        if self.local_lora_params is None:
            self.last_lora_gate_metrics = None
            self.local_lora_params = copy.deepcopy(global_lora_params)
            mt.set_model_params(global_lora_params)
            return global_lora_params

        if not self._use_lora_download_gate():
            self.last_lora_gate_metrics = None
            mt.set_model_params(global_lora_params)
            return global_lora_params

        safety_grad_params = self._build_safety_grad_params(self.local_lora_params)
        utility_anchor = self._capture_anchor_batches(
            self.local_training_data, "utility_anchor_batches"
        )
        safety_anchor = self._capture_anchor_batches(
            self.safety_train_data, "safety_anchor_batches"
        )
        gated_params, gate_metrics = mt.safety_aware_lora_download(
            global_params=global_lora_params,
            local_params=self.local_lora_params,
            utility_anchor=utility_anchor,
            safety_anchor=safety_anchor,
            device=self.device,
            args=self.args,
            safety_grad_params=safety_grad_params,
        )
        self.last_lora_gate_metrics = gate_metrics
        mt.set_model_params(gated_params)

        if gate_metrics.get("enabled", False):
            logging.info(
                f"Client {self.client_idx} | LoRA download gate: "
                f"groups={gate_metrics.get('groups', 0)} "
                f"alpha_mean={gate_metrics.get('alpha_mean', 1.0):.4f} "
                f"alpha_min={gate_metrics.get('alpha_min', 1.0):.4f} "
                f"alpha_max={gate_metrics.get('alpha_max', 1.0):.4f}"
            )
            if "lowest_alpha_groups" in gate_metrics:
                logging.info(
                    f"Client {self.client_idx} | lowest gate groups: "
                    f"{gate_metrics['lowest_alpha_groups']}"
                )
        else:
            logging.info(
                f"Client {self.client_idx} | LoRA download gate skipped: "
                f"{gate_metrics.get('reason', 'unknown')}"
            )
        return gated_params

    # ------------------------------------------------------------------
    # Per-method training implementations
    # ------------------------------------------------------------------
    def _train_baseline(self, lora_params):
        mt = self.model_trainer
        self._receive_lora_params(lora_params)
        mt.train(self.local_training_data, self.device, self.args)
        return mt.get_model_params()

    def _train_ewc(self, lora_params):
        mt = self.model_trainer
        if self.fisher_dict is None or self.ref_params is None:
            logging.warning(
                f"Client {self.client_idx} | EWC state missing, fallback to baseline."
            )
            return self._train_baseline(lora_params)
        self._receive_lora_params(lora_params)
        mt.train_with_ewc(
            self.local_training_data, self.device, self.args,
            self.fisher_dict, self.ref_params,
        )
        return mt.get_model_params()

    def _train_lwf(self, lora_params):
        mt = self.model_trainer
        if not self.soft_labels:
            logging.warning(
                f"Client {self.client_idx} | LwF soft labels missing, fallback to baseline."
            )
            return self._train_baseline(lora_params)

        self._receive_lora_params(lora_params)
        mt.train_with_lwf(
            self.local_training_data, self.device, self.args,
            self.soft_labels,
        )
        return mt.get_model_params()

    def _train_replay(self, lora_params):
        mt = self.model_trainer

        has_safety = self.safety_train_data is not None
        if has_safety and hasattr(self.safety_train_data, "__len__"):
            has_safety = len(self.safety_train_data) > 0

        if not has_safety:
            logging.warning(
                f"Client {self.client_idx} | Replay safety data missing/empty, fallback to baseline."
            )
            return self._train_baseline(lora_params)

        self._receive_lora_params(lora_params)
        mt.train_with_replay(
            self.local_training_data, self.safety_train_data,
            self.device, self.args,
        )
        return mt.get_model_params()

    def _train_local_safety_lora(self, lora_params):
        mt = self.model_trainer

        # 1. Train task LoRA
        self._receive_lora_params(lora_params)
        mt.train(self.local_training_data, self.device, self.args)
        task_params = mt.get_model_params()

        # 2. Train safety LoRA: start from task params, extract delta
        if self.safety_train_data is not None:
            # Apply existing safety delta on top of task params for training context
            if self.safety_lora_params is not None:
                merged = {k: task_params[k] + self.safety_lora_params[k]
                          for k in task_params}
            else:
                merged = copy.deepcopy(task_params)

            mt.set_model_params(merged)
            mt.train_safety(self.safety_train_data, self.device, self.args)
            after_safety = mt.get_model_params()

            # Store only the safety delta (relative to task params)
            self.safety_lora_params = {
                k: after_safety[k] - task_params[k] for k in task_params
            }

        # 3. Return only task LoRA for aggregation
        return task_params

    def _train_safety_embed(self, lora_params):
        mt = self.model_trainer
        # Restore backbone safety params before task training
        if self.safety_embed_params is not None:
            mt.set_safety_embed_params(self.safety_embed_params)
        self._receive_lora_params(lora_params)
        mt.train(self.local_training_data, self.device, self.args)
        return mt.get_model_params()

    def _train_local_safety_adapter(self, lora_params):
        mt = self.model_trainer

        # 1. Train task LoRA (default adapter)
        mt.enable_task_adapter_only()
        self._receive_lora_params(lora_params)
        mt.train(self.local_training_data, self.device, self.args)
        task_params = mt.get_model_params()

        # 2. Train safety adapter (independent, local only)
        has_safety = self.safety_train_data is not None
        if has_safety and hasattr(self.safety_train_data, "__len__"):
            has_safety = len(self.safety_train_data) > 0

        if has_safety:
            if self.safety_adapter_params is not None:
                mt.set_safety_adapter_params(self.safety_adapter_params)
            mt.train_safety_adapter(self.safety_train_data, self.device, self.args)
            self.safety_adapter_params = mt.get_safety_adapter_params()

        mt.enable_task_adapter_only()
        return task_params

    def _train_safelora(self, lora_params):
        mt = self.model_trainer
        self._receive_lora_params(lora_params)
        mt.train(self.local_training_data, self.device, self.args)
        mt.apply_safelora_projection(self.args)
        return mt.get_model_params()

    # ------------------------------------------------------------------
    # Helpers for evaluation
    # ------------------------------------------------------------------
    def get_merged_lora_params(self, global_task_params):
        """Merge global task LoRA + local safety LoRA (additive).

        Used by local_safety_lora method for per-client evaluation.
        """
        if self.safety_lora_params is None:
            return global_task_params
        merged = {}
        for key in global_task_params:
            task_p = global_task_params[key].float()
            safety_p = self.safety_lora_params.get(key, torch.zeros_like(task_p)).float()
            merged[key] = (task_p + safety_p).to(global_task_params[key].dtype)
        return merged

    def local_test(self, test_data):
        """Evaluate on given test data."""
        return self.model_trainer.test(test_data, self.device, self.args)
