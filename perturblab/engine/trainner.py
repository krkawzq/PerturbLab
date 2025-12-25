"""Unified training engine for PerturbLab models.

This module provides a robust Trainer class with:
- Automatic ModelIO detection and conversion
- Automatic Mixed Precision (AMP)
- Gradient Accumulation & Clipping
- Learning Rate Scheduling & Early Stopping
- Lifecycle Hooks for customization
- Batch dict format support ('inputs', 'labels', 'metadata')
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from perturblab.utils import get_logger

logger = get_logger()

__all__ = ["Trainer"]


class Trainer:
    """A powerful, customizable trainer for PyTorch models.

    This trainer manages the training loop, validation, optimization, and logging.
    It is designed to be subclassed for specific behaviors, or used out-of-the-box
    with standard PerturbLab models.

    Args:
        model: The PyTorch model to train.
        optimizer: The optimizer.
        loss_fn: The loss function. Should return a scalar tensor or a dict containing 'loss'.
        scheduler: Optional learning rate scheduler.
        device: Device to run on ('cuda', 'cpu', 'mps').
        input_adapter: Callable to convert batch dict to model input arguments/dataclass.

        # Training Strategy Configs
        use_amp: Whether to use Automatic Mixed Precision (float16).
        grad_accum_steps: Number of steps to accumulate gradients before updating.
        max_grad_norm: Max norm for gradient clipping (0.0 to disable).
        early_stopping_patience: Number of epochs to wait before stopping if val loss doesn't improve.

        # Logging & IO Configs
        experiment_name: Name for saving checkpoints.
        checkpoint_dir: Directory to save checkpoints.
        save_best_only: If True, only keeps the best checkpoint.
        log_interval: How many batches to wait before logging training status.
    """

    def __init__(
        self,
        # Core Components
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Callable | nn.Module,
        scheduler: Any | None = None,
        device: str | torch.device = "cuda",
        input_adapter: Callable[[dict], Any] | None = None,
        # Optimization Strategy
        use_amp: bool = False,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 0.0,
        early_stopping_patience: int = -1,
        # IO & Logging
        experiment_name: str = "default_exp",
        checkpoint_dir: str = "./checkpoints",
        save_best_only: bool = True,
        log_interval: int = 10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.input_adapter = input_adapter

        # Device setup
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model.to(self.device)
        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn.to(self.device)

        # Optimization State
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm

        # Training State
        self.global_step = 0
        self.current_epoch = 0
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}

        # Early Stopping State
        self.early_stopping_patience = early_stopping_patience
        self.patience_counter = 0
        self.best_val_loss = float("inf")

        # IO Setup
        self.experiment_name = experiment_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.log_interval = log_interval

    # =========================================================================
    # Helper Methods (Internal)
    # =========================================================================

    def _move_to_device(self, data: Any) -> Any:
        """Recursively move data to the configured device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._move_to_device(v) for v in data]
        elif isinstance(data, tuple):
            return tuple(self._move_to_device(v) for v in data)
        return data

    def _prepare_inputs(self, batch: dict) -> Any:
        """Processes batch dict into model-ready input.

        Auto-detects and converts to ModelIO if model follows PerturbLab conventions.

        Args:
            batch: Batch dictionary (already on device) with 'inputs' key.

        Returns:
            ModelIO instance, dict, or other format ready for model.forward().
        """
        # 1. Extract inputs dict
        inputs_dict = batch.get("inputs", batch)

        # 2. Use custom adapter if provided
        if self.input_adapter:
            return self.input_adapter(inputs_dict)

        # 3. Auto-detect ModelIO class (PerturbLab convention)
        # Look for model.input_class or model.InputClass or via type hints
        input_class = self._detect_input_class()

        if input_class and isinstance(inputs_dict, dict):
            # Get valid parameters for this Input class
            valid_params = self._get_valid_params(input_class, inputs_dict)
            try:
                return input_class(**valid_params)
            except Exception as e:
                logger.warning(
                    f"Failed to create {input_class.__name__} from inputs dict: {e}. "
                    f"Falling back to dict input."
                )
                return inputs_dict

        # 4. Fallback: pass dict directly
        return inputs_dict

    def _detect_input_class(self) -> type | None:
        """Detects the Input class for this model.

        Checks:
        1. model.input_class attribute
        2. model.InputClass attribute
        3. model.forward() type hints

        Returns:
            Input class if found, else None.
        """
        # Check for explicit input_class attribute
        if hasattr(self.model, "input_class"):
            return self.model.input_class

        if hasattr(self.model, "InputClass"):
            return self.model.InputClass

        # Check forward method type hints
        try:
            sig = inspect.signature(self.model.forward)
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param.annotation != inspect.Parameter.empty:
                    # Return first non-self parameter's type
                    return param.annotation
        except Exception:
            pass

        return None

    def _get_valid_params(self, input_class: type, inputs_dict: dict) -> dict:
        """Extracts only valid parameters for the Input class from inputs dict.

        Uses inspect to get the __init__ signature and filters inputs_dict
        to only include valid parameters.

        Args:
            input_class: The ModelIO or dataclass type.
            inputs_dict: Dictionary with potential input parameters.

        Returns:
            Filtered dictionary with only valid parameters.
        """
        try:
            # Get __init__ parameters
            sig = inspect.signature(input_class.__init__)
            valid_param_names = set(sig.parameters.keys()) - {"self"}

            # Filter inputs_dict
            valid_params = {k: v for k, v in inputs_dict.items() if k in valid_param_names}

            return valid_params
        except Exception:
            # Fallback: return all
            return inputs_dict

    def _process_loss(self, loss_output: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """Normalize loss output to (scalar_loss, metrics_dict)."""
        metrics = {}
        if isinstance(loss_output, dict):
            # Assume 'loss' or 'total_loss' is the key for backprop
            if "loss" in loss_output:
                scalar_loss = loss_output["loss"]
            elif "total_loss" in loss_output:
                scalar_loss = loss_output["total_loss"]
            else:
                # Fallback: take first item
                scalar_loss = list(loss_output.values())[0]

            # Record all items as metrics
            for k, v in loss_output.items():
                if isinstance(v, torch.Tensor):
                    metrics[k] = v.item()
                else:
                    metrics[k] = v
        else:
            scalar_loss = loss_output
            metrics["loss"] = scalar_loss.item()

        return scalar_loss, metrics

    # =========================================================================
    # Overridable Steps (The "Kernel" of the Trainer)
    # =========================================================================

    def training_step(self, batch: dict, batch_idx: int) -> dict[str, Any]:
        """Performs a single training step. Can be overridden.

        Args:
            batch: The batch data from DataLoader.
            batch_idx: Index of the current batch.

        Returns:
            Dict containing 'loss' (Tensor) and optional metrics.
        """
        model_input = self._prepare_inputs(batch)

        # Forward
        outputs = self.model(model_input)

        # Compute Loss (pass metadata/labels dynamically)
        labels = self._move_to_device(batch.get("labels", {}))
        metadata = batch.get("metadata", {})

        loss_output = self.loss_fn(
            outputs, labels, metadata, epoch=self.current_epoch, step=self.global_step
        )

        loss_scalar, metrics = self._process_loss(loss_output)
        return {"loss": loss_scalar, "metrics": metrics}

    def validation_step(self, batch: dict, batch_idx: int) -> dict[str, Any]:
        """Performs a single validation step. Can be overridden."""
        model_input = self._prepare_inputs(batch)

        outputs = self.model(model_input)

        labels = self._move_to_device(batch.get("labels", {}))
        metadata = batch.get("metadata", {})

        loss_output = self.loss_fn(
            outputs, labels, metadata, epoch=self.current_epoch, is_validation=True
        )

        _, metrics = self._process_loss(loss_output)
        return metrics

    # =========================================================================
    # Lifecycle Hooks (For external customization)
    # =========================================================================

    def on_train_start(self):
        """Called before training begins."""
        pass

    def on_epoch_start(self):
        """Called at the start of each epoch."""
        pass

    def on_train_batch_end(self, outputs: dict, batch: dict, batch_idx: int):
        """Called after optimization step."""
        pass

    def on_validation_end(self, metrics: dict[str, float]):
        """Called after validation loop completes."""
        pass

    def on_epoch_end(self):
        """Called at the end of each epoch."""
        pass

    # =========================================================================
    # Core Loops
    # =========================================================================

    def run_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Runs one training epoch."""
        self.model.train()
        epoch_loss = 0.0
        step_metrics = {}

        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch+1} [Train]", leave=False)

        for batch_idx, batch in enumerate(pbar):
            # 1. Forward & Loss Calculation (with AMP)
            with autocast(enabled=self.use_amp):
                step_out = self.training_step(batch, batch_idx)
                loss = step_out["loss"]

                # Normalize loss for accumulation
                loss = loss / self.grad_accum_steps

            # 2. Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 3. Optimization Step (Accumulation aware)
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient Clipping
                if self.max_grad_norm > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Optimizer Step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # 4. Logging & Hooks
            loss_val = step_out["loss"].item()
            epoch_loss += loss_val

            # Aggregate metrics
            for k, v in step_out["metrics"].items():
                step_metrics[k] = step_metrics.get(k, 0.0) + v

            self.on_train_batch_end(step_out, batch, batch_idx)

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        # Calculate epoch averages
        avg_loss = epoch_loss / len(loader)
        avg_metrics = {k: v / len(loader) for k, v in step_metrics.items()}
        avg_metrics["total_loss"] = avg_loss

        return avg_metrics

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        """Runs validation loop."""
        self.model.eval()
        val_metrics = {}

        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch+1} [Val]", leave=False)

        for batch_idx, batch in enumerate(pbar):
            metrics = self.validation_step(batch, batch_idx)

            for k, v in metrics.items():
                val_metrics[k] = val_metrics.get(k, 0.0) + v

        # Average metrics
        avg_metrics = {k: v / len(loader) for k, v in val_metrics.items()}
        self.on_validation_end(avg_metrics)
        return avg_metrics

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader | None = None, epochs: int = 100
    ):
        """Main entry point for training."""
        self.on_train_start()

        try:
            for epoch in range(epochs):
                self.current_epoch = epoch
                self.on_epoch_start()

                # --- Train ---
                train_metrics = self.run_epoch(train_loader)
                self.history["train_loss"].append(train_metrics["total_loss"])
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.history["lr"].append(current_lr)

                # --- Validate ---
                val_metrics_str = ""
                if val_loader:
                    val_metrics = self.validate(val_loader)
                    current_val_loss = val_metrics.get("loss", val_metrics.get("total_loss", 0.0))
                    self.history["val_loss"].append(current_val_loss)
                    val_metrics_str = f" | Val Loss: {current_val_loss:.4f}"

                    # Scheduler Step
                    if self.scheduler:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(current_val_loss)
                        else:
                            self.scheduler.step()

                    # Checkpointing & Early Stopping
                    self._handle_checkpointing(current_val_loss)
                    if self._check_early_stopping(current_val_loss):
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
                else:
                    # Scheduler Step (no metric)
                    if self.scheduler and not isinstance(
                        self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.scheduler.step()
                    self._save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

                self.on_epoch_end()

                # Logging
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_metrics['total_loss']:.4f}{val_metrics_str} | "
                    f"LR: {current_lr:.2e}"
                )

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user. Saving emergency checkpoint...")
            self._save_checkpoint("emergency_checkpoint.pt")

    # =========================================================================
    # Persistence & Control
    # =========================================================================

    def _handle_checkpointing(self, current_val_loss: float):
        """Logic for saving best models."""
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self._save_checkpoint("best_model.pt")
            logger.debug(f"New best model saved (loss: {current_val_loss:.4f})")

        if not self.save_best_only:
            self._save_checkpoint(f"checkpoint_epoch_{self.current_epoch}.pt")

    def _check_early_stopping(self, current_val_loss: float) -> bool:
        """Checks if training should stop."""
        if self.early_stopping_patience < 0:
            return False

        if current_val_loss < self.best_val_loss:  # Note: best_val updated in handle_checkpointing
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.early_stopping_patience

    def _save_checkpoint(self, filename: str):
        """Saves full training state."""
        path = self.checkpoint_dir / filename
        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }
        if self.scheduler:
            state["scheduler_state"] = self.scheduler.state_dict()
        if self.scaler:
            state["scaler_state"] = self.scaler.state_dict()

        torch.save(state, path)

    def load_checkpoint(self, path: str | Path):
        """Resumes training state from checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint.get("history", self.history)

        if self.scheduler and "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if self.scaler and "scaler_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
