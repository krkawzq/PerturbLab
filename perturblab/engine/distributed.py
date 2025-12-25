"""Distributed training engine for PerturbLab.

Extends the base Trainer to support multi-GPU training via PyTorch DDP
(Distributed Data Parallel). Compatible with torchrun and torch.distributed.launch.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from perturblab.utils import get_logger

from .trainner import Trainer

logger = get_logger()

__all__ = ["DistributedTrainer"]


class DistributedTrainer(Trainer):
    """Distributed Data Parallel (DDP) trainer for multi-GPU training.

    Seamlessly scales training across multiple GPUs/nodes while inheriting all
    functionality from the base Trainer. Automatically handles:
    - Process group initialization
    - Model wrapping with DDP
    - Gradient synchronization
    - Metric aggregation across processes
    - Checkpoint management (main process only)

    Args:
        *args: Positional arguments passed to base Trainer.
        local_rank: Local process rank. If None, auto-detects from environment.
        find_unused_parameters: DDP config for models with conditional flows.
            Defaults to False.
        sync_batchnorm: Whether to convert BatchNorm to SyncBatchNorm.
            Defaults to False.
        **kwargs: Keyword arguments passed to base Trainer.

    Examples:
        >>> from perturblab.engine import DistributedTrainer
        >>> from perturblab.models import MODELS
        >>> from perturblab.methods.gears import build_loss
        >>>
        >>> # Create trainer (same as regular Trainer)
        >>> trainer = DistributedTrainer(
        ...     model=model,
        ...     loss_fn=loss_fn,
        ...     optimizer=optimizer,
        ...     device='cuda'  # Will be overridden to cuda:{local_rank}
        ... )
        >>>
        >>> # Train (same API)
        >>> trainer.fit(train_loader, val_loader, epochs=20)
        >>>
        >>> # Run with torchrun
        >>> # $ torchrun --nproc_per_node=4 train_script.py

    Notes:
        Launch with torchrun:
            $ torchrun --nproc_per_node=4 train.py
            $ torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 train.py

        The trainer automatically:
        - Detects distributed environment
        - Wraps model with DDP
        - Synchronizes gradients
        - Averages metrics across processes
        - Saves checkpoints only on rank 0
    """

    def __init__(
        self,
        *args,
        local_rank: Optional[int] = None,
        find_unused_parameters: bool = False,
        sync_batchnorm: bool = False,
        **kwargs,
    ):
        # Setup distributed environment first
        self._setup_distributed(local_rank)

        # Override device to use local_rank
        kwargs['device'] = f"cuda:{self.local_rank}"

        # Initialize base trainer
        super().__init__(*args, **kwargs)

        # Keep reference to unwrapped model
        self.unwrapped_model = self.model

        # Convert BatchNorm to SyncBatchNorm if requested
        if sync_batchnorm:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if self.is_main_process:
                logger.info("✓ Converted BatchNorm to SyncBatchNorm")

        # Wrap model with DDP
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=find_unused_parameters,
        )

        if self.is_main_process:
            logger.info(f"✓ Model wrapped with DDP (world_size={self.world_size})")

    def _setup_distributed(self, local_rank: Optional[int]):
        """Initializes the distributed process group.

        Args:
            local_rank: Local process rank. If None, auto-detects from environment.
        """
        # Detect local rank
        if local_rank is not None:
            self.local_rank = local_rank
        else:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Detect world size and global rank
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.global_rank = int(os.environ.get("RANK", 0))

        # Initialize process group if not already initialized
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        # Set CUDA device
        torch.cuda.set_device(self.local_rank)

        if self.is_main_process:
            logger.info(
                f"🚀 Initialized DDP: "
                f"world_size={self.world_size}, "
                f"local_rank={self.local_rank}, "
                f"global_rank={self.global_rank}"
            )

    @property
    def is_main_process(self) -> bool:
        """Returns True if this is the main process (rank 0)."""
        return self.global_rank == 0

    def _detect_input_class(self) -> Optional[type]:
        """Detects Input class, handling DDP wrapper.

        DDP wraps the model, so we need to access model.module for attributes.
        """
        # Get underlying model (unwrap DDP if needed)
        model_to_inspect = self.model.module if isinstance(self.model, DDP) else self.model

        # Check for input_class attribute
        if hasattr(model_to_inspect, 'input_class'):
            return model_to_inspect.input_class

        if hasattr(model_to_inspect, 'InputClass'):
            return model_to_inspect.InputClass

        # Check forward signature
        try:
            sig = inspect.signature(model_to_inspect.forward)
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                if param.annotation != inspect.Parameter.empty:
                    return param.annotation
        except Exception:
            pass

        return None

    def _reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Aggregates metrics across all processes via all_reduce.

        Args:
            metrics: Dictionary of metric name to value (float).

        Returns:
            Dictionary with averaged metrics across all processes.
        """
        reduced_metrics = {}

        for key, value in metrics.items():
            # Convert to tensor
            tensor = torch.tensor(value, device=self.device, dtype=torch.float32)

            # All-reduce (sum)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

            # Average
            reduced_metrics[key] = (tensor.item() / self.world_size)

        return reduced_metrics

    def train_epoch(self, train_loader: DataLoader, **loss_kwargs) -> Dict[str, float]:
        """Trains for one epoch with DistributedSampler epoch setting.

        Args:
            train_loader: Training DataLoader with DistributedSampler.
            **loss_kwargs: Dynamic loss parameters.

        Returns:
            Dictionary with averaged training metrics across all processes.
        """
        # Set epoch for DistributedSampler (crucial for proper shuffling)
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(self.current_epoch)

        # Run local training
        local_metrics = super().train_epoch(train_loader, **loss_kwargs)

        # Synchronize metrics across processes
        if isinstance(local_metrics, dict):
            global_metrics = self._reduce_metrics(local_metrics)
        else:
            # If base trainer returns float (just loss)
            global_metrics = self._reduce_metrics({'loss': local_metrics})
            global_metrics = global_metrics['loss']

        return global_metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, **loss_kwargs) -> Dict[str, float]:
        """Validates with metric synchronization across processes.

        Args:
            val_loader: Validation DataLoader.
            **loss_kwargs: Dynamic loss parameters.

        Returns:
            Dictionary with averaged validation metrics.
        """
        # Run local validation
        local_metrics = super().validate(val_loader, **loss_kwargs)

        # Synchronize metrics
        if isinstance(local_metrics, dict):
            global_metrics = self._reduce_metrics(local_metrics)
        else:
            global_metrics = self._reduce_metrics({'loss': local_metrics})
            global_metrics = global_metrics['loss']

        return global_metrics

    def save_checkpoint(self, path: str | Path):
        """Saves checkpoint only on main process.

        Args:
            path: Path to save checkpoint.
        """
        if not self.is_main_process:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save unwrapped model state (model.module for DDP)
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        # Save scheduler if exists
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save scaler if using AMP
        if hasattr(self, 'scaler') and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"💾 Checkpoint saved to {path}")

    def load_checkpoint(self, path: str | Path):
        """Loads checkpoint on all processes.

        Args:
            path: Path to checkpoint file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Load on current device
        checkpoint = torch.load(path, map_location=self.device)

        # Load into unwrapped model
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore training state
        if hasattr(self, 'current_epoch'):
            self.current_epoch = checkpoint.get('epoch', 0) + 1

        # Load scheduler and scaler if exist
        if hasattr(self, 'scheduler') and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if hasattr(self, 'scaler') and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if self.is_main_process:
            logger.info(f"📂 Checkpoint loaded from {path}")

        # Synchronization barrier
        dist.barrier()

    @staticmethod
    def create_distributed_loader(
        dataset,
        batch_size: int,
        is_train: bool = True,
        num_workers: int = 0,
        **loader_kwargs
    ) -> DataLoader:
        """Creates a DataLoader with DistributedSampler.

        Static helper method for creating DDP-compatible DataLoaders.

        Args:
            dataset: PyTorch Dataset.
            batch_size: Batch size per GPU.
            is_train: Whether this is training data (enables shuffling).
            num_workers: Number of data loading workers.
            **loader_kwargs: Additional DataLoader arguments.

        Returns:
            DataLoader with DistributedSampler.

        Examples:
            >>> from perturblab.engine import DistributedTrainer
            >>>
            >>> train_loader = DistributedTrainer.create_distributed_loader(
            ...     train_dataset,
            ...     batch_size=32,
            ...     is_train=True,
            ...     num_workers=4
            ... )
        """
        # Auto-detect rank and world size from environment
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Create sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=is_train,
        )

        # Create DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,  # Sampler handles shuffling
            num_workers=num_workers,
            **loader_kwargs
        )

    def cleanup(self):
        """Cleans up distributed resources.

        Should be called at the end of training to properly close the process group.

        Examples:
            >>> trainer = DistributedTrainer(...)
            >>> try:
            ...     trainer.fit(train_loader, val_loader, epochs=20)
            ... finally:
            ...     trainer.cleanup()
        """
        if dist.is_initialized():
            dist.destroy_process_group()
            if self.is_main_process:
                logger.info("🔒 Distributed process group destroyed")

