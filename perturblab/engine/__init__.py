"""Training and inference engine for PerturbLab models.

This module provides unified training infrastructure that works with any model
following PerturbLab's conventions:
- Trainer: Single-GPU/CPU training
- DistributedTrainer: Multi-GPU training with DDP
"""

from .distributed import DistributedTrainer
from .trainner import Trainer

__all__ = ["Trainer", "DistributedTrainer"]
