"""Core components for PerturbLab.

Provides fundamental abstractions and base classes:
- Dataset: Base class for all datasets
- TorchDataset: PyTorch-compatible dataset
- Resource: Abstract interface for lazy-loadable resources
- ResourceRegistry: Registry for managing collections of resources
- ModelRegistry: Registry for managing model classes with decorator support

Note:
For model access with smart lazy loading, use:
    from perturblab.models import MODELS
"""

from .dataset import Dataset, TorchDataset
from .resource import Resource
from .resource_registry import ResourceRegistry
from .model_registry import ModelRegistry

__all__ = [
    # Dataset classes
    "Dataset",
    "TorchDataset",
    # Resource system
    "Resource",
    "ResourceRegistry",
    # Model registry (base class only, use perturblab.models.MODELS for instances)
    "ModelRegistry",
]

