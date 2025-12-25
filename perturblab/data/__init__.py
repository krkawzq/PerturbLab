"""Data loading and resource management for PerturbLab.

This module provides:
- Dataset loading functions
- Resource management system
- Pre-configured dataset registry

Examples
--------
>>> from perturblab.data import load_dataset
>>> # Load a perturbation dataset
>>> data = load_dataset("scperturb/norman_2019")
>>> print(data)
AnnData object with n_obs × n_vars = ...
"""

from .resources import (
    File,
    Files,
    dataset_registry,
    get_dataset,
    h5adFile,
    list_datasets,
    load_dataset,
)

__all__ = [
    # Resource types
    "File",
    "Files",
    "h5adFile",
    # Registry access
    "dataset_registry",
    # Convenience functions
    "list_datasets",
    "get_dataset",
    "load_dataset",
]
