"""Model registry with lazy loading support.

This module provides a unified registration system for all PerturbLab models with
intelligent lazy loading to avoid dependency conflicts.

Architecture:
    - Framework models use `register_lazy()` for declarative registration
    - User models use `@MODELS.register()` decorator for immediate registration
    - Models are only imported when first accessed via `build()` or `get()`

Usage:
    >>> from perturblab.models import MODELS
    >>>
    >>> # Access via registry (triggers lazy loading)
    >>> model = MODELS.build("GEARS.default", num_genes=1000)
    >>>
    >>> # List all available models (includes unloaded models)
    >>> print(MODELS.list_keys(recursive=True))
    >>>
    >>> # User-defined models (decorator registration)
    >>> @MODELS.register()
    >>> class MyCustomModel(nn.Module):
    ...     def __init__(self, hidden_dim):
    ...         super().__init__()
    ...         self.linear = nn.Linear(hidden_dim, hidden_dim)
"""

import importlib
import sys
from pathlib import Path

from perturblab.core import ModelRegistry
from perturblab.utils import get_logger

logger = get_logger()

__all__ = ["MODELS"]


MODELS = ModelRegistry("MODELS")


def _auto_import_model_packages():
    """Automatically imports all model package `__init__.py` files to trigger registration.
    
    This function scans the models directory and imports each model package's
    `__init__.py` (but not the actual model code in `_modeling`). This ensures
    all `register_lazy()` calls are executed, making models discoverable without
    actually loading their implementations.
    """
    try:
        current_file = Path(__file__).resolve()
        models_path = current_file.parent
        
        if not models_path.exists():
            logger.debug(f"Models directory not found: {models_path}")
            return
        
        for model_dir in models_path.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith("_"):
                continue
            
            model_name = model_dir.name
            module_name = f"perturblab.models.{model_name}"
            
            try:
                if module_name not in sys.modules:
                    importlib.import_module(module_name)
                    logger.debug(f"Registered model package: {model_name}")
            except Exception as e:
                logger.debug(f"Failed to import {module_name}: {e}")
    
    except Exception as e:
        logger.warning(f"Failed to auto-import model packages: {e}")


_auto_import_model_packages()
