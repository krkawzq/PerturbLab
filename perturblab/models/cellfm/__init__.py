"""CellFM: Cell Foundation Model.

This module implements CellFM, a foundation model for single-cell transcriptomics
that uses Retention mechanism for efficient sequence modeling.

References:
    CellFM paper (to be published)

Copyright (c) 2023 CellFM Authors
Licensed under CC BY-NC-ND 4.0
"""

from perturblab.core.model_registry import register_lazy_models

from .config import CellFMConfig
from .io import CellFMInput, CellFMOutput

__all__ = [
    "CellFMConfig",
    "CellFMInput",
    "CellFMOutput",
    "CellFMModel",
    "CellFM",  # Alias
    "requirements",
    "dependencies",
]

requirements = ["torch"]
dependencies = []


def _get_models_registry():
    """Lazily imports MODELS to avoid circular dependency."""
    from perturblab.models import MODELS

    return MODELS


CELLFM_REGISTRY = _get_models_registry().child("CellFM")
CELLFM_COMPONENTS = CELLFM_REGISTRY.child("components")

# Register main model
register_lazy_models(
    registry=CELLFM_REGISTRY,
    models={
        "default": "CellFMModel",
        "CellFMModel": "CellFMModel",
    },
    base_module="perturblab.models.cellfm._modeling",
    requirements=requirements,
    dependencies=dependencies,
)

# Register all components from _modeling
from perturblab.models.cellfm._modeling import __all__ as cellfm_components

component_models = {name: name for name in cellfm_components if name != "CellFMModel"}

register_lazy_models(
    registry=CELLFM_COMPONENTS,
    models=component_models,
    base_module="perturblab.models.cellfm._modeling",
    requirements=requirements,
    dependencies=dependencies,
)


def __getattr__(name: str):
    """Lazy load CellFM model class on attribute access."""
    if name == "CellFMModel":
        try:
            from ._modeling.model import CellFMModel

            return CellFMModel
        except ImportError as e:
            from perturblab.utils import DependencyError

            raise DependencyError(
                f"CellFMModel requires: {', '.join(requirements)}\n"
                f"Install them with: pip install {' '.join(requirements)}"
            ) from e
    elif name == "CellFM":
        # Alias for CellFMModel
        return __getattr__("CellFMModel")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
