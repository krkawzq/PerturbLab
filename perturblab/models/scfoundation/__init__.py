"""scFoundation: Large-scale Foundation Model for Single-cell Transcriptomics.

This module implements the scFoundation model for single-cell RNA-seq analysis.

References:
    Wang et al. (2023). "scFoundation: Large-scale Foundation Model for
    Single-cell Transcriptomics." bioRxiv.

Copyright (c) 2023 BioMap (Beijing) Intelligence Technology Limited
Licensed under the MIT License
"""

from perturblab.core.model_registry import register_lazy_models

from .config import scFoundationConfig
from .io import scFoundationInput, scFoundationOutput

__all__ = [
    "scFoundationConfig",
    "scFoundationInput",
    "scFoundationOutput",
    "scFoundationModel",
    "scFoundation",  # Alias
    "requirements",
    "dependencies",
]

requirements = []
dependencies = ["local_attention", "fast_transformers"]


def _get_models_registry():
    """Lazily imports MODELS to avoid circular dependency."""
    from perturblab.models import MODELS

    return MODELS


SCFOUNDATION_REGISTRY = _get_models_registry().child("scFoundation")
SCFOUNDATION_COMPONENTS = SCFOUNDATION_REGISTRY.child("components")

# Register main model
register_lazy_models(
    registry=SCFOUNDATION_REGISTRY,
    models={
        "default": "scFoundationModel",
        "scFoundationModel": "scFoundationModel",
    },
    base_module="perturblab.models.scfoundation._modeling",
    requirements=requirements,
    dependencies=dependencies,
)

# Register all components from _modeling
from perturblab.models.scfoundation._modeling import __all__ as scfoundation_components

component_models = {name: name for name in scfoundation_components if name != "scFoundationModel"}

register_lazy_models(
    registry=SCFOUNDATION_COMPONENTS,
    models=component_models,
    base_module="perturblab.models.scfoundation._modeling",
    requirements=requirements,
    dependencies=dependencies,
)

def __getattr__(name: str):
    """Lazy load scFoundation model class on attribute access."""
    if name == "scFoundationModel":
        try:
            from ._modeling.model import scFoundationModel
            return scFoundationModel
        except ImportError as e:
            from perturblab.utils import DependencyError
            raise DependencyError(
                f"scFoundationModel requires: {', '.join(requirements or ['torch'])}. "
                f"Optional dependencies: {', '.join(dependencies)}\n"
                f"Install them with: pip install torch {' '.join(dependencies)}"
            ) from e
    elif name == "scFoundation":
        # Alias for scFoundationModel
        return __getattr__("scFoundationModel")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
