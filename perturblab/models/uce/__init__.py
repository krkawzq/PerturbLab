"""UCE: Universal Cell Embeddings.

This module implements the UCE (Universal Cell Embeddings) model for generating
universal cell representations from single-cell RNA sequencing data.

References:
    Rosen et al. (2023). "Universal Cell Embeddings: A Foundation Model
    for Cell Biology." bioRxiv.
    https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1

Copyright (c) 2023 SNAP Lab, Stanford University
Licensed under the MIT License
"""

from perturblab.core.model_registry import register_lazy_models

from .config import UCEConfig
from .io import UCEInput, UCEOutput, UCEPredictInput, UCEPredictOutput

__all__ = [
    "UCEConfig",
    "UCEInput",
    "UCEOutput",
    "UCEPredictInput",
    "UCEPredictOutput",
    "UCEModel",
    "requirements",
    "dependencies",
]

requirements = []
dependencies = ["accelerate"]


def _get_models_registry():
    """Lazily imports MODELS to avoid circular dependency."""
    from perturblab.models import MODELS
    return MODELS


UCE_REGISTRY = _get_models_registry().child("UCE")
UCE_COMPONENTS = UCE_REGISTRY.child("components")

# Register main model
register_lazy_models(
    registry=UCE_REGISTRY,
    models={
        "default": "UCEModel",
        "UCEModel": "UCEModel",
    },
    base_module="perturblab.models.uce._modeling",
    requirements=requirements,
    dependencies=dependencies,
)

# Register PositionalEncoding component
register_lazy_models(
    registry=UCE_COMPONENTS,
    models={"PositionalEncoding": "PositionalEncoding"},
    base_module="perturblab.models.uce._modeling",
    requirements=requirements,
    dependencies=dependencies,
)

try:
    from ._modeling.model import UCEModel
except ImportError:
    pass
