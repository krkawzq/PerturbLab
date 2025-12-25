"""GEARS: Predicting transcriptional outcomes of novel multi-gene perturbations.

This module implements the GEARS (Graph-Enhanced gene Activation and Repression Simulator)
method for predicting cellular responses to genetic perturbations using graph neural networks.

References:
    Roohani et al. (2023). "GEARS: Predicting transcriptional outcomes
    of novel multi-gene perturbations." Nature Methods.
    https://www.nature.com/articles/s41592-023-01905-6

Copyright (c) 2023 SNAP Lab, Stanford University
Licensed under the MIT License
"""

from perturblab.core.model_registry import register_lazy_models

from .config import GEARSConfig
from .io import GEARSInput, GEARSOutput

__all__ = [
    "GEARSConfig",
    "GEARSInput",
    "GEARSOutput",
    "GEARSModel",
    "requirements",
    "dependencies",
]

requirements = ["torch_geometric"]
dependencies = []


def _get_models_registry():
    """Lazily imports MODELS to avoid circular dependency."""
    from perturblab.models import MODELS
    return MODELS


GEARS_REGISTRY = _get_models_registry().child("GEARS")
GEARS_COMPONENTS = GEARS_REGISTRY.child("components")

# Register main model
register_lazy_models(
    registry=GEARS_REGISTRY,
    models={
        "default": "GEARSModel",
        "GEARSModel": "GEARSModel",
    },
    base_module="perturblab.models.gears._modeling",
    requirements=requirements,
    dependencies=dependencies,
)

# Register MLP component
register_lazy_models(
    registry=GEARS_COMPONENTS,
    models={"MLP": "MLP"},
    base_module="perturblab.models.gears._modeling",
    requirements=requirements,
    dependencies=dependencies,
)

try:
    from ._modeling.model import GEARSModel
except ImportError:
    pass
