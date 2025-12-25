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
    "GEARS",  # Alias for GEARSModel
    "requirements",
    "dependencies",
]

requirements = ["torch_geometric"]
dependencies = []


# Get MODELS registry - must import after perturblab.models.__init__ completes
# This import is safe because __init__.py imports model packages at the END
from perturblab.models import MODELS

GEARS_REGISTRY = MODELS.child("gears")
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


def __getattr__(name: str):
    """Lazy load GEARS model class on attribute access."""
    if name == "GEARSModel":
        try:
            from ._modeling.model import GEARSModel

            return GEARSModel
        except ImportError as e:
            from perturblab.utils import DependencyError

            raise DependencyError(
                f"GEARSModel requires: {', '.join(requirements)}\n"
                f"Install them with: pip install {' '.join(requirements)}"
            ) from e
    elif name == "GEARS":
        # Alias for GEARSModel
        return __getattr__("GEARSModel")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
