"""GEARS: Predicting transcriptional outcomes of novel multi-gene perturbations.

This module implements the GEARS (Graph-Enhanced gene Activation and Repression Simulator)
method for predicting cellular responses to genetic perturbations using graph neural networks.

References
----------
.. [1] Roohani et al. (2023). "GEARS: Predicting transcriptional outcomes
       of novel multi-gene perturbations." Nature Methods.
       https://www.nature.com/articles/s41592-023-01905-6

Copyright (c) 2023 SNAP Lab, Stanford University
Licensed under the MIT License
"""

from perturblab.models import MODELS
from perturblab.utils import DependencyError

from .config import GEARSConfig
from .io import GEARSInput, GEARSOutput

__all__ = [
    "GEARSConfig",
    "GEARS_REGISTRY",
    "GEARSInput",
    "GEARSOutput",
]


# Create GEARS sub-registry
GEARS_REGISTRY = MODELS.child("GEARS")


# Register GEARS model with dependency checking
try:
    from ._modeling import GEARSModel

    # Register the main model
    GEARS_REGISTRY.register("GEARSModel")(GEARSModel)
    GEARS_REGISTRY.register("default")(GEARSModel)

    # Add to __all__ if successfully imported
    __all__.append("GEARSModel")

except (DependencyError, ImportError):
    # Dependencies not satisfied - models won't be available
    # But the registry, config, and IO schemas are still accessible
    pass
