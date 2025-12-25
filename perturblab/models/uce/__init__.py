"""UCE: Universal Cell Embeddings

This module implements the UCE (Universal Cell Embeddings) model for generating
universal cell representations from single-cell RNA sequencing data.

References
----------
.. [1] Rosen et al. (2023). "Universal Cell Embeddings: A Foundation Model
       for Cell Biology." bioRxiv.
       https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1

Copyright (c) 2023 SNAP Lab, Stanford University
Licensed under the MIT License
"""

from perturblab.utils import DependencyError

from .config import UCEConfig
from .io import UCEInput, UCEOutput, UCEPredictInput, UCEPredictOutput

__all__ = [
    "UCEConfig",
    "UCE_REGISTRY",
    "UCEInput",
    "UCEOutput",
    "UCEPredictInput",
    "UCEPredictOutput",
]


def _get_models_registry():
    """Lazy import MODELS to avoid circular dependency."""
    from perturblab.models import MODELS
    return MODELS


# Create UCE sub-registry (lazy)
UCE_REGISTRY = _get_models_registry().child("UCE")


# Register UCE model with dependency checking
try:
    from ._modeling import UCEModel

    # Register the main model (aliased as both UCEModel and default)
    UCE_REGISTRY.register("UCEModel")(UCEModel)
    UCE_REGISTRY.register("default")(UCEModel)

    # Add to __all__ if successfully imported
    __all__.append("UCEModel")

except (DependencyError, ImportError):
    # Dependencies not satisfied - models won't be available
    # But the registry, config, and IO schemas are still accessible
    pass
