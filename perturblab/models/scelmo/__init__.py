"""scELMo: Single-Cell Embeddings from Language Models

This module implements scELMo, a non-parametric model that generates cell
embeddings by aggregating pre-computed gene embeddings from language models.

scELMo is inference-only (no training) and provides instant cell embeddings.

References
----------
.. [1] scELMo paper (to be published)

Copyright (c) 2024 scELMo Authors
"""

from perturblab.models import MODELS
from perturblab.utils import DependencyError

from .config import scELMoConfig
from .io import scELMoInput, scELMoOutput

__all__ = [
    "scELMoConfig",
    "scELMoInput",
    "scELMoOutput",
    "SCELMO_REGISTRY",
]


# Create scELMo sub-registry
SCELMO_REGISTRY = MODELS.child("scELMo")


# Register scELMo model (no dependencies, pure Python)
try:
    from ._modeling import scELMoModel

    # Register the main model
    SCELMO_REGISTRY.register("scELMoModel")(scELMoModel)
    SCELMO_REGISTRY.register("default")(scELMoModel)

    # Add to __all__ if successfully imported
    __all__.append("scELMoModel")

except (DependencyError, ImportError) as e:
    # Should not happen as scELMo has no special dependencies
    import warnings

    warnings.warn(f"Failed to import scELMoModel: {e}")
