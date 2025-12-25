"""scELMo: Single-Cell Embeddings from Language Models.

This module implements scELMo, a non-parametric model that generates cell
embeddings by aggregating pre-computed gene embeddings from language models.

scELMo is inference-only (no training) and provides instant cell embeddings.

References:
    scELMo paper (to be published)

Copyright (c) 2024 scELMo Authors
"""

from perturblab.core.model_registry import register_lazy_models

from .config import scELMoConfig
from .io import scELMoInput, scELMoOutput

__all__ = [
    "scELMoConfig",
    "scELMoInput",
    "scELMoOutput",
    "requirements",
    "dependencies",
    "scELMoModel",
]

requirements = []
dependencies = []


def _get_models_registry():
    """Lazily imports MODELS to avoid circular dependency."""
    from perturblab.models import MODELS

    return MODELS


SCELMO_REGISTRY = _get_models_registry().child("scELMo")

register_lazy_models(
    registry=SCELMO_REGISTRY,
    models={
        "default": "scELMoModel",
        "scELMoModel": "scELMoModel",
    },
    base_module="perturblab.models.scelmo._modeling",
    requirements=requirements,
    dependencies=dependencies,
)

try:
    from ._modeling.model import scELMoModel
except ImportError:
    pass
