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
    "scELMoModel",
    "scELMo",  # Alias
    "requirements",
    "dependencies",
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

def __getattr__(name: str):
    """Lazy load scELMo model class on attribute access."""
    if name == "scELMoModel":
        try:
            from ._modeling.model import scELMoModel
            return scELMoModel
        except ImportError as e:
            from perturblab.utils import DependencyError
            raise DependencyError(
                f"scELMoModel requires: {', '.join(requirements or ['torch'])}\n"
            ) from e
    elif name == "scELMo":
        # Alias for scELMoModel
        return __getattr__("scELMoModel")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
