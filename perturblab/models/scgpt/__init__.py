"""scGPT: single-cell Generative Pretrained Transformer.

This module provides configuration and registry for scGPT models.

For full scGPT functionality, install: https://github.com/bowang-lab/scGPT

References:
    Cui et al. (2024). "scGPT: Toward Building a Foundation Model for
    Single-Cell Multi-omics Using Generative AI." Nature Methods.
    https://doi.org/10.1038/s41592-024-02201-0

Copyright (c) 2023 Bo Wang Lab
Licensed under the MIT License
"""

from perturblab.core.model_registry import register_lazy_models

from .config import scGPTConfig
from .io import scGPTInput, scGPTOutput

__all__ = [
    "scGPTConfig",
    "scGPTInput",
    "scGPTOutput",
    "scGPTModel",
    "scGPTMultiOmicModel",
    "scGPTPerturbationModel",
    "requirements",
    "dependencies",
]

requirements = []
dependencies = ["flash_attn", "fast_transformers"]


def _get_models_registry():
    """Lazily imports MODELS to avoid circular dependency."""
    from perturblab.models import MODELS

    return MODELS


SCGPT_REGISTRY = _get_models_registry().child("scGPT")
SCGPT_COMPONENTS = SCGPT_REGISTRY.child("components")

# Register main models
register_lazy_models(
    registry=SCGPT_REGISTRY,
    models={
        "default": "scGPTModel",
        "scGPTModel": "scGPTModel",
        "scGPTMultiOmicModel": "scGPTMultiOmicModel",
        "scGPTPerturbationModel": "scGPTPerturbationModel",
    },
    base_module="perturblab.models.scgpt._modeling",
    requirements=requirements,
    dependencies=dependencies,
)

# Register all components from _modeling
from perturblab.models.scgpt._modeling import __all__ as scgpt_components

component_models = {
    name: name
    for name in scgpt_components
    if name.startswith(
        (
            "Gene",
            "Expr",
            "MVC",
            "Cls",
            "Positional",
            "Continuous",
            "Category",
            "Batch",
            "Fast",
            "Flash",
            "Domain",
            "Adversarial",
            "Similarity",
        )
    )
}

register_lazy_models(
    registry=SCGPT_COMPONENTS,
    models=component_models,
    base_module="perturblab.models.scgpt._modeling",
    requirements=requirements,
    dependencies=dependencies,
)

try:
    from ._modeling.model import (
        scGPTModel,
        scGPTMultiOmicModel,
        scGPTPerturbationModel,
    )
except ImportError:
    pass
