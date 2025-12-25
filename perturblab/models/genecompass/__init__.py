"""GeneCompass: Knowledge-Informed Cross-Species Foundation Model.

GeneCompass is a BERT-based foundation model for single-cell RNA-seq analysis.
It integrates five types of biological prior knowledge to enable enhanced gene 
expression prediction, cross-species transfer learning, zero-shot cell type 
prediction, and gene expression imputation.

References:
    GeneCompass: Decoding Universal Gene Expression Signatures Across Species 
    and Sequencing Platforms. bioRxiv 2023.

Dependencies:
    Required: transformers
    Install: pip install perturblab[genecompass]
"""

from perturblab.core.model_registry import register_lazy_models

from .config import GeneCompassConfig
from .io import GeneCompassInput, GeneCompassOutput, MaskedLMOutputBoth

__all__ = [
    "GeneCompassConfig",
    "GeneCompassInput",
    "GeneCompassOutput",
    "MaskedLMOutputBoth",
    "requirements",
    "dependencies",
    "GeneCompassModel",
]

requirements = ["transformers"]
dependencies = []


def _get_models_registry():
    """Lazily imports MODELS to avoid circular dependency."""
    from perturblab.models import MODELS
    return MODELS


GENECOMPASS_REGISTRY = _get_models_registry().child("GeneCompass")
GENECOMPASS_COMPONENTS = GENECOMPASS_REGISTRY.child("components")

# Register main model
register_lazy_models(
    registry=GENECOMPASS_REGISTRY,
    models={
        "default": "GeneCompassModel",
        "GeneCompassModel": "GeneCompassModel",
    },
    base_module="perturblab.models.genecompass._modeling",
    requirements=requirements,
    dependencies=dependencies,
)

# Register all components from _modeling
from perturblab.models.genecompass._modeling import __all__ as genecompass_components

component_models = {name: name for name in genecompass_components if name != "GeneCompassModel"}

register_lazy_models(
    registry=GENECOMPASS_COMPONENTS,
    models=component_models,
    base_module="perturblab.models.genecompass._modeling",
    requirements=requirements,
    dependencies=dependencies,
)

try:
    from ._modeling.model import GeneCompassModel
except ImportError:
    pass 
