"""scFoundation: Large-scale Foundation Model for Single-cell Transcriptomics

This module implements the scFoundation model for single-cell RNA-seq analysis.

Model Variants:
    - scFoundationModel: Main MAE-Autobin model

Components:
    All embedding and transformer components are available via the components sub-registry.

References
----------
.. [1] Wang et al. (2023). "scFoundation: Large-scale Foundation Model for
       Single-cell Transcriptomics." bioRxiv.

Copyright (c) 2023 BioMap (Beijing) Intelligence Technology Limited
Licensed under the MIT License
"""

from perturblab.models import MODELS
from perturblab.utils import DependencyError

from .config import scFoundationConfig
from .io import scFoundationInput, scFoundationOutput


__all__ = [
    "scFoundationConfig",
    "scFoundationInput",
    "scFoundationOutput",
    "SCFOUNDATION_REGISTRY",
    "SCFOUNDATION_COMPONENTS",
]


# Create scFoundation sub-registry for models
SCFOUNDATION_REGISTRY = MODELS.child("scFoundation")

# Create components sub-registry
SCFOUNDATION_COMPONENTS = SCFOUNDATION_REGISTRY.child("components")


# Register scFoundation models with dependency checking
try:
    from ._modeling.model import scFoundationModel
    
    # Register the main model
    SCFOUNDATION_REGISTRY.register("scFoundationModel")(scFoundationModel)
    SCFOUNDATION_REGISTRY.register("default")(scFoundationModel)
    
    # Add to __all__ if successfully imported
    __all__.append("scFoundationModel")
    
except (DependencyError, ImportError):
    # Dependencies not satisfied - models won't be available
    pass


# Register components (embeddings, transformers)
try:
    from ._modeling.components import (
        # Embeddings
        AutoDiscretizationEmbedding,
        RandomPositionalEmbedding,
        # Transformers
        Transformer,
    )
    
    # Register embeddings
    SCFOUNDATION_COMPONENTS.register("AutoDiscretizationEmbedding")(AutoDiscretizationEmbedding)
    SCFOUNDATION_COMPONENTS.register("RandomPositionalEmbedding")(RandomPositionalEmbedding)
    
    # Register transformer
    SCFOUNDATION_COMPONENTS.register("Transformer")(Transformer)
    
    # Add components to __all__
    __all__.extend([
        "AutoDiscretizationEmbedding",
        "RandomPositionalEmbedding",
        "Transformer",
    ])
    
except (DependencyError, ImportError):
    # Components not available
    pass

