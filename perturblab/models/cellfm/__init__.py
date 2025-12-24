"""CellFM: Cell Foundation Model

This module implements CellFM, a foundation model for single-cell transcriptomics
that uses Retention mechanism for efficient sequence modeling.

Model Variants:
    - CellFMModel: Main retention-based model

Components:
    All retention layers, encoders, and decoders are available via the components sub-registry.

References
----------
.. [1] CellFM paper (to be published)

Copyright (c) 2023 CellFM Authors
Licensed under CC BY-NC-ND 4.0
"""

from perturblab.models import MODELS
from perturblab.utils import DependencyError

from .config import CellFMConfig
from .io import CellFMInput, CellFMOutput


__all__ = [
    "CellFMConfig",
    "CellFMInput",
    "CellFMOutput",
    "CELLFM_REGISTRY",
    "CELLFM_COMPONENTS",
]


# Create CellFM sub-registry for models
CELLFM_REGISTRY = MODELS.child("CellFM")

# Create components sub-registry
CELLFM_COMPONENTS = CELLFM_REGISTRY.child("components")


# Register CellFM model with dependency checking
try:
    from ._modeling import CellFMModel
    
    # Register the main model
    CELLFM_REGISTRY.register("CellFMModel")(CellFMModel)
    CELLFM_REGISTRY.register("default")(CellFMModel)
    
    # Add to __all__ if successfully imported
    __all__.append("CellFMModel")
    
except (DependencyError, ImportError):
    # Dependencies not satisfied - models won't be available
    pass


# Register components
try:
    from ._modeling.components import (
        # Core Model
        FinetuneModel,
        MaskedMSE,
        BCELoss,
        # Encoders/Decoders
        FFN,
        ValueEncoder,
        ValueDecoder,
        CellwiseDecoder,
        # Retention Layers
        SiLU,
        Kernel,
        LoraBlock,
        SRMSNorm,
        DropPath,
        MHRetention,
        GatedLinearUnit,
        RetentionLayer,
    )
    
    # Register core components
    CELLFM_COMPONENTS.register("FinetuneModel")(FinetuneModel)
    CELLFM_COMPONENTS.register("MaskedMSE")(MaskedMSE)
    CELLFM_COMPONENTS.register("BCELoss")(BCELoss)
    
    # Register encoders/decoders
    CELLFM_COMPONENTS.register("FFN")(FFN)
    CELLFM_COMPONENTS.register("ValueEncoder")(ValueEncoder)
    CELLFM_COMPONENTS.register("ValueDecoder")(ValueDecoder)
    CELLFM_COMPONENTS.register("CellwiseDecoder")(CellwiseDecoder)
    
    # Register retention layers
    CELLFM_COMPONENTS.register("SiLU")(SiLU)
    CELLFM_COMPONENTS.register("Kernel")(Kernel)
    CELLFM_COMPONENTS.register("LoraBlock")(LoraBlock)
    CELLFM_COMPONENTS.register("SRMSNorm")(SRMSNorm)
    CELLFM_COMPONENTS.register("DropPath")(DropPath)
    CELLFM_COMPONENTS.register("MHRetention")(MHRetention)
    CELLFM_COMPONENTS.register("GatedLinearUnit")(GatedLinearUnit)
    CELLFM_COMPONENTS.register("RetentionLayer")(RetentionLayer)
    
    # Add components to __all__
    __all__.extend([
        "FinetuneModel", "MaskedMSE", "BCELoss",
        "FFN", "ValueEncoder", "ValueDecoder", "CellwiseDecoder",
        "SiLU", "Kernel", "LoraBlock", "SRMSNorm", "DropPath",
        "MHRetention", "GatedLinearUnit", "RetentionLayer",
    ])
    
except (DependencyError, ImportError):
    # Components not available
    pass

