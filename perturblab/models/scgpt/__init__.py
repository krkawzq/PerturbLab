"""scGPT: single-cell Generative Pretrained Transformer

This module provides configuration and registry for scGPT models.

**Note**: For full scGPT functionality, please install and use the original
scGPT package: https://github.com/bowang-lab/scGPT

This module primarily provides:
1. scGPTConfig for configuration management
2. Integration with PerturbLab's model registry
3. Standardized interface for future scGPT integration

References
----------
.. [1] Cui et al. (2024). "scGPT: Toward Building a Foundation Model for
       Single-Cell Multi-omics Using Generative AI." Nature Methods.
       https://doi.org/10.1038/s41592-024-02201-0

.. [2] Original implementation:
       https://github.com/bowang-lab/scGPT

Copyright (c) 2023 Bo Wang Lab
Licensed under the MIT License
"""

from perturblab.models import MODELS
from perturblab.utils import DependencyError

from .config import scGPTConfig
from .io import scGPTInput, scGPTOutput

__all__ = [
    "scGPTConfig",
    "scGPTInput",
    "scGPTOutput",
    "SCGPT_REGISTRY",
    "SCGPT_COMPONENTS",
]


# Create scGPT sub-registry for models
SCGPT_REGISTRY = MODELS.child("scGPT")

# Create components sub-registry (for encoder/decoder components)
SCGPT_COMPONENTS = SCGPT_REGISTRY.child("components")


# Register scGPT models with dependency checking
try:
    from ._modeling.model import (
        scGPTModel,
        scGPTMultiOmicModel,
        scGPTPerturbationModel,
    )

    # Register model variants
    SCGPT_REGISTRY.register("scGPTModel")(scGPTModel)
    SCGPT_REGISTRY.register("default")(scGPTModel)
    SCGPT_REGISTRY.register("scGPTMultiOmicModel")(scGPTMultiOmicModel)
    SCGPT_REGISTRY.register("scGPTPerturbationModel")(scGPTPerturbationModel)

    # Add to __all__ if successfully imported
    __all__.extend(["scGPTModel", "scGPTMultiOmicModel", "scGPTPerturbationModel"])

except (DependencyError, ImportError):
    # Dependencies not satisfied - models won't be available
    pass


# Register components (encoders, decoders, attention layers)
try:
    from ._modeling.components import (  # Encoders; Decoders; Attention; Batch Normalization; Gradient Reversal; Misc
        AdversarialDiscriminator,
        BatchLabelEncoder,
        CategoryValueEncoder,
        ClsDecoder,
        ContinuousValueEncoder,
        DomainSpecificBatchNorm1d,
        DomainSpecificBatchNorm2d,
        ExprDecoder,
        FastTransformerEncoderWrapper,
        FlashTransformerEncoderLayer,
        GeneEncoder,
        GradReverse,
        MVCDecoder,
        PositionalEncoding,
        Similarity,
        generate_square_subsequent_mask,
        grad_reverse,
    )

    # Register encoders
    SCGPT_COMPONENTS.register("GeneEncoder")(GeneEncoder)
    SCGPT_COMPONENTS.register("PositionalEncoding")(PositionalEncoding)
    SCGPT_COMPONENTS.register("ContinuousValueEncoder")(ContinuousValueEncoder)
    SCGPT_COMPONENTS.register("CategoryValueEncoder")(CategoryValueEncoder)
    SCGPT_COMPONENTS.register("BatchLabelEncoder")(BatchLabelEncoder)

    # Register decoders
    SCGPT_COMPONENTS.register("ExprDecoder")(ExprDecoder)
    SCGPT_COMPONENTS.register("MVCDecoder")(MVCDecoder)
    SCGPT_COMPONENTS.register("ClsDecoder")(ClsDecoder)
    SCGPT_COMPONENTS.register("AdversarialDiscriminator")(AdversarialDiscriminator)

    # Register attention layers
    SCGPT_COMPONENTS.register("FastTransformerEncoderWrapper")(FastTransformerEncoderWrapper)
    SCGPT_COMPONENTS.register("FlashTransformerEncoderLayer")(FlashTransformerEncoderLayer)

    # Register normalization
    SCGPT_COMPONENTS.register("DomainSpecificBatchNorm1d")(DomainSpecificBatchNorm1d)
    SCGPT_COMPONENTS.register("DomainSpecificBatchNorm2d")(DomainSpecificBatchNorm2d)

    # Add components to __all__
    __all__.extend(
        [
            "GeneEncoder",
            "PositionalEncoding",
            "ContinuousValueEncoder",
            "CategoryValueEncoder",
            "BatchLabelEncoder",
            "ExprDecoder",
            "MVCDecoder",
            "ClsDecoder",
            "AdversarialDiscriminator",
            "FastTransformerEncoderWrapper",
            "FlashTransformerEncoderLayer",
            "DomainSpecificBatchNorm1d",
            "DomainSpecificBatchNorm2d",
            "grad_reverse",
            "GradReverse",
            "Similarity",
            "generate_square_subsequent_mask",
        ]
    )

except (DependencyError, ImportError):
    # Components not available - dependencies missing
    pass
