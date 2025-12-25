"""scGPT model implementations."""

from .model import (
    scGPTBaseModel,
    scGPTModel,
    scGPTMultiOmicModel,
    scGPTPerturbationModel,
)
from .components import (
    # Encoders
    GeneEncoder,
    PositionalEncoding,
    ContinuousValueEncoder,
    CategoryValueEncoder,
    BatchLabelEncoder,
    # Decoders
    ExprDecoder,
    MVCDecoder,
    ClsDecoder,
    AdversarialDiscriminator,
    # Attention
    FastTransformerEncoderWrapper,
    FlashTransformerEncoderLayer,
    # Normalization
    DomainSpecificBatchNorm1d,
    DomainSpecificBatchNorm2d,
    # Gradient reversal
    grad_reverse,
    GradReverse,
    # Misc
    Similarity,
    generate_square_subsequent_mask,
)

__all__ = [
    # Main models
    "scGPTBaseModel",
    "scGPTModel",
    "scGPTMultiOmicModel",
    "scGPTPerturbationModel",
    # Components - Encoders
    "GeneEncoder",
    "PositionalEncoding",
    "ContinuousValueEncoder",
    "CategoryValueEncoder",
    "BatchLabelEncoder",
    # Components - Decoders
    "ExprDecoder",
    "MVCDecoder",
    "ClsDecoder",
    "AdversarialDiscriminator",
    # Components - Attention
    "FastTransformerEncoderWrapper",
    "FlashTransformerEncoderLayer",
    # Components - Normalization
    "DomainSpecificBatchNorm1d",
    "DomainSpecificBatchNorm2d",
    # Components - Gradient reversal
    "grad_reverse",
    "GradReverse",
    # Components - Misc
    "Similarity",
    "generate_square_subsequent_mask",
]
