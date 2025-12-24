"""scGPT model components.

This package contains all reusable components for scGPT models.
Components are organized by functionality:
- encoders.py: Gene, value, batch, and positional encoders
- decoders.py: Expression, MVC, and classification decoders
- attention.py: Fast transformer and flash attention implementations
- dsbn.py: Domain-specific batch normalization
- grad_reverse.py: Gradient reversal layer for adversarial training
- misc.py: Utility modules (Similarity, etc.)
"""

from .encoders import (
    GeneEncoder,
    PositionalEncoding,
    ContinuousValueEncoder,
    CategoryValueEncoder,
    BatchLabelEncoder,
)
from .decoders import (
    ExprDecoder,
    MVCDecoder,
    ClsDecoder,
    AdversarialDiscriminator,
)
from .attention import (
    FastTransformerEncoderWrapper,
    FlashTransformerEncoderLayer,
)
from .dsbn import (
    DomainSpecificBatchNorm1d,
    DomainSpecificBatchNorm2d,
)
from .grad_reverse import (
    grad_reverse,
    GradReverse,
)
from .misc import (
    Similarity,
    generate_square_subsequent_mask,
)


__all__ = [
    # Encoders
    "GeneEncoder",
    "PositionalEncoding",
    "ContinuousValueEncoder",
    "CategoryValueEncoder",
    "BatchLabelEncoder",
    # Decoders
    "ExprDecoder",
    "MVCDecoder",
    "ClsDecoder",
    "AdversarialDiscriminator",
    # Attention
    "FastTransformerEncoderWrapper",
    "FlashTransformerEncoderLayer",
    # Batch normalization
    "DomainSpecificBatchNorm1d",
    "DomainSpecificBatchNorm2d",
    # Gradient reversal
    "grad_reverse",
    "GradReverse",
    # Misc
    "Similarity",
    "generate_square_subsequent_mask",
]
