"""scFoundation model implementations."""

from .model import scFoundationModel
from .components import (
    # Embeddings
    AutoDiscretizationEmbedding,
    RandomPositionalEmbedding,
    # Transformer
    Transformer,
    # Performer attention
    FastAttention,
    ReZero,
    PreScaleNorm,
    PreLayerNorm,
    Chunk,
    FeedForward,
    SelfAttention,
    AbsolutePositionalEmbedding,
    Gene2VecPositionalEmbedding,
    Performer,
    PerformerModule,
    # Reversible layers
    Deterministic,
    ReversibleBlock,
    SequentialSequence,
    SequentialSequenceGAU,
    ReversibleSequence,
)

__all__ = [
    # Main model
    "scFoundationModel",
    # Embeddings
    "AutoDiscretizationEmbedding",
    "RandomPositionalEmbedding",
    # Transformer
    "Transformer",
    # Performer attention
    "FastAttention",
    "ReZero",
    "PreScaleNorm",
    "PreLayerNorm",
    "Chunk",
    "FeedForward",
    "SelfAttention",
    "AbsolutePositionalEmbedding",
    "Gene2VecPositionalEmbedding",
    "Performer",
    "PerformerModule",
    # Reversible layers
    "Deterministic",
    "ReversibleBlock",
    "SequentialSequence",
    "SequentialSequenceGAU",
    "ReversibleSequence",
]
