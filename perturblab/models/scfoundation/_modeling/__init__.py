"""scFoundation model implementations."""

from .components import (  # Embeddings; Transformer; Performer attention; Reversible layers
    AbsolutePositionalEmbedding,
    AutoDiscretizationEmbedding,
    Chunk,
    Deterministic,
    FastAttention,
    FeedForward,
    Gene2VecPositionalEmbedding,
    Performer,
    PerformerModule,
    PreLayerNorm,
    PreScaleNorm,
    RandomPositionalEmbedding,
    ReversibleBlock,
    ReversibleSequence,
    ReZero,
    SelfAttention,
    SequentialSequence,
    SequentialSequenceGAU,
    Transformer,
)
from .model import scFoundationModel

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
