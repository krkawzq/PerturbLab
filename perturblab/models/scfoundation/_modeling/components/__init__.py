"""scFoundation Model Components.

This module provides building blocks for scFoundation models, including
embeddings, transformer backends, performer attention, and reversible layers.

Copyright (c) 2023 BioMap (Beijing) Intelligence Technology Limited
Licensed under the MIT License (see forks/scFoundation/LICENSE for details)
"""

from .embeddings import AutoDiscretizationEmbedding, RandomPositionalEmbedding
from .performer import (
    AbsolutePositionalEmbedding,
    Chunk,
    FastAttention,
    FeedForward,
    Gene2VecPositionalEmbedding,
    Performer,
    PerformerModule,
    PreLayerNorm,
    PreScaleNorm,
    ReZero,
    SelfAttention,
)
from .reversible import (
    Deterministic,
    ReversibleBlock,
    ReversibleSequence,
    SequentialSequence,
    SequentialSequenceGAU,
)
from .transformer import Transformer

__all__ = [
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
