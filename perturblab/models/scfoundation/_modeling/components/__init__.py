"""scFoundation Model Components.

This module provides building blocks for scFoundation models, including
embeddings, transformer backends, and reversible layers.

Copyright (c) 2023 BioMap (Beijing) Intelligence Technology Limited
Licensed under the MIT License (see forks/scFoundation/LICENSE for details)
"""

from .embeddings import AutoDiscretizationEmbedding, RandomPositionalEmbedding
from .transformer import Transformer

# Performer and Reversible components have external dependencies
# and are imported lazily

__all__ = [
    "AutoDiscretizationEmbedding",
    "RandomPositionalEmbedding",
    "Transformer",
]
