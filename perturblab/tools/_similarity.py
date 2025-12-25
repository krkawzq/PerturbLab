"""Similarity metrics for sets and vectors.

This module provides general-purpose similarity metrics that can be applied
to sets, vectors, or other mathematical objects. These metrics are used across
various analysis tasks including graph construction, clustering, and comparison.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = [
    "jaccard_similarity",
    "overlap_coefficient",
    "cosine_similarity_sets",
    "cosine_similarity_vectors",
    "pairwise_similarities",
]


def jaccard_similarity(set1: set, set2: set) -> float:
    """Computes Jaccard similarity between two sets.

    The Jaccard similarity measures the intersection divided by the union:
    J(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        set1: First set.
        set2: Second set.

    Returns:
        Similarity score in [0, 1], where 1 means identical sets.

    Examples:
        >>> jaccard_similarity({1, 2, 3}, {2, 3, 4})
        0.5
        
        >>> jaccard_similarity({'TP53', 'KRAS'}, {'KRAS', 'MYC'})
        0.3333333333333333
        
        >>> jaccard_similarity(set(), {1, 2})
        0.0

    Notes:
        Commonly used in:
        - GO term similarity (GEARS)
        - Document similarity
        - Recommendation systems
    """
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


def overlap_coefficient(set1: set, set2: set) -> float:
    """Computes overlap coefficient between two sets.

    The overlap coefficient measures intersection divided by the smaller set:
    O(A, B) = |A ∩ B| / min(|A|, |B|)

    Args:
        set1: First set.
        set2: Second set.

    Returns:
        Similarity score in [0, 1], where 1 means one set is a subset of the other.

    Examples:
        >>> overlap_coefficient({1, 2, 3}, {2, 3, 4, 5})
        0.6666666666666666
        
        >>> overlap_coefficient({1, 2}, {1, 2, 3, 4})
        1.0  # First set is subset of second
        
        >>> overlap_coefficient(set(), {1, 2})
        0.0

    Notes:
        More lenient than Jaccard similarity, especially when sets have
        different sizes. Useful for:
        - Hierarchical relationships
        - GO term similarity with different specificity levels
    """
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    min_size = min(len(set1), len(set2))

    if min_size == 0:
        return 0.0

    return intersection / min_size


def cosine_similarity_sets(set1: set, set2: set) -> float:
    """Computes cosine similarity between two sets.

    Treats sets as binary vectors and computes cosine similarity:
    cos(A, B) = |A ∩ B| / sqrt(|A| * |B|)

    Args:
        set1: First set.
        set2: Second set.

    Returns:
        Similarity score in [0, 1].

    Examples:
        >>> cosine_similarity_sets({1, 2, 3}, {2, 3, 4})
        0.6666666666666666
        
        >>> cosine_similarity_sets({'TP53', 'KRAS'}, {'KRAS', 'MYC'})
        0.5
        
        >>> cosine_similarity_sets(set(), {1, 2})
        0.0

    Notes:
        Equivalent to treating sets as binary vectors (1 if element present,
        0 otherwise) and computing vector cosine similarity. Often used in:
        - Text similarity
        - GO term similarity
        - Feature set comparison
    """
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    denominator = np.sqrt(len(set1) * len(set2))

    if denominator == 0:
        return 0.0

    return intersection / denominator


def cosine_similarity_vectors(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two vectors.

    cos(A, B) = (A · B) / (||A|| ||B||)

    Args:
        vec1: First vector, shape (n,).
        vec2: Second vector, shape (n,).

    Returns:
        Similarity score in [-1, 1], where 1 means identical direction.

    Examples:
        >>> vec1 = np.array([1, 2, 3])
        >>> vec2 = np.array([2, 4, 6])
        >>> cosine_similarity_vectors(vec1, vec2)
        1.0  # Parallel vectors
        
        >>> vec1 = np.array([1, 0, 0])
        >>> vec2 = np.array([0, 1, 0])
        >>> cosine_similarity_vectors(vec1, vec2)
        0.0  # Orthogonal vectors

    Notes:
        Standard vector cosine similarity. Commonly used for:
        - Document similarity
        - Embedding similarity
        - Expression profile similarity
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def pairwise_similarities(
    node_idx: int,
    node_neighbors: list,
    all_neighbors: list,
    similarity_func: Callable,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Computes pairwise similarities between one node and all other nodes.

    This is a general-purpose worker function for computing similarities
    in graph projection algorithms. It computes similarity between one source
    node and all subsequent nodes based on their neighbor sets.

    Args:
        node_idx: Index of the current node.
        node_neighbors: List of neighbors for the current node.
        all_neighbors: List of neighbor lists for all nodes.
        similarity_func: Similarity function taking two sets and returning float.
        threshold: Minimum similarity threshold to include an edge.

    Returns:
        List of (source, target, weight) tuples for edges above threshold.

    Examples:
        >>> from perturblab.tools import compute_pairwise_similarities, jaccard_similarity
        >>>
        >>> # Node 0 has neighbors [0, 1]
        >>> # Node 1 has neighbors [1, 2]
        >>> # Node 2 has neighbors [2, 3]
        >>> all_neighbors = [[0, 1], [1, 2], [2, 3]]
        >>>
        >>> # Compute similarities for node 0
        >>> edges = compute_pairwise_similarities(
        ...     node_idx=0,
        ...     node_neighbors=[0, 1],
        ...     all_neighbors=all_neighbors,
        ...     similarity_func=jaccard_similarity,
        ...     threshold=0.1
        ... )
        >>> # Returns: [(0, 1, similarity_score)] if above threshold

    Notes:
        This function only computes similarities with nodes that have index
        greater than node_idx to avoid duplicate computation in undirected graphs.
        Used internally by bipartite graph projection algorithms.
    """
    edge_list = []
    node_set = set(node_neighbors)

    for other_idx in range(len(all_neighbors)):
        if other_idx <= node_idx:
            continue

        other_set = set(all_neighbors[other_idx])
        similarity = similarity_func(node_set, other_set)

        if similarity > threshold:
            edge_list.append((node_idx, other_idx, similarity))

    return edge_list

