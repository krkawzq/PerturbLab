from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from perturblab.tools import project_bipartite_graph
from perturblab.types import BipartiteGraph
from perturblab.utils import get_logger

logger = get_logger()

__all__ = [
    "compute_gene_similarity_from_go",
]

# =============================================================================
# Similarity Metrics
# =============================================================================


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets.

    Jaccard similarity = |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    set1, set2 : set
        Input sets.

    Returns
    -------
    float
        Similarity score in [0, 1].

    Examples
    --------
    >>> jaccard_similarity({1, 2, 3}, {2, 3, 4})
    0.5
    """
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


def overlap_coefficient(set1: set, set2: set) -> float:
    """Compute overlap coefficient between two sets.

    Overlap coefficient = |A ∩ B| / min(|A|, |B|)

    Parameters
    ----------
    set1, set2 : set
        Input sets.

    Returns
    -------
    float
        Similarity score in [0, 1].

    Examples
    --------
    >>> overlap_coefficient({1, 2, 3}, {2, 3, 4, 5})
    0.6666...
    """
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    min_size = min(len(set1), len(set2))

    if min_size == 0:
        return 0.0

    return intersection / min_size


def cosine_similarity_sets(set1: set, set2: set) -> float:
    """Compute cosine similarity between two sets.

    Cosine similarity = |A ∩ B| / sqrt(|A| * |B|)

    Parameters
    ----------
    set1, set2 : set
        Input sets.

    Returns
    -------
    float
        Similarity score in [0, 1].

    Examples
    --------
    >>> cosine_similarity_sets({1, 2, 3}, {2, 3, 4})
    0.6666...
    """
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    denominator = np.sqrt(len(set1) * len(set2))

    if denominator == 0:
        return 0.0

    return intersection / denominator


# =============================================================================
# Worker function for parallel computation
# =============================================================================


def _compute_similarities_for_node(
    args: tuple,
) -> list[tuple[int, int, float]]:
    """Worker function to compute similarities for one source node.

    Parameters
    ----------
    args : tuple
        Tuple of (node_idx, node_neighbors, all_neighbors, similarity_func, threshold).

    Returns
    -------
    list[tuple[int, int, float]]
        List of (source, target, weight) tuples.
    """
    node_idx, node_neighbors, all_neighbors, similarity_func, threshold = args

    edge_list = []
    node_set = set(node_neighbors)

    # Compute similarity with all other nodes
    for other_idx in range(len(all_neighbors)):
        if other_idx <= node_idx:  # Skip self and already computed pairs
            continue

        other_set = set(all_neighbors[other_idx])
        similarity = similarity_func(node_set, other_set)

        if similarity > threshold:
            edge_list.append((node_idx, other_idx, similarity))

    return edge_list


def compute_gene_similarity_from_go(
    gene_to_go: dict[str, set[str]],
    *,
    similarity: Literal["jaccard", "overlap", "cosine"] = "jaccard",
    threshold: float = 0.1,
    num_workers: int = 1,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute gene-gene similarity network from gene-GO annotations.

    This is a convenience wrapper around `project_bipartite_graph` specifically
    for gene-GO networks. It's a general-purpose tool used by GEARS and other methods.

    Parameters
    ----------
    gene_to_go : dict[str, set[str]]
        Dictionary mapping gene names to sets of GO term IDs.
        Example: {'KRAS': {'GO:0001', 'GO:0002'}, 'TP53': {'GO:0002', 'GO:0003'}}
    similarity : {'jaccard', 'overlap', 'cosine'}, default='jaccard'
        Similarity metric. Default: 'jaccard'.
    threshold : float, default=0.1
        Minimum similarity to include an edge. GEARS uses 0.1.
    num_workers : int, default=1
        Number of parallel workers.
    show_progress : bool, default=True
        Whether to show progress bar.

    Returns
    -------
    pd.DataFrame
        Edge list with columns ['source', 'target', 'weight'], where source
        and target are gene names.

    Examples
    --------
    >>> gene_to_go = {
    ...     'KRAS': {'GO:0001', 'GO:0002', 'GO:0003'},
    ...     'TP53': {'GO:0002', 'GO:0003', 'GO:0004'},
    ...     'MYC': {'GO:0003', 'GO:0004', 'GO:0005'},
    ... }
    >>> similarity_graph = compute_gene_similarity_from_go(
    ...     gene_to_go, similarity='jaccard', threshold=0.1
    ... )
    >>> print(similarity_graph.head())

    Notes
    -----
    This function replicates the `make_GO` function from GEARS but with a
    more general interface and better performance.

    See Also
    --------
    project_bipartite_graph : General bipartite graph projection
    """
    logger.info(f"🧬 Building gene similarity network from GO annotations")
    logger.info(f"   Genes: {len(gene_to_go)}")

    # Convert to BipartiteGraph
    # First, create mapping of GO terms to indices
    all_go_terms = set()
    for go_terms in gene_to_go.values():
        all_go_terms.update(go_terms)

    go_to_idx = {go: idx for idx, go in enumerate(sorted(all_go_terms))}
    logger.info(f"   GO terms: {len(go_to_idx)}")

    # Create gene name to index mapping
    gene_names = sorted(gene_to_go.keys())
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}

    # Build edge list
    edges = []
    for gene, go_terms in gene_to_go.items():
        gene_idx = gene_to_idx[gene]
        for go_term in go_terms:
            go_idx = go_to_idx[go_term]
            edges.append((gene_idx, go_idx))

    logger.info(f"   Gene-GO edges: {len(edges)}")

    # Create BipartiteGraph
    shape = (len(gene_names), len(go_to_idx))
    bg = BipartiteGraph(edges, shape=shape)

    # Project to gene-gene similarity
    return project_bipartite_graph(
        bg,
        source_names=gene_names,
        similarity=similarity,
        threshold=threshold,
        num_workers=num_workers,
        show_progress=show_progress,
    )
