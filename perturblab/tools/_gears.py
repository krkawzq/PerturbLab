"""GEARS-inspired bipartite graph projection for gene similarity networks.

This module provides functions to project bipartite graphs (e.g., gene-GO term)
into weighted undirected graphs (e.g., gene-gene similarity) using various
similarity metrics.

Ported and abstracted from GEARS (https://github.com/snap-stanford/GEARS).
"""

from __future__ import annotations

from typing import Literal, Callable
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, triu
from tqdm import tqdm

from perturblab.data.datatype.math import BipartiteGraph
from perturblab.utils import get_logger

logger = get_logger()

__all__ = [
    "project_bipartite_graph",
    "compute_gene_similarity_from_go",
    "jaccard_similarity",
    "overlap_coefficient",
    "cosine_similarity_sets",
]


# =============================================================================
# Similarity Metrics
# =============================================================================

def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets.
    
    Jaccard similarity = |A ∩ B| / |A ∪ B|
    
    Parameters
    ----------
    set1, set2
        Input sets
    
    Returns
    -------
    float
        Similarity score in [0, 1]
    
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
    set1, set2
        Input sets
    
    Returns
    -------
    float
        Similarity score in [0, 1]
    
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
    set1, set2
        Input sets
    
    Returns
    -------
    float
        Similarity score in [0, 1]
    
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
    args
        Tuple of (node_idx, node_neighbors, all_neighbors, similarity_func, threshold)
    
    Returns
    -------
    list[tuple[int, int, float]]
        List of (source, target, weight) tuples
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


# =============================================================================
# Main projection function
# =============================================================================

def project_bipartite_graph(
    bipartite_graph: BipartiteGraph,
    source_names: list[str] | np.ndarray | None = None,
    *,
    similarity: Literal["jaccard", "overlap", "cosine"] | Callable = "jaccard",
    threshold: float = 0.1,
    num_workers: int = 1,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Project bipartite graph to weighted undirected graph using similarity.
    
    Given a bipartite graph (e.g., genes -> GO terms), this function computes
    pairwise similarity between source nodes based on their shared target nodes,
    creating a weighted undirected graph.
    
    This is the core algorithm used in GEARS for constructing gene similarity
    networks from Gene Ontology annotations.
    
    Parameters
    ----------
    bipartite_graph
        Input bipartite graph (source nodes -> target nodes).
        For gene-GO networks, source nodes are genes, target nodes are GO terms.
    source_names
        Optional names for source nodes. If None, uses indices.
    similarity
        Similarity metric to use:
        - 'jaccard': Jaccard similarity (default, used in GEARS)
        - 'overlap': Overlap coefficient
        - 'cosine': Cosine similarity
        - Or a custom function: func(set1, set2) -> float
    threshold
        Minimum similarity to include an edge. GEARS uses 0.1.
    num_workers
        Number of parallel workers. Set to 1 for serial computation.
    show_progress
        Whether to show progress bar.
    
    Returns
    -------
    pd.DataFrame
        Edge list with columns ['source', 'target', 'weight'].
        Nodes are identified by names if provided, else by indices.
    
    Examples
    --------
    >>> # Create gene-GO bipartite graph
    >>> edges = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]  # genes -> GO terms
    >>> bg = BipartiteGraph(edges)
    >>> gene_names = ['KRAS', 'TP53', 'MYC']
    >>> 
    >>> # Project to gene-gene similarity graph
    >>> similarity_graph = project_bipartite_graph(
    ...     bg, gene_names, similarity='jaccard', threshold=0.1
    ... )
    
    Notes
    -----
    The algorithm computes pairwise similarity for all pairs of source nodes.
    Time complexity: O(n^2 * m) where n is number of source nodes and m is
    average number of target nodes per source.
    
    For large graphs (>1000 nodes), consider using parallel computation with
    `num_workers > 1`.
    
    References
    ----------
    .. [1] Roohani et al. (2023). "GEARS: Predicting transcriptional outcomes 
           of novel multi-gene perturbations." Nature Methods.
    """
    logger.info(
        f"🔄 Projecting bipartite graph: {bipartite_graph.n_source} source nodes, "
        f"{bipartite_graph.n_target} target nodes"
    )
    
    # Select similarity function
    if isinstance(similarity, str):
        similarity_map = {
            "jaccard": jaccard_similarity,
            "overlap": overlap_coefficient,
            "cosine": cosine_similarity_sets,
        }
        if similarity not in similarity_map:
            raise ValueError(
                f"Unknown similarity metric: {similarity}. "
                f"Choose from: {list(similarity_map.keys())}"
            )
        similarity_func = similarity_map[similarity]
    else:
        similarity_func = similarity
    
    # Get neighbors for all source nodes
    logger.info(f"📊 Retrieving neighbors for all source nodes...")
    all_neighbors = bipartite_graph.to_adjacency_list()
    
    # Convert to sets for faster set operations
    all_neighbors_sets = [set(neighbors) for neighbors in all_neighbors]
    
    # Prepare arguments for parallel computation
    n_sources = bipartite_graph.n_source
    args_list = [
        (i, all_neighbors_sets[i], all_neighbors_sets, similarity_func, threshold)
        for i in range(n_sources)
    ]
    
    # Compute similarities
    logger.info(
        f"🧮 Computing pairwise similarities (method={similarity}, "
        f"threshold={threshold})..."
    )
    
    if num_workers > 1:
        logger.info(f"   Using {num_workers} parallel workers")
        with Pool(num_workers) as pool:
            if show_progress:
                all_edge_lists = list(
                    tqdm(
                        pool.imap(_compute_similarities_for_node, args_list),
                        total=n_sources,
                        desc="Computing similarities",
                    )
                )
            else:
                all_edge_lists = pool.map(_compute_similarities_for_node, args_list)
    else:
        if show_progress:
            all_edge_lists = [
                _compute_similarities_for_node(args)
                for args in tqdm(args_list, desc="Computing similarities")
            ]
        else:
            all_edge_lists = [
                _compute_similarities_for_node(args)
                for args in args_list
            ]
    
    # Flatten edge list
    edge_list = []
    for edges in all_edge_lists:
        edge_list.extend(edges)
    
    logger.info(f"✅ Found {len(edge_list)} edges above threshold {threshold}")
    
    # Convert to DataFrame
    if not edge_list:
        logger.warning("⚠️  No edges found above threshold! Consider lowering threshold.")
        df = pd.DataFrame(columns=['source', 'target', 'weight'])
    else:
        df = pd.DataFrame(edge_list, columns=['source', 'target', 'weight'])
        
        # Map indices to names if provided
        if source_names is not None:
            source_names = np.asarray(source_names)
            df['source'] = source_names[df['source'].values]
            df['target'] = source_names[df['target'].values]
    
    # Add reverse edges to make undirected
    df_reverse = df.copy()
    df_reverse.columns = ['target', 'source', 'weight']
    df_undirected = pd.concat([df, df_reverse], ignore_index=True)
    
    logger.info(f"📈 Created undirected graph: {len(df)} unique edges, "
                f"{len(df_undirected)} total edges (undirected)")
    
    return df_undirected


# =============================================================================
# Convenience function for gene-GO networks
# =============================================================================

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
    for gene-GO networks. It matches the GEARS implementation.
    
    Parameters
    ----------
    gene_to_go
        Dictionary mapping gene names to sets of GO term IDs.
        Example: {'KRAS': {'GO:0001', 'GO:0002'}, 'TP53': {'GO:0002', 'GO:0003'}}
    similarity
        Similarity metric ('jaccard', 'overlap', or 'cosine'). Default: 'jaccard'.
    threshold
        Minimum similarity to include an edge. GEARS uses 0.1.
    num_workers
        Number of parallel workers.
    show_progress
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

