"""Bipartite graph projection and similarity computation tools.

This module provides general-purpose tools for projecting bipartite graphs
(e.g., gene-GO term networks) into weighted undirected graphs (e.g., gene-gene
similarity networks) using various similarity metrics.

These are method-agnostic utilities that can be used for GEARS and other
graph-based methods. GEARS-specific logic is in perturblab.methods.gears.

Ported and abstracted from GEARS (https://github.com/snap-stanford/GEARS).

Copyright (c) 2023 SNAP Lab, Stanford University
Licensed under the MIT License
"""

from __future__ import annotations

from typing import Literal, Callable
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from perturblab.types import BipartiteGraph
from perturblab.utils import get_logger
from ._gene_similarity import (
    jaccard_similarity,
    overlap_coefficient,
    cosine_similarity_sets,
    _compute_similarities_for_node,
)

logger = get_logger()

__all__ = [
    "project_bipartite_graph",
]


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
    
    This is a general-purpose algorithm used in GEARS and other methods for
    constructing similarity networks from bipartite relationships.
    
    Parameters
    ----------
    bipartite_graph : BipartiteGraph
        Input bipartite graph (source nodes -> target nodes).
        For gene-GO networks, source nodes are genes, target nodes are GO terms.
    source_names : list[str] or np.ndarray, optional
        Optional names for source nodes. If None, uses indices.
    similarity : {'jaccard', 'overlap', 'cosine'} or callable, default='jaccard'
        Similarity metric to use:
        - 'jaccard': Jaccard similarity (default, used in GEARS)
        - 'overlap': Overlap coefficient
        - 'cosine': Cosine similarity
        - Or a custom function: func(set1, set2) -> float
    threshold : float, default=0.1
        Minimum similarity to include an edge. GEARS uses 0.1.
    num_workers : int, default=1
        Number of parallel workers. Set to 1 for serial computation.
    show_progress : bool, default=True
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

