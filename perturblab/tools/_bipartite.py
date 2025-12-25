"""Bipartite graph projection tools.

This module provides general-purpose tools for projecting bipartite graphs
into weighted undirected graphs using various similarity metrics.
"""

from __future__ import annotations

from multiprocessing import Pool
from typing import Callable, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from perturblab.types import BipartiteGraph, WeightedGraph

from ._similarity import (
    compute_pairwise_similarities,
    cosine_similarity_sets,
    jaccard_similarity,
    overlap_coefficient,
)

__all__ = [
    "project_bipartite_graph",
    "project_bipartite_graph_df",
]


def project_bipartite_graph(
    bipartite_graph: BipartiteGraph,
    source_names: list[str] | np.ndarray | None = None,
    *,
    similarity: Literal["jaccard", "overlap", "cosine"] | Callable = "jaccard",
    threshold: float = 0.1,
    num_workers: int = 1,
    show_progress: bool = False,
) -> WeightedGraph:
    """Projects bipartite graph to weighted undirected graph using similarity.

    Given a bipartite graph (e.g., genes -> GO terms), computes pairwise
    similarity between source nodes based on their shared target nodes.

    Args:
        bipartite_graph: Input bipartite graph (source nodes -> target nodes).
        source_names: Optional names for source nodes. If None, uses indices.
        similarity: Similarity metric ('jaccard', 'overlap', 'cosine') or
            custom function: func(set1, set2) -> float. Defaults to 'jaccard'.
        threshold: Minimum similarity to include an edge. Defaults to 0.1.
        num_workers: Number of parallel workers. Defaults to 1.
        show_progress: Whether to show progress bar. Defaults to False.

    Returns:
        WeightedGraph with similarity edges between source nodes.

    Examples:
        >>> from perturblab.types import BipartiteGraph
        >>> from perturblab.tools import project_bipartite_graph
        >>>
        >>> # Gene-GO bipartite graph
        >>> edges = [(0, 0), (0, 1), (1, 1), (1, 2)]
        >>> bg = BipartiteGraph(edges, shape=(3, 3))
        >>> gene_names = ['KRAS', 'TP53', 'MYC']
        >>>
        >>> # Project to gene-gene similarity
        >>> graph = project_bipartite_graph(
        ...     bg,
        ...     source_names=gene_names,
        ...     similarity='jaccard',
        ...     threshold=0.1
        ... )
        >>> print(f"{graph.n_nodes} nodes, {graph.n_unique_edges} edges")
    """
    # Select similarity function
    if isinstance(similarity, str):
        similarity_funcs = {
            'jaccard': jaccard_similarity,
            'overlap': overlap_coefficient,
            'cosine': cosine_similarity_sets,
        }
        if similarity not in similarity_funcs:
            raise ValueError(f"Unknown similarity: {similarity}")
        similarity_func = similarity_funcs[similarity]
    else:
        similarity_func = similarity

    # Get neighbors for each source node
    n_source = bipartite_graph.shape[0]
    all_neighbors = [bipartite_graph.neighbors(i, side='source') for i in range(n_source)]

    # Compute pairwise similarities
    edges = []

    if num_workers > 1:
        # Parallel computation
        args_list = [
            (i, all_neighbors[i], all_neighbors, similarity_func, threshold)
            for i in range(n_source)
        ]

        with Pool(num_workers) as pool:
            if show_progress:
                results = list(tqdm(
                    pool.imap(_worker_wrapper, args_list),
                    total=len(args_list),
                    desc="Computing similarities"
                ))
            else:
                results = pool.map(_worker_wrapper, args_list)

        for edge_list in results:
            edges.extend(edge_list)
    else:
        # Serial computation
        iterator = range(n_source)
        if show_progress:
            iterator = tqdm(iterator, desc="Computing similarities")

        for i in iterator:
            edge_list = compute_pairwise_similarities(
                i, all_neighbors[i], all_neighbors, similarity_func, threshold
            )
            edges.extend(edge_list)

    # Add reverse edges for undirected graph
    reverse_edges = [(t, s, w) for s, t, w in edges]
    all_edges = edges + reverse_edges

    # Create WeightedGraph
    if source_names is not None:
        source_names = list(source_names)

    graph = WeightedGraph(all_edges, n_nodes=n_source, node_names=source_names)

    return graph


def _worker_wrapper(args: tuple) -> list[tuple[int, int, float]]:
    """Wrapper for parallel execution."""
    return compute_pairwise_similarities(*args)


def project_bipartite_graph_df(
    bipartite_graph: BipartiteGraph,
    source_names: list[str] | np.ndarray | None = None,
    *,
    similarity: Literal["jaccard", "overlap", "cosine"] | Callable = "jaccard",
    threshold: float = 0.1,
    num_workers: int = 1,
    show_progress: bool = False,
) -> pd.DataFrame:
    """DataFrame wrapper for project_bipartite_graph.

    Projects bipartite graph and returns edge list as DataFrame.

    Args:
        bipartite_graph: Input bipartite graph.
        source_names: Optional names for source nodes.
        similarity: Similarity metric. Defaults to 'jaccard'.
        threshold: Minimum similarity threshold. Defaults to 0.1.
        num_workers: Number of parallel workers. Defaults to 1.
        show_progress: Whether to show progress bar. Defaults to False.

    Returns:
        DataFrame with columns ['source', 'target', 'weight'].

    Examples:
        >>> from perturblab.tools import project_bipartite_graph_df
        >>>
        >>> # Returns DataFrame instead of WeightedGraph
        >>> df = project_bipartite_graph_df(
        ...     bg,
        ...     source_names=gene_names,
        ...     similarity='jaccard'
        ... )
        >>> print(df.head())
    """
    graph = project_bipartite_graph(
        bipartite_graph,
        source_names=source_names,
        similarity=similarity,
        threshold=threshold,
        num_workers=num_workers,
        show_progress=show_progress,
    )

    # Convert to DataFrame
    from ._df_graph_converting import weighted_graph_to_dataframe

    return weighted_graph_to_dataframe(graph, include_node_names=True)
