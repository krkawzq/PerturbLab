"""Gene similarity computation from GO annotations.

This module provides functions for computing gene-gene similarity networks
from Gene Ontology (GO) annotations using bipartite graph projection.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from perturblab.types import BipartiteGraph, GeneVocab, WeightedGraph

from ._bipartite import project_bipartite_graph

__all__ = [
    "compute_gene_similarity_from_go",
    "compute_gene_similarity_from_go_df",
]


def compute_gene_similarity_from_go(
    gene_to_go: dict[str, set[str]],
    *,
    similarity: Literal["jaccard", "overlap", "cosine"] = "jaccard",
    threshold: float = 0.1,
    num_workers: int = 1,
    show_progress: bool = False,
) -> tuple[GeneVocab, WeightedGraph]:
    """Computes gene-gene similarity network from gene-GO annotations.

    Uses bipartite graph projection to compute similarities between genes
    based on their shared GO terms.

    Args:
        gene_to_go: Dictionary mapping gene names to sets of GO term IDs.
            Example: {'KRAS': {'GO:0001', 'GO:0002'}}
        similarity: Similarity metric ('jaccard', 'overlap', 'cosine').
            Defaults to 'jaccard'.
        threshold: Minimum similarity to include an edge. Defaults to 0.1.
        num_workers: Number of parallel workers. Defaults to 1.
        show_progress: Whether to show progress bar. Defaults to False.

    Returns:
        Tuple of (vocab, graph):
            - vocab: GeneVocab containing the gene list
            - graph: WeightedGraph with similarity edges

    Examples:
        >>> from perturblab.tools import compute_gene_similarity_from_go
        >>>
        >>> gene_to_go = {
        ...     'KRAS': {'GO:0001', 'GO:0002', 'GO:0003'},
        ...     'TP53': {'GO:0002', 'GO:0003', 'GO:0004'},
        ...     'MYC': {'GO:0003', 'GO:0004', 'GO:0005'},
        ... }
        >>> vocab, graph = compute_gene_similarity_from_go(
        ...     gene_to_go,
        ...     similarity='jaccard',
        ...     threshold=0.1
        ... )
    """
    # Create mapping of GO terms to indices
    all_go_terms = set()
    for go_terms in gene_to_go.values():
        all_go_terms.update(go_terms)

    go_to_idx = {go: idx for idx, go in enumerate(sorted(all_go_terms))}

    # Create gene name to index mapping (sorted for consistency)
    gene_names = sorted(gene_to_go.keys())
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}

    # Build edge list for BipartiteGraph
    edges = []
    for gene, go_terms in gene_to_go.items():
        gene_idx = gene_to_idx[gene]
        for go_term in go_terms:
            go_idx = go_to_idx[go_term]
            edges.append((gene_idx, go_idx))

    # Create BipartiteGraph
    shape = (len(gene_names), len(go_to_idx))
    bg = BipartiteGraph(edges, shape=shape)

    # Project to gene-gene similarity
    graph = project_bipartite_graph(
        bg,
        source_names=gene_names,
        similarity=similarity,
        threshold=threshold,
        num_workers=num_workers,
        show_progress=show_progress,
    )

    # Create GeneVocab
    vocab = GeneVocab(gene_names)

    return vocab, graph


def compute_gene_similarity_from_go_df(
    gene_to_go: dict[str, set[str]],
    *,
    similarity: Literal["jaccard", "overlap", "cosine"] = "jaccard",
    threshold: float = 0.1,
    num_workers: int = 1,
    show_progress: bool = False,
) -> pd.DataFrame:
    """DataFrame wrapper for compute_gene_similarity_from_go.

    Args:
        gene_to_go: Dictionary mapping gene names to GO term sets.
        similarity: Similarity metric. Defaults to 'jaccard'.
        threshold: Minimum similarity threshold. Defaults to 0.1.
        num_workers: Number of parallel workers. Defaults to 1.
        show_progress: Whether to show progress bar. Defaults to False.

    Returns:
        DataFrame with columns ['source', 'target', 'weight'].

    Examples:
        >>> from perturblab.tools import compute_gene_similarity_from_go_df
        >>>
        >>> # Returns DataFrame for backwards compatibility
        >>> df = compute_gene_similarity_from_go_df(
        ...     gene_to_go,
        ...     similarity='jaccard'
        ... )
    """
    vocab, graph = compute_gene_similarity_from_go(
        gene_to_go,
        similarity=similarity,
        threshold=threshold,
        num_workers=num_workers,
        show_progress=show_progress,
    )

    # Convert to DataFrame
    from ._df_graph_converting import weighted_graph_to_dataframe

    return weighted_graph_to_dataframe(graph, include_node_names=True)
