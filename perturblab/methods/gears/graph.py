"""GEARS graph construction utilities.

This module provides functions for building gene similarity graphs for GEARS models:
- GO-based functional similarity graphs
- Co-expression networks from single-cell data
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import scipy.sparse as sp

from perturblab.data.resources import load_dataset
from perturblab.tools import compute_gene_similarity_from_go
from perturblab.types import CellData, GeneGraph, GeneVocab, PerturbationData, WeightedGraph
from perturblab.utils import get_logger

logger = get_logger()

__all__ = [
    "build_go_similarity_graph",
    "build_coexpression_graph",
    "compute_pearson_correlation",
]


def compute_pearson_correlation(X: np.ndarray) -> np.ndarray:
    """Computes pairwise Pearson correlation between genes (columns).

    Args:
        X: Expression matrix of shape (n_cells, n_genes).

    Returns:
        Correlation matrix of shape (n_genes, n_genes).

    Examples:
        >>> X = np.random.rand(100, 50)
        >>> corr = compute_pearson_correlation(X)
        >>> print(corr.shape)  # (50, 50)
    """
    # Center the data
    X_centered = X - X.mean(axis=0, keepdims=True)

    # Compute correlation
    numer = X_centered.T @ X_centered
    denom = np.sqrt(
        (X_centered**2).sum(axis=0, keepdims=True).T @ (X_centered**2).sum(axis=0, keepdims=True)
    )

    # Avoid division by zero
    denom = np.where(denom == 0, 1e-10, denom)
    corr = numer / denom

    # Handle NaN values
    corr[np.isnan(corr)] = 0.0

    return corr


def build_go_similarity_graph(
    gene_vocab: GeneVocab | list[str] | np.ndarray | None = None,
    *,
    go_annotation_file: str | Path | None = None,
    similarity_metric: Literal["jaccard", "overlap", "cosine"] = "jaccard",
    threshold: float = 0.1,
    k: int | None = None,
    num_workers: int = 1,
    show_progress: bool = True,
) -> GeneGraph:
    """Builds a gene similarity graph based on shared GO annotations.

    Constructs the core GO-based relational graph for GEARS by computing pairwise
    functional similarities between genes based on their Gene Ontology (GO) terms.

    Args:
        gene_vocab: List of genes or GeneVocab object. If None, uses all genes
            from the GO annotation file. Defaults to None.
        go_annotation_file: Path to `gene2go_all.pkl`. If None, automatically
            loads from resource system (downloads if needed).
        similarity_metric: Metric to compute similarity between gene GO term sets.
            Options: 'jaccard', 'overlap', 'cosine'. Defaults to 'jaccard'.
        threshold: Minimum similarity score (0.0 to 1.0) to keep an edge.
            Edges with similarity below this value are discarded. Defaults to 0.1.
        k: If provided, keeps only top-k edges for each gene. If None, keeps
            all edges above threshold. Defaults to None.
        num_workers: Number of parallel workers for similarity computation.
            Defaults to 1 (single-threaded).
        show_progress: Whether to display progress bars during computation.
            Defaults to True.

    Returns:
        GeneGraph with GO-based functional similarity edges and vocabulary.

    Raises:
        ValueError: If gene_vocab is provided but no genes are found in GO database.

    Examples:
        >>> from perturblab.types import GeneVocab
        >>> from perturblab.methods.gears import build_go_similarity_graph
        >>>
        >>> # Auto-initialize from all GO genes
        >>> gene_graph = build_go_similarity_graph(
        ...     similarity_metric='jaccard',
        ...     threshold=0.1
        ... )
        >>> print(f"Graph with {len(gene_graph.genes)} genes from GO")
        >>>
        >>> # From gene list
        >>> gene_list = ['TP53', 'KRAS', 'MYC', 'EGFR']
        >>> gene_graph = build_go_similarity_graph(
        ...     gene_list,
        ...     similarity_metric='jaccard',
        ...     threshold=0.1
        ... )
        >>>
        >>> # From GeneVocab
        >>> vocab = GeneVocab(gene_list)
        >>> gene_graph = build_go_similarity_graph(
        ...     vocab,
        ...     threshold=0.2,
        ...     k=20  # Keep top-20 neighbors per gene
        ... )
        >>>
        >>> # Access graph and vocab
        >>> print(gene_graph.genes)  # List of gene names
        >>> neighbors = gene_graph.neighbors('TP53')  # Query by name

    Notes:
        The GO annotation file is automatically cached by the resource system
        at `~/.cache/perturblab/`. Subsequent calls will use the cached version.

        The similarity metric measures overlap between GO term sets:
        - Jaccard: |A ∩ B| / |A ∪ B|
        - Overlap: |A ∩ B| / min(|A|, |B|)
        - Cosine: (A · B) / (||A|| ||B||)
    """
    # 1. Load GO annotations via resource system
    if go_annotation_file is None:
        logger.info(f"📥 Loading GO annotations from resource system...")
        go_annotation_file = load_dataset("go/gene2go_gears")
    else:
        go_annotation_file = Path(go_annotation_file)
        logger.info(f"📥 Loading GO annotations from: {go_annotation_file}")

    # 2. Load GO data
    logger.info(f"📖 Loading GO annotations...")
    with open(go_annotation_file, "rb") as f:
        gene2go_all = pickle.load(f)

    # 3. Process gene_vocab parameter
    if gene_vocab is None:
        # Use all genes from GO annotation file
        genes = sorted(gene2go_all.keys())
        logger.info(f"🧬 Building GO similarity graph with all genes from GO: {len(genes)} genes")
        gene2go = gene2go_all
        vocab = GeneVocab(genes)
    else:
        # Filter to requested genes
        if isinstance(gene_vocab, GeneVocab):
            genes = gene_vocab.itos
            logger.info(f"🧬 Building GO similarity graph from GeneVocab: {len(genes)} genes")
            vocab = gene_vocab
        else:
            genes = list(gene_vocab) if isinstance(gene_vocab, np.ndarray) else gene_vocab
            logger.info(f"🧬 Building GO similarity graph from gene list: {len(genes)} genes")
            vocab = GeneVocab(genes)

        # Filter GO data to only include requested genes
        genes_set = set(genes)
        gene2go = {g: go for g, go in gene2go_all.items() if g in genes_set}

        genes_found_count = len(gene2go)
        logger.info(f"✓ Found GO terms for {genes_found_count:,} / {len(genes):,} genes")

        if genes_found_count == 0:
            raise ValueError(
                "No genes found in GO database. Please check gene symbol format "
                "(expected: 'TP53', 'KRAS', etc.)."
            )

        # Log missing genes
        missing_genes = genes_set - set(gene2go.keys())
        if missing_genes:
            sample_missing = list(missing_genes)[:5]
            logger.warning(
                f"⚠️  {len(missing_genes)} genes not in GO database. " f"Sample: {sample_missing}"
            )

    # 4. Compute gene similarity using optimized tool
    logger.info(f"🔄 Computing {similarity_metric} similarity (threshold={threshold})...")

    vocab_out, graph = compute_gene_similarity_from_go(
        gene2go,
        similarity=similarity_metric,
        threshold=threshold,
        num_workers=num_workers,
        show_progress=show_progress,
    )

    # Overwrite with GeneVocab if compute_gene_similarity_from_go returns a different vocab type
    if not isinstance(vocab_out, GeneVocab):
        vocab = GeneVocab(list(vocab_out.itos))
    else:
        vocab = vocab_out

    # 5. Apply top-k filtering if requested
    if k is not None:
        logger.info(f"🔧 Filtering to top-{k} edges per gene...")

        # Get edges as array
        edges = graph.edges

        # Build adjacency structure for filtering
        from collections import defaultdict

        node_edges = defaultdict(list)

        for source, target, weight in edges:
            node_edges[int(target)].append((int(source), int(target), weight))

        # Keep top-k edges per target node
        filtered_edges = []
        for target_idx, edge_list in node_edges.items():
            # Sort by weight descending and keep top-k
            edge_list_sorted = sorted(edge_list, key=lambda x: x[2], reverse=True)
            filtered_edges.extend(edge_list_sorted[:k])

        # Recreate weighted graph with filtered edges
        filtered_weighted_graph = WeightedGraph(filtered_edges, n_nodes=graph.n_nodes)

        # Recreate GeneGraph
        graph = GeneGraph(filtered_weighted_graph, vocab)

        logger.info(f"   Kept {graph.graph.n_unique_edges} edges after top-{k} filtering")

    logger.info(
        f"✅ GO similarity graph built: "
        f"{graph.n_nodes} nodes, {graph.graph.n_unique_edges} unique edges"
    )

    return graph


def build_coexpression_graph(
    data: ad.AnnData | CellData | PerturbationData,
    threshold: float = 0.4,
    k: int = 20,
    vocab: GeneVocab | None = None,
    use_control_only: bool = True,
) -> GeneGraph:
    """Builds gene co-expression network from expression data.

    Constructs a gene co-expression network by computing Pearson correlations
    between genes and keeping the top-k most correlated genes for each gene.
    Uses absolute correlation values to capture both positive and negative
    co-expression patterns.

    Args:
        data: AnnData, CellData, or PerturbationData instance.
        threshold: Minimum absolute correlation threshold. Edges with
            |correlation| below this value are removed. Defaults to 0.4.
        k: Number of top neighbors to keep for each gene. Defaults to 20.
        vocab: Optional GeneVocab. If None, creates from data.var_names.
            Defaults to None.
        use_control_only: If True and data is PerturbationData, only uses
            control cells for correlation computation. Ignored for AnnData
            and CellData. Defaults to True.

    Returns:
        GeneGraph with co-expression edges weighted by absolute correlation.

    Raises:
        ValueError: If use_control_only=True but no control cells found.

    Examples:
        >>> from perturblab.types import PerturbationData, GeneVocab
        >>> from perturblab.methods.gears import build_coexpression_graph
        >>> import anndata as ad
        >>>
        >>> # From AnnData
        >>> adata = ad.read_h5ad('data.h5ad')
        >>> gene_graph = build_coexpression_graph(
        ...     adata,
        ...     threshold=0.4,
        ...     k=20
        ... )
        >>>
        >>> # From PerturbationData (control cells only)
        >>> pert_data = PerturbationData(adata, perturbation_col='condition')
        >>> gene_graph = build_coexpression_graph(
        ...     pert_data,
        ...     threshold=0.4,
        ...     k=20,
        ...     use_control_only=True
        ... )
        >>>
        >>> # With custom vocab
        >>> custom_vocab = GeneVocab(['TP53', 'KRAS', 'MYC'])
        >>> gene_graph = build_coexpression_graph(
        ...     adata,
        ...     vocab=custom_vocab
        ... )
        >>>
        >>> # Access graph and vocab
        >>> print(gene_graph.genes)  # List of gene names
        >>> neighbors = gene_graph.neighbors('TP53')  # Query by name

    Notes:
        The function computes absolute Pearson correlation to capture both
        positive and negative co-expression. For GEARS, using control cells
        only is recommended to build an unbiased baseline network.
    """
    logger.info(f"🧬 Building co-expression network (threshold={threshold}, k={k})...")

    # Extract AnnData and expression matrix
    if isinstance(data, PerturbationData):
        # PerturbationData: can use control_only
        if use_control_only:
            ctrl_mask = data.is_control
            if not ctrl_mask.any():
                raise ValueError("No control cells found in PerturbationData")
            X = data.adata.X[ctrl_mask]
            logger.info(f"  Using {ctrl_mask.sum()} control cells from PerturbationData")
        else:
            X = data.adata.X
            logger.info(f"  Using all {X.shape[0]} cells from PerturbationData")
        adata = data.adata
    elif isinstance(data, CellData):
        # CellData: use all cells
        adata = data.adata
        X = adata.X
        logger.info(f"  Using all {X.shape[0]} cells from CellData")
    elif isinstance(data, ad.AnnData):
        # AnnData: use all cells
        adata = data
        X = adata.X
        logger.info(f"  Using all {X.shape[0]} cells from AnnData")
    else:
        raise TypeError(f"Expected AnnData, CellData, or PerturbationData, got {type(data)}")

    # Convert to dense if sparse
    if sp.issparse(X):
        X = X.toarray()

    # Compute correlation matrix
    logger.info(f"  Computing Pearson correlation...")
    corr = compute_pearson_correlation(X)
    corr_abs = np.abs(corr)

    # Get gene names and create/use vocab
    gene_names = adata.var_names.tolist()
    n_genes = len(gene_names)

    if vocab is None:
        vocab = GeneVocab(gene_names)
    else:
        # Validate vocab matches data
        if len(vocab) != n_genes or set(vocab.itos) != set(gene_names):
            logger.warning(
                f"Provided vocab doesn't match data genes. "
                f"Creating new vocab from data.var_names."
            )
            vocab = GeneVocab(gene_names)

    logger.info(f"  Selecting top-{k} neighbors for each gene...")

    # For each gene, keep top-k neighbors
    edges = []
    for i in range(n_genes):
        # Get top-k+1 genes (includes self)
        top_k_indices = np.argsort(corr_abs[i])[: -(k + 1) : -1]
        top_k_values = corr_abs[i, top_k_indices]

        for j, val in zip(top_k_indices, top_k_values):
            # Skip self-loops and edges below threshold
            if i != j and val > threshold:
                edges.append((i, j, val))

    if not edges:
        logger.warning(
            f"No edges found with threshold={threshold} and k={k}. "
            f"Consider lowering threshold or increasing k."
        )
        # Return empty graph
        weighted_graph = WeightedGraph([], n_nodes=n_genes)
        return GeneGraph(weighted_graph, vocab)

    logger.info(f"  Created {len(edges)} edges")

    # Create WeightedGraph and GeneGraph
    weighted_graph = WeightedGraph(edges, n_nodes=n_genes)
    gene_graph = GeneGraph(weighted_graph, vocab)

    logger.info(
        f"✅ Co-expression network built: "
        f"{gene_graph.n_nodes} nodes, {gene_graph.graph.n_unique_edges} unique edges"
    )

    return gene_graph
