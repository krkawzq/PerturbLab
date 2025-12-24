"""GEARS-specific utility functions."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from perturblab.tools import compute_gene_similarity_from_go
from perturblab.types import GeneVocab, WeightedGraph
from perturblab.utils import get_logger

logger = get_logger()

__all__ = [
    "build_perturbation_graph",
    "filter_perturbations_in_go",
    "get_perturbation_genes",
    "dataframe_to_weighted_graph",
    "weighted_graph_to_dataframe",
]


def filter_perturbations_in_go(
    perturbations: list[str] | np.ndarray,
    go_genes: set[str] | list[str],
) -> list[str]:
    """Filter perturbations to only those present in GO graph.

    This replicates GEARS's `filter_pert_in_go()` function. For GEARS models,
    only perturbations where all genes are in the GO graph can be predicted.

    Parameters
    ----------
    perturbations : list[str] or np.ndarray
        List of perturbation conditions. Can be single genes ('TP53') or
        combinations ('TP53+KRAS').
    go_genes : set[str] or list[str]
        Set of genes present in the GO graph (from perturbation graph).

    Returns
    -------
    list[str]
        Filtered list of perturbations where all genes are in GO graph.

    Examples
    --------
    >>> perts = ['TP53', 'KRAS', 'TP53+KRAS', 'UNKNOWN_GENE', 'TP53+UNKNOWN']
    >>> go_genes = {'TP53', 'KRAS', 'MYC'}
    >>> filtered = filter_perturbations_in_go(perts, go_genes)
    >>> print(filtered)
    ['TP53', 'KRAS', 'TP53+KRAS']

    Notes
    -----
    For combination perturbations (e.g., 'TP53+KRAS'), all genes in the
    combination must be present in the GO graph for the perturbation to pass.
    Control conditions ('ctrl', 'control') are always kept.
    """
    if not isinstance(go_genes, set):
        go_genes = set(go_genes)

    filtered = []
    for pert in perturbations:
        if pert == "ctrl" or pert == "control":
            # Always keep control
            filtered.append(pert)
            continue

        # Split combination perturbations
        genes = pert.split("+")

        # Check if all genes are in GO
        if all(gene in go_genes for gene in genes):
            filtered.append(pert)

    return filtered


def get_perturbation_genes(
    perturbations: list[str] | np.ndarray,
    exclude_control: bool = True,
) -> list[str]:
    """Extract unique genes from a list of perturbation conditions.

    This replicates GEARS's `get_genes_from_perts()` function. Useful for
    extracting the set of genes that need to be in the perturbation graph.

    Parameters
    ----------
    perturbations : list[str] or np.ndarray
        List of perturbation conditions. Can be single genes ('TP53') or
        combinations ('TP53+KRAS').
    exclude_control : bool, default=True
        Whether to exclude control conditions ('ctrl', 'control').

    Returns
    -------
    list[str]
        Sorted unique list of gene names.

    Examples
    --------
    >>> perts = ['ctrl', 'TP53', 'KRAS', 'TP53+KRAS', 'KRAS+MYC']
    >>> genes = get_perturbation_genes(perts)
    >>> print(genes)
    ['KRAS', 'MYC', 'TP53']

    Notes
    -----
    This function is typically used to determine which genes need to be
    included in the perturbation graph before building it.
    """
    all_genes = set()

    for pert in perturbations:
        # Skip control conditions
        if exclude_control and pert.lower() in ["ctrl", "control"]:
            continue

        # Split combination perturbations
        genes = pert.split("+")
        all_genes.update(genes)

    return sorted(all_genes)


# =============================================================================
# Graph Conversion Utilities
# =============================================================================


def dataframe_to_weighted_graph(
    df: pd.DataFrame,
    node_names: list[str] | None = None,
    make_undirected: bool = True,
) -> WeightedGraph:
    """Convert pandas DataFrame to WeightedGraph.

    This is a utility function for converting edge lists (commonly used in
    GEARS and graph libraries) to PerturbLab's WeightedGraph type.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ['source', 'target', 'weight'].
        Source and target can be either node indices (int) or node names (str).
    node_names : list[str], optional
        Node names. If None and df contains string node IDs, infers from data.
    make_undirected : bool, default=True
        If True and reverse edges are missing, adds them automatically.
        Set to False if the DataFrame already contains both directions.

    Returns
    -------
    WeightedGraph
        Graph with edges from the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'source': ['TP53', 'TP53', 'KRAS'],
    ...     'target': ['KRAS', 'MYC', 'MYC'],
    ...     'weight': [0.5, 0.3, 0.7]
    ... })
    >>> graph = dataframe_to_weighted_graph(df)
    >>> print(graph)

    Notes
    -----
    This function handles the conversion logic separately from the WeightedGraph
    class to maintain a clean separation between data structures and I/O logic.
    """
    required_cols = ["source", "target", "weight"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must have columns: {required_cols}")

    # Check if DataFrame has string node IDs
    has_string_ids = isinstance(df["source"].iloc[0], str)

    if has_string_ids:
        # Infer node names from data if not provided
        if node_names is None:
            unique_nodes = sorted(set(df["source"]) | set(df["target"]))
            node_names = unique_nodes

        # Create node name to index mapping
        node_to_idx = {name: i for i, name in enumerate(node_names)}

        # Convert to indices
        df = df.copy()
        df["source"] = df["source"].map(node_to_idx)
        df["target"] = df["target"].map(node_to_idx)

    # Extract edges
    edges = df[["source", "target", "weight"]].values
    n_nodes = int(max(edges[:, 0].max(), edges[:, 1].max())) + 1

    # Ensure node_names matches n_nodes if provided
    if node_names is not None and len(node_names) != n_nodes:
        # Pad with None or truncate as needed
        if len(node_names) < n_nodes:
            node_names = list(node_names) + [None] * (n_nodes - len(node_names))
        else:
            node_names = node_names[:n_nodes]

    # Add reverse edges if needed for undirected graph
    if make_undirected:
        edges_set = set((int(s), int(t)) for s, t, w in edges)
        reverse_edges = []
        for source, target, weight in edges:
            if (int(target), int(source)) not in edges_set:
                reverse_edges.append((target, source, weight))
        if reverse_edges:
            edges = np.vstack([edges, reverse_edges])

    return WeightedGraph(edges, n_nodes, node_names)


def weighted_graph_to_dataframe(
    graph: WeightedGraph,
    include_node_names: bool = True,
) -> pd.DataFrame:
    """Convert WeightedGraph to pandas DataFrame.

    This is a utility function for converting PerturbLab's WeightedGraph
    to edge list format (commonly used in GEARS and graph libraries).

    Parameters
    ----------
    graph : WeightedGraph
        Graph to convert.
    include_node_names : bool, default=True
        If True and graph has node names, use them in the DataFrame.
        Otherwise, use node indices.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['source', 'target', 'weight'].

    Examples
    --------
    >>> edges = [(0, 1, 0.5), (1, 0, 0.5)]
    >>> graph = WeightedGraph(edges, n_nodes=2, node_names=['TP53', 'KRAS'])
    >>> df = weighted_graph_to_dataframe(graph)
    >>> print(df)

    Notes
    -----
    This function handles the conversion logic separately from the WeightedGraph
    class to maintain a clean separation between data structures and I/O logic.
    """
    df = pd.DataFrame(graph.edges, columns=["source", "target", "weight"])

    if include_node_names and graph.node_names is not None:
        # Map indices to names
        df["source"] = (
            df["source"].astype(int).map({i: name for i, name in enumerate(graph.node_names)})
        )
        df["target"] = (
            df["target"].astype(int).map({i: name for i, name in enumerate(graph.node_names)})
        )
    else:
        df["source"] = df["source"].astype(int)
        df["target"] = df["target"].astype(int)

    return df


def build_perturbation_graph(
    gene_vocab: GeneVocab | list[str] | np.ndarray,
    *,
    go_annotation_file: str | None = None,
    similarity: Literal["jaccard", "overlap", "cosine"] = "jaccard",
    threshold: float = 0.1,
    num_workers: int = 1,
    show_progress: bool = True,
) -> WeightedGraph:
    """Build GEARS-style perturbation graph from GO annotations.

    Constructs a gene similarity network where edges represent functional similarity
    based on shared Gene Ontology (GO) term annotations. This is the core graph used
    in GEARS for modeling perturbation effects.

    Parameters
    ----------
    gene_vocab : GeneVocab or list[str] or np.ndarray
        Gene vocabulary or list of gene names to include in the graph.
        If GeneVocab is provided, uses its gene names and preserves the vocabulary.
        If list/array is provided, creates a new GeneVocab internally.
        Gene names should be gene symbols (e.g., 'TP53', 'KRAS').
    go_annotation_file : str, optional
        Path to GO annotation pickle file (gene2go mapping). If None, attempts
        to load from default location (~/.perturblab/data/go/gene2go_all.pkl)
        or downloads from Harvard Dataverse.
    similarity : {'jaccard', 'overlap', 'cosine'}, default='jaccard'
        Similarity metric to compute gene-gene similarity:

        - 'jaccard': |A ∩ B| / |A ∪ B| (default in GEARS)
        - 'overlap': |A ∩ B| / min(|A|, |B|)
        - 'cosine': |A ∩ B| / sqrt(|A| * |B|)

    threshold : float, default=0.1
        Minimum similarity to include an edge. GEARS uses 0.1 by default.
        Lower values create denser graphs, higher values create sparser graphs.
    num_workers : int, default=1
        Number of parallel workers for similarity computation.
        Set to > 1 for large gene sets (>1000 genes).
    show_progress : bool, default=True
        Whether to show progress bar during computation.

    Returns
    -------
    WeightedGraph
        Weighted undirected graph with gene similarity edges.
        - Nodes correspond to genes in the input vocabulary
        - Edge weights are similarity scores in [threshold, 1.0]
        - Graph can be queried with node names (gene symbols)

    Examples
    --------
    >>> import perturblab as pl
    >>> from perturblab.types import GeneVocab
    >>>
    >>> # Build perturbation graph for GEARS model
    >>> gene_vocab = GeneVocab(['TP53', 'KRAS', 'MYC', 'EGFR', 'BRCA1'])
    >>> pert_graph = pl.methods.gears.build_perturbation_graph(
    ...     gene_vocab,
    ...     similarity='jaccard',
    ...     threshold=0.1,
    ...     num_workers=4
    ... )
    >>>
    >>> # Query graph
    >>> print(pert_graph)
    >>> neighbors = pert_graph.neighbors('TP53')
    >>> weights = pert_graph.get_weights('TP53')
    >>>
    >>> # Convert to PyTorch Geometric format
    >>> from perturblab.tools._gears import weighted_graph_to_dataframe
    >>> edge_df = weighted_graph_to_dataframe(pert_graph)

    Notes
    -----
    **Algorithm**:

    1. Load gene-GO term annotations from Harvard Dataverse (if not cached)
    2. Filter genes to those present in GO database
    3. Compute pairwise similarity between all gene pairs based on shared GO terms
    4. Keep edges above threshold
    5. Return undirected weighted graph

    **GO Annotation File**:

    The GO annotation file (gene2go_all.pkl) is downloaded from Harvard Dataverse
    on first use and cached at `~/.perturblab/data/go/gene2go_all.pkl`.
    Format: `{gene_symbol: set(GO_term_ids)}`

    **Performance**:

    - Time complexity: O(n² * m) where n is number of genes, m is avg GO terms per gene
    - Memory: O(n²) for dense graphs
    - Parallelization: Use `num_workers > 1` for large gene sets

    **Graph Properties**:

    - Undirected (both (i,j) and (j,i) edges stored)
    - Weighted (similarity scores)
    - No self-loops
    - Connected components depend on GO annotation overlap

    See Also
    --------
    perturblab.tools.compute_gene_similarity_from_go : Core similarity computation
    perturblab.tools.project_bipartite_graph : General bipartite graph projection
    perturblab.methods.gears.filter_perturbations_in_go : Filter perturbations

    References
    ----------
    .. [1] Roohani et al. (2023). "GEARS: Predicting transcriptional outcomes
           of novel multi-gene perturbations." Nature Methods.
    """
    # Handle gene_vocab input
    if isinstance(gene_vocab, GeneVocab):
        genes = gene_vocab.itos
        logger.info(f"🧬 Building GEARS perturbation graph")
        logger.info(f"   Using provided GeneVocab: {len(genes)} genes")
    else:
        genes = list(gene_vocab) if isinstance(gene_vocab, np.ndarray) else gene_vocab
        gene_vocab = GeneVocab(genes)
        logger.info(f"🧬 Building GEARS perturbation graph")
        logger.info(f"   Created GeneVocab from gene list: {len(genes)} genes")

    # Load gene2go mapping
    if go_annotation_file is None:
        # Use default location in user's home directory
        home_dir = Path.home()
        data_dir = home_dir / ".perturblab" / "data"
        go_annotation_file = data_dir / "go" / "gene2go_all.pkl"

        if not go_annotation_file.exists():
            logger.info(f"   📥 Downloading GO annotations from Harvard Dataverse...")
            go_annotation_file.parent.mkdir(parents=True, exist_ok=True)
            from perturblab.io.download import download_http

            download_http(
                url="https://dataverse.harvard.edu/api/access/datafile/6153417",
                dest=go_annotation_file,
                desc="Downloading gene2go annotations",
            )
    else:
        go_annotation_file = Path(go_annotation_file)

    logger.info(f"   📖 Loading GO annotations: {go_annotation_file.name}")
    with open(go_annotation_file, "rb") as f:
        gene2go_all = pickle.load(f)

    logger.info(f"   Total genes in GO database: {len(gene2go_all):,}")

    # Filter for genes that exist in GO database
    genes_set = set(genes)
    gene2go = {gene: go_terms for gene, go_terms in gene2go_all.items() if gene in genes_set}

    logger.info(f"   ✓ Genes with GO annotations: {len(gene2go):,}")

    if len(gene2go) == 0:
        raise ValueError(
            "No genes found in GO database. Check that gene names match "
            "the format in the GO annotation file (usually gene symbols, e.g., 'TP53')."
        )

    # Warn about genes not in GO
    genes_not_in_go = genes_set - set(gene2go.keys())
    if genes_not_in_go:
        logger.warning(f"⚠️  {len(genes_not_in_go)} genes not in GO database (will be excluded):")
        if len(genes_not_in_go) <= 10:
            logger.warning(f"   {sorted(genes_not_in_go)}")
        else:
            sample = sorted(genes_not_in_go)[:10]
            logger.warning(f"   {sample} ... and {len(genes_not_in_go) - 10} more")

    # Compute gene-gene similarity (returns DataFrame)
    logger.info(f"   🔄 Computing pairwise gene similarities...")
    edge_df = compute_gene_similarity_from_go(
        gene2go,
        similarity=similarity,
        threshold=threshold,
        num_workers=num_workers,
        show_progress=show_progress,
    )

    # Create filtered GeneVocab with only genes in GO
    genes_in_go = sorted(gene2go.keys())
    filtered_vocab = gene_vocab.select_genes(genes_in_go, keep_order=True)

    # Convert DataFrame to WeightedGraph
    logger.info(f"   🔧 Building graph structure...")
    graph = dataframe_to_weighted_graph(
        edge_df,
        node_names=filtered_vocab.itos,
        make_undirected=False,  # Already undirected from compute_gene_similarity_from_go
    )

    logger.info(f"✅ GEARS perturbation graph built successfully:")
    logger.info(f"   Nodes: {graph.n_nodes:,}")
    logger.info(f"   Edges: {graph.n_unique_edges:,}")
    logger.info(f"   Average degree: {2 * graph.n_unique_edges / graph.n_nodes:.1f}")
    logger.info(f"   Similarity: {similarity}, threshold: {threshold}")

    return graph
