"""DataFrame and WeightedGraph conversion utilities.

This module provides bidirectional conversion between pandas DataFrames
(edge list format) and PerturbLab's WeightedGraph type.
"""


import numpy as np
import pandas as pd

from perturblab.types import WeightedGraph

__all__ = [
    "dataframe_to_weighted_graph",
    "weighted_graph_to_dataframe",
]


def dataframe_to_weighted_graph(
    df: pd.DataFrame,
    node_names: list[str] | None = None,
    make_undirected: bool = True,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str | None = "weight",
    default_weight: float = 1.0,
) -> WeightedGraph:
    """Converts pandas DataFrame to WeightedGraph with unweighted support.

    This utility converts edge lists (commonly used in GEARS and graph libraries)
    to PerturbLab's WeightedGraph type. Supports both weighted and unweighted
    edges, string node names, and integer indices.

    Args:
        df: DataFrame with edge information.
        node_names: Node names. If None and df contains string node IDs,
            infers from data.
        make_undirected: If True and reverse edges are missing, adds them
            automatically. Set to False if the DataFrame already contains
            both directions.
        source_col: Column name for source nodes. Defaults to "source".
        target_col: Column name for target nodes. Defaults to "target".
        weight_col: Column name for edge weights. If None, creates unweighted
            graph with default_weight for all edges. Defaults to "weight".
        default_weight: Default weight for edges when weight_col is None.
            Defaults to 1.0.

    Returns:
        WeightedGraph with edges from the DataFrame.

    Raises:
        ValueError: If required columns are missing from DataFrame.

    Examples:
        >>> import pandas as pd
        >>> from perturblab.tools import dataframe_to_weighted_graph
        >>>
        >>> # Weighted graph with string node IDs
        >>> df = pd.DataFrame({
        ...     'source': ['TP53', 'TP53', 'KRAS'],
        ...     'target': ['KRAS', 'MYC', 'MYC'],
        ...     'weight': [0.5, 0.3, 0.7]
        ... })
        >>> graph = dataframe_to_weighted_graph(df)
        >>> print(f"Graph with {graph.n_nodes} nodes")

        >>> # Unweighted graph (no weight column)
        >>> df = pd.DataFrame({
        ...     'source': ['TP53', 'KRAS'],
        ...     'target': ['KRAS', 'MYC']
        ... })
        >>> graph = dataframe_to_weighted_graph(df, weight_col=None)
        >>> # All edges have weight 1.0

        >>> # Unweighted with custom default
        >>> graph = dataframe_to_weighted_graph(
        ...     df,
        ...     weight_col=None,
        ...     default_weight=0.5
        ... )

        >>> # Custom column names
        >>> df = pd.DataFrame({
        ...     'from': ['A', 'B'],
        ...     'to': ['B', 'C'],
        ...     'similarity': [0.8, 0.6]
        ... })
        >>> graph = dataframe_to_weighted_graph(
        ...     df,
        ...     source_col='from',
        ...     target_col='to',
        ...     weight_col='similarity'
        ... )

    Notes:
        This function maintains a clean separation between data structures
        and I/O logic by keeping conversion logic out of the WeightedGraph class.
    """
    # Check for required columns
    required_cols = [source_col, target_col]
    if weight_col is not None:
        required_cols.append(weight_col)

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"DataFrame is missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Check if DataFrame has string node IDs
    has_string_ids = isinstance(df[source_col].iloc[0], str)

    if has_string_ids:
        # Infer node names from data if not provided
        if node_names is None:
            unique_nodes = sorted(set(df[source_col]) | set(df[target_col]))
            node_names = unique_nodes

        # Create node name to index mapping
        node_to_idx = {name: i for i, name in enumerate(node_names)}

        # Convert to indices
        df = df.copy()
        df[source_col] = df[source_col].map(node_to_idx)
        df[target_col] = df[target_col].map(node_to_idx)

    # Extract edges
    if weight_col is not None and weight_col in df.columns:
        # Weighted edges
        edges = df[[source_col, target_col, weight_col]].values
    else:
        # Unweighted edges - add default weight
        edges = df[[source_col, target_col]].values
        weights = np.full((edges.shape[0], 1), default_weight, dtype=np.float64)
        edges = np.hstack([edges, weights])

    n_nodes = int(max(edges[:, 0].max(), edges[:, 1].max())) + 1

    # Ensure node_names matches n_nodes if provided
    if node_names is not None and len(node_names) != n_nodes:
        if len(node_names) < n_nodes:
            # Pad with None for missing node names
            node_names = list(node_names) + [None] * (n_nodes - len(node_names))
        else:
            # Truncate to match n_nodes
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
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Converts WeightedGraph to pandas DataFrame.

    This utility converts PerturbLab's WeightedGraph to edge list format
    (commonly used in GEARS and graph libraries).

    Args:
        graph: Graph to convert.
        include_node_names: If True and graph has node names, use them
            in the DataFrame. Otherwise, use node indices.
        source_col: Column name for source nodes. Defaults to "source".
        target_col: Column name for target nodes. Defaults to "target".
        weight_col: Column name for edge weights. Defaults to "weight".

    Returns:
        DataFrame with edge information.

    Examples:
        >>> from perturblab.types import WeightedGraph
        >>> from perturblab.tools import weighted_graph_to_dataframe
        >>>
        >>> # With node names
        >>> edges = [(0, 1, 0.5), (1, 0, 0.5)]
        >>> graph = WeightedGraph(
        ...     edges,
        ...     n_nodes=2,
        ...     node_names=['TP53', 'KRAS']
        ... )
        >>> df = weighted_graph_to_dataframe(graph)
        >>> print(df)
        #   source target  weight
        # 0   TP53  KRAS     0.5
        # 1  KRAS   TP53     0.5

        >>> # With integer indices only
        >>> df = weighted_graph_to_dataframe(graph, include_node_names=False)
        >>> print(df)
        #   source  target  weight
        # 0      0       1     0.5
        # 1      1       0     0.5

        >>> # Custom column names
        >>> df = weighted_graph_to_dataframe(
        ...     graph,
        ...     source_col='from',
        ...     target_col='to',
        ...     weight_col='similarity'
        ... )
        >>> print(df.columns)
        # Index(['from', 'to', 'similarity'], dtype='object')

    Notes:
        This function maintains a clean separation between data structures
        and I/O logic by keeping conversion logic out of the WeightedGraph class.
    """
    df = pd.DataFrame(graph.edges, columns=[source_col, target_col, weight_col])

    if include_node_names and graph.node_names is not None:
        # Map indices to names
        idx_to_name = {i: name for i, name in enumerate(graph.node_names)}
        df[source_col] = df[source_col].astype(int).map(idx_to_name)
        df[target_col] = df[target_col].astype(int).map(idx_to_name)
    else:
        df[source_col] = df[source_col].astype(int)
        df[target_col] = df[target_col].astype(int)

    return df
