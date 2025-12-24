"""Weighted undirected graph for gene similarity networks."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class WeightedGraph:
    """Weighted undirected graph representation.

    Stores a weighted undirected graph as an edge list with weights.
    Designed for gene similarity networks constructed from GO annotations.

    Attributes
    ----------
    edges : np.ndarray
        Edge list of shape (n_edges, 3) with columns [source, target, weight].
        For undirected graphs, both (i, j) and (j, i) are stored.
    n_nodes : int
        Number of nodes in the graph.
    node_names : Optional[List[str]]
        Optional list of node names (e.g., gene symbols).

    Examples
    --------
    >>> # Create from edge list
    >>> edges = [(0, 1, 0.5), (0, 2, 0.3), (1, 2, 0.7)]
    >>> graph = WeightedGraph(edges, n_nodes=3)
    >>>
    >>> # With node names
    >>> graph = WeightedGraph(edges, n_nodes=3, node_names=['TP53', 'KRAS', 'MYC'])
    >>>
    >>> # Query neighbors
    >>> neighbors = graph.neighbors(0)
    >>> weights = graph.get_weights(0)
    """

    def __init__(
        self,
        edges: List[Tuple[int, int, float]] | np.ndarray,
        n_nodes: int,
        node_names: Optional[List[str]] = None,
    ):
        """Initialize weighted graph.

        Parameters
        ----------
        edges : list of tuples or np.ndarray
            Edge list with (source, target, weight) tuples.
            For undirected graphs, should include both directions.
        n_nodes : int
            Total number of nodes.
        node_names : list of str, optional
            Optional node names. If provided, must have length n_nodes.
        """
        if isinstance(edges, list):
            if len(edges) == 0:
                self.edges = np.zeros((0, 3), dtype=np.float64)
            else:
                self.edges = np.array(edges, dtype=np.float64)
        else:
            self.edges = edges.astype(np.float64)

        self.n_nodes = n_nodes
        self.node_names = node_names

        if node_names is not None and len(node_names) != n_nodes:
            raise ValueError(
                f"Length of node_names ({len(node_names)}) must match " f"n_nodes ({n_nodes})"
            )

        # Build adjacency list for efficient queries
        self._adjacency: Optional[Dict[int, List[Tuple[int, float]]]] = None

    @property
    def n_edges(self) -> int:
        """Number of edges (including both directions for undirected)."""
        return len(self.edges)

    @property
    def n_unique_edges(self) -> int:
        """Number of unique edges (undirected count)."""
        return len(self.edges) // 2

    def _build_adjacency(self):
        """Build adjacency list from edge list."""
        if self._adjacency is not None:
            return

        self._adjacency = {i: [] for i in range(self.n_nodes)}

        for source, target, weight in self.edges:
            source = int(source)
            target = int(target)
            self._adjacency[source].append((target, weight))

    def neighbors(self, node: int | str) -> np.ndarray:
        """Get neighbor node indices.

        Parameters
        ----------
        node : int or str
            Node index or name.

        Returns
        -------
        np.ndarray
            Array of neighbor node indices.
        """
        if isinstance(node, str):
            if self.node_names is None:
                raise ValueError("Graph has no node names")
            node = self.node_names.index(node)

        self._build_adjacency()
        neighbors = [target for target, _ in self._adjacency[node]]
        return np.array(neighbors, dtype=np.int64)

    def get_weights(self, node: int | str) -> np.ndarray:
        """Get edge weights for a node's neighbors.

        Parameters
        ----------
        node : int or str
            Node index or name.

        Returns
        -------
        np.ndarray
            Array of edge weights corresponding to neighbors.
        """
        if isinstance(node, str):
            if self.node_names is None:
                raise ValueError("Graph has no node names")
            node = self.node_names.index(node)

        self._build_adjacency()
        weights = [weight for _, weight in self._adjacency[node]]
        return np.array(weights, dtype=np.float64)

    def get_edge_weight(self, source: int | str, target: int | str) -> float:
        """Get weight of edge between two nodes.

        Parameters
        ----------
        source, target : int or str
            Source and target node indices or names.

        Returns
        -------
        float
            Edge weight, or 0.0 if edge doesn't exist.
        """
        if isinstance(source, str):
            if self.node_names is None:
                raise ValueError("Graph has no node names")
            source = self.node_names.index(source)

        if isinstance(target, str):
            if self.node_names is None:
                raise ValueError("Graph has no node names")
            target = self.node_names.index(target)

        self._build_adjacency()

        for neighbor, weight in self._adjacency[source]:
            if neighbor == target:
                return weight

        return 0.0

    def degree(self, node: int | str) -> int:
        """Get degree (number of neighbors) of a node.

        Parameters
        ----------
        node : int or str
            Node index or name.

        Returns
        -------
        int
            Number of neighbors.
        """
        return len(self.neighbors(node))

    def to_adjacency_matrix(self, sparse: bool = False) -> np.ndarray:
        """Convert to adjacency matrix.

        Parameters
        ----------
        sparse : bool, default=False
            If True, return scipy sparse matrix. Otherwise return dense numpy array.

        Returns
        -------
        np.ndarray or scipy.sparse.csr_matrix
            Adjacency matrix of shape (n_nodes, n_nodes).
        """
        if sparse:
            from scipy.sparse import csr_matrix

            row = self.edges[:, 0].astype(int)
            col = self.edges[:, 1].astype(int)
            data = self.edges[:, 2]
            return csr_matrix((data, (row, col)), shape=(self.n_nodes, self.n_nodes))
        else:
            adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
            for source, target, weight in self.edges:
                adj[int(source), int(target)] = weight
            return adj

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"WeightedGraph(n_nodes={self.n_nodes}, "
            f"n_edges={self.n_unique_edges}, "
            f"named={self.node_names is not None})"
        )
