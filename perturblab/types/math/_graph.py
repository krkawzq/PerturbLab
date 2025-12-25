"""Pure mathematical graph data structures.

This module provides domain-agnostic graph representations efficiently implemented
using NumPy and SciPy. It deals strictly with integer indices and edge weights.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sparse

__all__ = ["WeightedGraph"]


class WeightedGraph:
    """Weighted undirected graph as a pure mathematical structure.

    Stores a weighted undirected graph as an edge list. Nodes are represented
    strictly by integer indices [0, n_nodes-1].

    Attributes:
        edges (np.ndarray): Edge list of shape (n_edges, 3) with columns
            [source_idx, target_idx, weight].
        n_nodes (int): Total number of nodes in the graph.
    """

    def __init__(
        self,
        edges: List[Tuple[int, int, float]] | List[Tuple[int, int]] | np.ndarray,
        n_nodes: int,
        default_weight: float = 1.0,
    ):
        """Initializes a weighted graph.

        Args:
            edges: Edge list. Can be:
                - List of (u, v) tuples (unweighted).
                - List of (u, v, w) tuples (weighted).
                - Numpy array of shape (N, 2) or (N, 3).
            n_nodes: Total number of nodes (integer indices).
            default_weight: Weight assigned to edges if input is unweighted.
        """
        # Normalize input to numpy array
        if isinstance(edges, list):
            if len(edges) == 0:
                edges_array = np.zeros((0, 3), dtype=np.float64)
            else:
                edges_array = np.array(edges, dtype=np.float64)
        else:
            edges_array = np.asarray(edges, dtype=np.float64)

        # Handle dimensions
        if edges_array.ndim == 1 and edges_array.size > 0:
             edges_array = edges_array.reshape(1, -1)
        
        if edges_array.ndim == 1 and edges_array.size == 0:
             edges_array = np.zeros((0, 3), dtype=np.float64)

        # Normalize columns (add weights if missing)
        if edges_array.shape[1] == 2:
            weights = np.full((edges_array.shape[0], 1), default_weight, dtype=np.float64)
            self.edges = np.hstack([edges_array, weights])
        elif edges_array.shape[1] == 3:
            self.edges = edges_array
        else:
            raise ValueError(f"Edges must have 2 or 3 columns, got {edges_array.shape[1]}")

        self.n_nodes = n_nodes
        self._adjacency: Optional[Dict[int, List[Tuple[int, float]]]] = None

    @property
    def n_edges(self) -> int:
        """Total number of edges in the edge list."""
        return len(self.edges)

    @property
    def n_unique_edges(self) -> int:
        """Number of unique edges (assuming undirected storage implies duplication)."""
        return len(self.edges) // 2

    def _build_adjacency(self) -> None:
        """Lazily builds an adjacency list for O(1) neighbor lookups."""
        if self._adjacency is not None:
            return
        
        self._adjacency = {i: [] for i in range(self.n_nodes)}
        
        # Optimize loop slightly by avoiding explicit unpacking if not needed
        # but tuple unpacking is readable.
        for row in self.edges:
            source, target, weight = int(row[0]), int(row[1]), row[2]
            # Safety check for bounds
            if source < self.n_nodes:
                self._adjacency[source].append((target, weight))

    def neighbors(self, node: int) -> np.ndarray:
        """Get indices of neighbor nodes."""
        self._build_adjacency()
        if node >= self.n_nodes:
            return np.array([], dtype=np.int64)
        neighbors = [target for target, _ in self._adjacency[node]]
        return np.array(neighbors, dtype=np.int64)

    def get_weights(self, node: int) -> np.ndarray:
        """Get weights of edges connected to the node."""
        self._build_adjacency()
        if node >= self.n_nodes:
            return np.array([], dtype=np.float64)
        weights = [weight for _, weight in self._adjacency[node]]
        return np.array(weights, dtype=np.float64)

    def get_edge_weight(self, source: int, target: int) -> float:
        """Get specific edge weight or 0.0 if not found."""
        self._build_adjacency()
        if source >= self.n_nodes:
            return 0.0
            
        for neighbor, weight in self._adjacency[source]:
            if neighbor == target:
                return weight
        return 0.0

    def degree(self, node: int) -> int:
        """Get node degree."""
        return len(self.neighbors(node))

    def to_adjacency_matrix(self, sparse_fmt: bool = False) -> np.ndarray | sparse.csr_matrix:
        """Convert to adjacency matrix (dense or sparse)."""
        if self.n_edges == 0:
            mat = sparse.csr_matrix((self.n_nodes, self.n_nodes), dtype=np.float64)
        else:
            row = self.edges[:, 0].astype(int)
            col = self.edges[:, 1].astype(int)
            data = self.edges[:, 2]
            mat = sparse.csr_matrix((data, (row, col)), shape=(self.n_nodes, self.n_nodes))
        
        if sparse_fmt:
            return mat
        return mat.toarray()

    def __repr__(self) -> str:
        return f"WeightedGraph(n_nodes={self.n_nodes}, n_edges={self.n_unique_edges})"
