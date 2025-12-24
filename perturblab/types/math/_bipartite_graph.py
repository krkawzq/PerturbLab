"""Bipartite graph data structure with efficient query operations."""

from typing import Literal

import numpy as np
import scipy.sparse as sparse

# Avoid circular import: delay import to method level
# from perturblab.kernels.mapping import bipartite_graph_query


class BipartiteGraph:
    """Efficient bipartite graph representation using CSR sparse matrix.

    Supports both binary (unweighted) and weighted graphs with optimized
    batch query operations for retrieving node neighbors.

    Args:
        edges: List of (source, target) index pairs.
        weights: Optional edge weights. If None, creates binary graph.
        shape: Optional explicit shape (n_source, n_target).
            If None, inferred from max indices in edges.
        dtype: Data type for weights (default: bool for binary, float32 for weighted).
    """

    def __init__(
        self,
        edges: list[tuple[int, int]],
        weights: list[float] | np.ndarray | None = None,
        shape: tuple[int, int] | None = None,
        dtype: type = None,
    ):
        if shape is None:
            shape = self._infer_shape_from_edges(edges)

        if weights is None:
            self.graph_type = "binary"
            self.graph = self._create_csr_graph(shape, edges, None, dtype or bool)
        else:
            self.graph_type = "weighted"
            if len(edges) != len(weights):
                raise ValueError(
                    f"Number of edges ({len(edges)}) must match "
                    f"number of weights ({len(weights)})"
                )
            self.graph = self._create_csr_graph(shape, edges, weights, dtype or np.float32)

    @staticmethod
    def _infer_shape_from_edges(
        edges: list[tuple[int, int]],
    ) -> tuple[int, int]:
        """Infer graph shape from edge list.

        Args:
            edges: List of (source, target) index pairs.

        Returns:
            tuple[int, int]: Shape (n_source, n_target).
        """
        if not edges:
            return (0, 0)

        sources = [edge[0] for edge in edges]
        targets = [edge[1] for edge in edges]

        if any(s < 0 for s in sources) or any(t < 0 for t in targets):
            raise ValueError("Edge indices must be non-negative")

        n_source = max(sources) + 1
        n_target = max(targets) + 1
        return n_source, n_target

    @staticmethod
    def _create_csr_graph(
        shape: tuple[int, int],
        edges: list[tuple[int, int]],
        weights: list[float] | np.ndarray | None,
        dtype: type,
    ) -> sparse.csr_matrix:
        """Create CSR sparse matrix from edges.

        Args:
            shape: Graph shape (n_source, n_target).
            edges: List of (source, target) index pairs.
            weights: Optional edge weights.
            dtype: Data type for matrix values.

        Returns:
            sparse.csr_matrix: CSR sparse matrix representation.
        """
        if not edges:
            return sparse.csr_matrix(shape, dtype=dtype)

        sources = [edge[0] for edge in edges]
        targets = [edge[1] for edge in edges]
        data = weights if weights is not None else [True] * len(edges)

        return sparse.csr_matrix((data, (sources, targets)), shape=shape, dtype=dtype)

    @classmethod
    def from_adjacency_list(
        cls,
        adjacency: list[list[int]],
        weights: list[list[float]] | None = None,
        n_target: int | None = None,
    ) -> "BipartiteGraph":
        """Construct from adjacency list representation.

        Args:
            adjacency: List where adjacency[i] contains target indices
                connected to source node i.
            weights: Optional weights for each edge in adjacency.
            n_target: Number of target nodes. If None, inferred from max index.

        Returns:
            BipartiteGraph: New graph instance.
        """
        edges = []
        edge_weights = [] if weights is not None else None

        for source_idx, targets in enumerate(adjacency):
            for local_idx, target_idx in enumerate(targets):
                edges.append((source_idx, target_idx))
                if weights is not None:
                    edge_weights.append(weights[source_idx][local_idx])

        if n_target is None and edges:
            n_target = max(t for _, t in edges) + 1
        elif n_target is None:
            n_target = 0

        shape = (len(adjacency), n_target)
        return cls(edges, edge_weights, shape)

    @classmethod
    def from_csr_matrix(
        cls,
        matrix: sparse.csr_matrix,
    ) -> "BipartiteGraph":
        """Construct from existing CSR matrix.

        Args:
            matrix: CSR sparse matrix.

        Returns:
            BipartiteGraph: New graph instance wrapping the matrix.
        """
        instance = cls.__new__(cls)
        instance.graph = matrix
        instance.graph_type = "binary" if matrix.dtype == bool else "weighted"
        return instance

    def query(
        self,
        sources: list[int] | np.ndarray,
        check_bounds: bool = True,
    ) -> list[np.ndarray]:
        """Query neighbors for source nodes.

        Args:
            sources: Source node indices to query.
            check_bounds: Whether to check index bounds.

        Returns:
            list[np.ndarray]: List where item i contains target node indices
                connected to sources[i].
        """
        # Import here to avoid circular import
        from perturblab.kernels.mapping import bipartite_graph_query

        return bipartite_graph_query(self.graph, sources, check_bounds)

    def get_neighbors(
        self,
        source: int,
        check_bounds: bool = True,
    ) -> np.ndarray:
        """Get neighbors for a single source node.

        Args:
            source: Source node index.
            check_bounds: Whether to check index bounds.

        Returns:
            np.ndarray: Array of target node indices.
        """
        return self.query([source], check_bounds)[0]

    def get_weights(
        self,
        sources: list[int] | np.ndarray,
    ) -> list[np.ndarray]:
        """Get edge weights for source nodes.

        Args:
            sources: Source node indices to query.

        Returns:
            list[np.ndarray]: List where item i contains weights of edges
                from sources[i].
        """
        if self.graph_type == "binary":
            raise ValueError("Binary graphs do not have weights")

        results = []
        for source in sources:
            start = self.graph.indptr[source]
            end = self.graph.indptr[source + 1]
            results.append(self.graph.data[start:end])
        return results

    def degree(
        self,
        axis: Literal[0, 1] = 0,
    ) -> np.ndarray:
        """Compute node degrees.

        Args:
            axis: 0 for source node degrees, 1 for target node degrees.

        Returns:
            np.ndarray: Array of node degrees.
        """
        return np.asarray(self.graph.sum(axis=1 - axis)).flatten()

    def T(self) -> "BipartiteGraph":
        """Create transposed graph.

        Returns:
            BipartiteGraph: New graph with swapped source/target nodes.
        """
        transposed = BipartiteGraph.__new__(BipartiteGraph)
        transposed.graph_type = self.graph_type
        transposed.graph = self.graph.T.tocsr()
        return transposed

    @property
    def shape(self) -> tuple[int, int]:
        """Get graph shape.

        Returns:
            tuple[int, int]: (n_source, n_target).
        """
        return self.graph.shape

    @property
    def n_source(self) -> int:
        """Number of source nodes."""
        return self.graph.shape[0]

    @property
    def n_target(self) -> int:
        """Number of target nodes."""
        return self.graph.shape[1]

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return self.graph.nnz

    def to_adjacency_list(self) -> list[list[int]]:
        """Convert to adjacency list representation.

        Returns:
            list[list[int]]: List where item i contains target indices
                connected to source node i.
        """
        adjacency = []
        for i in range(self.n_source):
            start = self.graph.indptr[i]
            end = self.graph.indptr[i + 1]
            adjacency.append(self.graph.indices[start:end].tolist())
        return adjacency

    def to_edge_list(
        self,
        include_weights: bool = False,
    ) -> list[tuple[int, int]] | list[tuple[int, int, float]]:
        """Convert to edge list representation.

        Args:
            include_weights: Whether to include edge weights.

        Returns:
            list: List of (source, target) or (source, target, weight) tuples.
        """
        coo = self.graph.tocoo()
        if include_weights and self.graph_type == "weighted":
            return list(zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()))
        else:
            return list(zip(coo.row.tolist(), coo.col.tolist()))

    def is_function(self) -> bool:
        """Check if graph represents a function.

        A bipartite graph is a function if each source node connects
        to exactly one target node.

        Returns:
            bool: True if every source node has exactly one neighbor.
        """
        source_degrees = self.degree(axis=0)
        return np.all(source_degrees == 1)

    def is_total_function(self) -> bool:
        """Check if graph represents a total function.

        Alias for is_function() - a total function means every source
        node is mapped to exactly one target node.

        Returns:
            bool: True if every source node has exactly one neighbor.
        """
        return self.is_function()

    def is_partial_function(self) -> bool:
        """Check if graph represents a partial function.

        A bipartite graph is a partial function if each source node
        connects to at most one target node.

        Returns:
            bool: True if every source node has at most one neighbor.
        """
        source_degrees = self.degree(axis=0)
        return np.all(source_degrees <= 1)

    def is_injective(self) -> bool:
        """Check if graph represents an injective function.

        A bipartite graph is injective (one-to-one) if:
        1. Each source node connects to exactly one target node (is a function)
        2. Each target node is connected by at most one source node

        Returns:
            bool: True if the graph is an injective function.
        """
        if not self.is_function():
            return False

        target_degrees = self.degree(axis=1)
        return np.all(target_degrees <= 1)

    def is_surjective(self) -> bool:
        """Check if graph represents a surjective function.

        A bipartite graph is surjective (onto) if:
        1. Each source node connects to exactly one target node (is a function)
        2. Each target node is connected by at least one source node

        Returns:
            bool: True if the graph is a surjective function.
        """
        if not self.is_function():
            return False

        target_degrees = self.degree(axis=1)
        return np.all(target_degrees >= 1)

    def is_bijective(self) -> bool:
        """Check if graph represents a bijective function.

        A bipartite graph is bijective (one-to-one and onto) if:
        1. It is both injective and surjective
        2. Equivalently: n_source == n_target and each node has degree 1

        Returns:
            bool: True if the graph is a bijective function.
        """
        if self.n_source != self.n_target:
            return False

        return self.is_injective() and self.is_surjective()

    def get_function_mapping(self) -> dict[int, int] | None:
        """Get function mapping if graph represents a function.

        Returns:
            dict[int, int]: Dictionary mapping source to target indices,
                or None if graph is not a function.
        """
        if not self.is_function():
            return None

        mapping = {}
        for source in range(self.n_source):
            neighbors = self.get_neighbors(source, check_bounds=False)
            if len(neighbors) > 0:
                mapping[source] = int(neighbors[0])

        return mapping

    def get_inverse_mapping(self) -> dict[int, list[int]]:
        """Get inverse mapping from target to source nodes.

        Returns:
            dict[int, list[int]]: Dictionary mapping each target index
                to list of source indices connected to it.
        """
        inverse = {}
        transposed = self.T()

        for target in range(self.n_target):
            sources = transposed.get_neighbors(target, check_bounds=False)
            if len(sources) > 0:
                inverse[target] = sources.tolist()

        return inverse

    def save(self, path: str) -> None:
        """Save bipartite graph to NPZ file.

        Uses scipy.sparse.save_npz to efficiently store the sparse matrix.

        Args:
            path: Path to save the graph (NPZ format).
        """
        sparse.save_npz(path, self.graph)

    @classmethod
    def load(cls, path: str) -> "BipartiteGraph":
        """Load bipartite graph from NPZ file.

        Args:
            path: Path to load the graph from.

        Returns:
            BipartiteGraph: Loaded graph instance.
        """
        matrix = sparse.load_npz(path)

        # Convert to CSR format if not already
        if not sparse.isspmatrix_csr(matrix):
            matrix = matrix.tocsr()

        # Create instance and set attributes
        instance = cls.__new__(cls)
        instance.graph = matrix

        # Determine graph type from dtype
        if matrix.dtype == bool:
            instance.graph_type = "binary"
        else:
            instance.graph_type = "weighted"

        return instance

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"BipartiteGraph(shape={self.shape}, n_edges={self.n_edges}, "
            f"type='{self.graph_type}')"
        )

    def __len__(self) -> int:
        """Return number of source nodes."""
        return self.n_source
