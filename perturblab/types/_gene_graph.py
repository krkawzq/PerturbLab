"""Gene-specific graph structures with vocabulary alignment.

This module binds the mathematical WeightedGraph with a gene Vocabulary,
enabling operations like subgraph extraction by gene name and alignment
to experimental datasets (AnnData).
"""

from __future__ import annotations

import numpy as np

from .math._graph import WeightedGraph
import scipy.sparse as sparse

from ._vocab import Vocab

__all__ = ["GeneGraph"]


class GeneGraph:
    """Gene similarity graph with vocabulary mapping and alignment capabilities.

    Wraps a WeightedGraph with a Vocab. Provides advanced methods to align,
    subset, and reorder the graph to match specific experimental datasets.

    Attributes:
        graph (WeightedGraph): The underlying mathematical graph.
        vocab (Vocab): Mapping between gene names and graph indices.
    """

    def __init__(self, graph: WeightedGraph, vocab: Vocab):
        """Initializes GeneGraph.

        Args:
            graph: Underlying WeightedGraph structure.
            vocab: Vocab object for gene name mappings.

        Raises:
            ValueError: If vocab size doesn't match graph n_nodes.
        """
        if len(vocab) != graph.n_nodes:
            raise ValueError(
                f"Vocab size ({len(vocab)}) must match graph n_nodes ({graph.n_nodes})"
            )

        self.graph = graph
        self.vocab = vocab

    # =========================================================================
    # Alignment & Subsetting (Core Features)
    # =========================================================================

    def subgraph(self, nodes: list[str] | list[int] | np.ndarray) -> GeneGraph:
        """Extract a subgraph containing only the specified nodes.

        Edges are preserved only if both source and target are in `nodes`.
        Nodes in the new graph are re-indexed to match the order of `nodes`.

        Args:
            nodes: List of gene names (str) or indices (int) to keep.

        Returns:
            GeneGraph: A new graph instance with the subset of nodes.
        """
        if len(nodes) == 0:
            return GeneGraph(WeightedGraph([], 0), self.vocab.__class__([]))

        # 1. Resolve indices and names
        if isinstance(nodes[0], str):
            # Query by name
            indices = self.vocab.lookup_indices(nodes, warn_unknown=True)

            # Filter valid (remove -1/unknown)
            valid_pairs = [(i, name) for i, name in zip(indices, nodes) if i != -1]
            indices = [p[0] for p in valid_pairs]
            node_names = [p[1] for p in valid_pairs]
        else:
            # Query by index
            indices = list(nodes)
            node_names = self.vocab.lookup_tokens(indices)

        # 2. Create mapping: old_graph_index -> new_graph_index
        # new_graph_index corresponds to the position in the 'nodes' list
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}

        # 3. Filter and renumber edges
        # Optimization: Use set for O(1) membership check
        valid_indices_set = set(indices)
        current_edges = self.graph.edges
        new_edges = []

        for u, v, w in current_edges:
            u_int, v_int = int(u), int(v)
            if u_int in valid_indices_set and v_int in valid_indices_set:
                new_edges.append((old_to_new[u_int], old_to_new[v_int], w))

        # 4. Construct new objects
        new_graph = WeightedGraph(new_edges, n_nodes=len(indices))

        # Create new vocab (using same class as original)
        new_vocab = self.vocab.__class__(node_names)

        return GeneGraph(new_graph, new_vocab)

    def align_to_vocab(self, target_vocab: Vocab | list[str]) -> GeneGraph:
        """Align graph to match a target vocabulary structure exactly.

        This is critical for machine learning models (like ScGPT/GEARS).
        It reorders the adjacency matrix to match the feature matrix of the target data.

        - **Intersection**: Common genes are preserved and reordered.
        - **Missing in Graph**: Target genes not in graph become isolated nodes (degree 0).
        - **Extra in Graph**: Genes in graph but not in target are removed.

        Args:
            target_vocab: The target vocabulary (e.g., from adata.var_names).

        Returns:
            GeneGraph: A new graph perfectly aligned to `target_vocab`.
        """
        if not isinstance(target_vocab, Vocab):
            # Use the class of the current vocab to create the new one
            target_vocab = self.vocab.__class__(list(target_vocab))

        target_genes = target_vocab.itos

        # 1. Map target genes to current graph indices
        # Returns index if found, -1 if not found
        current_indices = self.vocab.lookup_indices(target_genes, fallback=-1)

        # 2. Build Renumbering Map
        # We need to map: current_graph_index -> target_vocab_index
        # Only for the genes that actually exist in the current graph
        old_to_new_map = {}
        for new_idx, old_idx in enumerate(current_indices):
            if old_idx != -1:
                old_to_new_map[old_idx] = new_idx

        # 3. Reconstruct Edge List
        new_edges = []
        current_edges = self.graph.edges

        for u, v, w in current_edges:
            u_int, v_int = int(u), int(v)
            # Only keep edge if BOTH ends exist in the target vocab
            if u_int in old_to_new_map and v_int in old_to_new_map:
                new_edges.append((old_to_new_map[u_int], old_to_new_map[v_int], w))

        # 4. Create Result
        # n_nodes is strictly len(target_vocab) to match dimensions
        new_graph = WeightedGraph(new_edges, n_nodes=len(target_vocab))

        return GeneGraph(new_graph, target_vocab)

    def intersection(self, other_vocab: Vocab | list[str]) -> GeneGraph:
        """Return a subgraph containing only genes present in both vocabs.

        Unlike `align_to_vocab`, this strictly shrinks the graph.
        """
        if isinstance(other_vocab, Vocab):
            other_genes = set(other_vocab.tokens)
        else:
            other_genes = set(other_vocab)

        common_genes = [g for g in self.vocab.itos if g in other_genes]
        return self.subgraph(common_genes)

    # =========================================================================
    # Proxy Methods (Delegates to WeightedGraph)
    # =========================================================================

    @property
    def n_nodes(self) -> int:
        return self.graph.n_nodes

    @property
    def n_edges(self) -> int:
        return self.graph.n_edges

    @property
    def genes(self) -> list[str]:
        return self.vocab.itos

    def neighbors(self, node: int | str) -> np.ndarray | list[str]:
        """Get neighbors. Returns names if input is name, indices if input is index."""
        if isinstance(node, str):
            node_idx = self.vocab.stoi.get(node)
            if node_idx is None:
                raise ValueError(f"Gene '{node}' not found in graph vocabulary.")
            neighbor_indices = self.graph.neighbors(node_idx)
            return [self.vocab.itos[i] for i in neighbor_indices]
        else:
            return self.graph.neighbors(node)

    def get_weights(self, node: int | str) -> np.ndarray:
        if isinstance(node, str):
            node = self.vocab.stoi.get(node)
            if node is None:
                return np.array([])
        return self.graph.get_weights(node)

    def degree(self, node: int | str) -> int:
        if isinstance(node, str):
            node = self.vocab.stoi.get(node)
            if node is None:
                return 0
        return self.graph.degree(node)

    def to_adjacency_matrix(self, sparse_fmt: bool = False) -> np.ndarray | sparse.csr_matrix:
        """Convert to adjacency matrix."""
        return self.graph.to_adjacency_matrix(sparse_fmt=sparse_fmt)

    # =========================================================================
    # I/O
    # =========================================================================

    def save(self, path: str):
        """Save graph and vocabulary to disk.

        Creates:
            - {path}_edges.npy
            - {path}_vocab.json
        """
        np.save(f"{path}_edges.npy", self.graph.edges)
        self.vocab.save(f"{path}_vocab.json")

    @classmethod
    def load(cls, path: str, vocab_cls=None) -> GeneGraph:
        """Load graph from disk."""
        if vocab_cls is None:
            # Defaults to standard Vocab if not provided
            # This allows loading custom vocab subclasses if needed
            vocab_cls = Vocab

        edges = np.load(f"{path}_edges.npy")
        vocab = vocab_cls.load(f"{path}_vocab.json")

        graph = WeightedGraph(edges, n_nodes=len(vocab))
        return cls(graph, vocab)

    def __repr__(self) -> str:
        return f"GeneGraph(n_genes={self.n_nodes}, " f"n_edges={self.graph.n_unique_edges})"
