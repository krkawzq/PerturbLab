"""PerturbLab Hypergraph Core.

A general hypergraph infrastructure designed for network analysis in bioinformatics.

Features:
    1. Recursive definition (Edge can also be Node).
    2. Supports both directed and undirected edges.
    3. Fast auto-indexing for O(1) neighbor lookups.
    4. Sparse matrix export (compatible with Scanpy/Scipy).
"""

from abc import ABC, abstractmethod
from typing import (
    Any, Generic, Hashable, Set, Tuple, Type, TypeVar,
    Dict, List, Iterator, Optional, Union, Iterable
)
from collections import defaultdict
import itertools

# -----------------------------------------------------------------------------
# 1. Node (Vertex) Abstraction
# -----------------------------------------------------------------------------

class Node(ABC):
    """Base node in the graph.

    Any entity participating in a relationship should inherit this class and implement
    the `uid` property.

    Attributes:
        None
    """
    __slots__ = ()

    @property
    @abstractmethod
    def uid(self) -> Hashable:
        """Returns the unique identifier of the node.

        Returns:
            Hashable: The unique ID (e.g., gene name or perturbation ID).
        """
        pass

    def __hash__(self):
        return hash(self.uid)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.uid == other.uid

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.uid}>"

# -----------------------------------------------------------------------------
# 2. HyperEdge (Connection Unit)
# -----------------------------------------------------------------------------

class HyperEdge(Node, ABC):
    """Hyperedge: indivisible unit connecting two or more Nodes.

    Note:
        HyperEdge itself is a Node, supporting "edge of edges" (recursive graph).
    
    Attributes:
        _metadata: Metadata dictionary for storing edge properties (weight, confidence, etc.).
    """
    __slots__ = ('_metadata',)

    def __init__(self, **metadata):
        """Initializes a HyperEdge with optional metadata.

        Args:
            **metadata: Arbitrary edge attributes or annotations.
        """
        self._metadata = metadata

    @property
    @abstractmethod
    def participants(self) -> Tuple[Node, ...]:
        """Returns all nodes participating in this edge.

        Returns:
            Tuple[Node, ...]: Tuple of participating nodes.
        """
        pass

    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns metadata about the edge.

        Returns:
            Dict[str, Any]: Edge attributes (such as weight, confidence, source, etc.).
        """
        return self._metadata

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from metadata with optional default.

        Args:
            key (str): Key for a metadata attribute.
            default (Any): Value to return if key does not exist.

        Returns:
            Any: Value from metadata or the default.
        """
        return self._metadata.get(key, default)

    def uid(self) -> Hashable:
        """Default edge ID generation: (Class name, tuple of participant IDs).

        Returns:
            Hashable: Unique identifier for this edge.
        """
        part_ids = tuple(n.uid for n in self.participants)
        return (self.__class__.__name__,) + part_ids

# --- Concrete Edge Types ---

class UndirectedEdge(HyperEdge):
    """Undirected edge: (A, B) is equivalent to (B, A).

    Automatically sorts participants by UID to ensure canonical ordering.

    Attributes:
        _participants: Tuple of sorted participating nodes.
    """
    __slots__ = ('_participants',)

    def __init__(self, nodes: Iterable[Node], **metadata):
        """Initializes an undirected edge.

        Args:
            nodes (Iterable[Node]): Nodes to connect in an undirected edge.
            **metadata: Edge metadata.
        """
        super().__init__(**metadata)
        # Sort by UID string to ensure canonical ordering
        self._participants = tuple(
            sorted(nodes, key=lambda x: str(x.uid))
        )

    @property
    def participants(self) -> Tuple[Node, ...]:
        """Returns participating nodes (sorted tuple).

        Returns:
            Tuple[Node, ...]: Edge participants.
        """
        return self._participants

class DirectedEdge(HyperEdge):
    """Directed edge: Source -> Target.

    Attributes:
        _source: Source node.
        _target: Target node.
    """
    __slots__ = ('_source', '_target')

    def __init__(self, source: Node, target: Node, **metadata):
        """Initializes a directed edge.

        Args:
            source (Node): Source node.
            target (Node): Target node.
            **metadata: Edge metadata.
        """
        super().__init__(**metadata)
        self._source = source
        self._target = target

    @property
    def participants(self) -> Tuple[Node, ...]:
        """Returns the source and target nodes tuple.

        Returns:
            Tuple[Node, ...]: (source, target)
        """
        return (self._source, self._target)

    @property
    def source(self) -> Node:
        """Returns the source node.

        Returns:
            Node: Source node.
        """
        return self._source

    @property
    def target(self) -> Node:
        """Returns the target node.

        Returns:
            Node: Target node.
        """
        return self._target

# -----------------------------------------------------------------------------
# 3. Relation (The Container)
# -----------------------------------------------------------------------------

T_Edge = TypeVar("T_Edge", bound=HyperEdge)

class Relation(Generic[T_Edge]):
    """A collection of edges of a specific type.

    - Maintains the set of edges.
    - Maintains inverted incidence index (Node -> Edges).
    - Provides graph analysis interfaces.

    Example applications: "Gene co-expression relation," "TF-target regulation relation," etc.

    Attributes:
        name: Name of this relation.
        _edge_cls: The expected edge class type.
        _edges: Set of all edges.
        _incidence_index: Node UID to set of edges incidence index.
        _node_registry: Node UID to node instance.
    """
    def __init__(self, name: str, edge_cls: Type[T_Edge]):
        """Initializes a Relation container.

        Args:
            name (str): Name of the relation.
            edge_cls (Type[T_Edge]): Expected type of edges in this relation.
        """
        self.name = name
        self._edge_cls = edge_cls

        # Core storage for edges
        self._edges: Set[T_Edge] = set()

        # Node UID -> Set[Edge]: For O(1) lookups of all edges a node participates in
        self._incidence_index: Dict[Hashable, Set[T_Edge]] = defaultdict(set)

        # Node UID -> Node: Avoid redundant node objects and support fast lookup
        self._node_registry: Dict[Hashable, Node] = {}

    def add(self, edge: T_Edge):
        """Add an edge to the relation and update incidence indices.

        Args:
            edge (T_Edge): Edge to add.

        Raises:
            TypeError: If edge is not of the correct class.
        """
        if not isinstance(edge, self._edge_cls):
            raise TypeError(
                f"Relation '{self.name}' expects {self._edge_cls.__name__}, "
                f"got {type(edge).__name__}"
            )

        if edge in self._edges:
            # Idempotent: ignore if already present
            return

        self._edges.add(edge)

        # Update indices
        for node in edge.participants:
            nid = node.uid
            self._incidence_index[nid].add(edge)
            if nid not in self._node_registry:
                self._node_registry[nid] = node

    def add_batch(self, edges: Iterable[T_Edge]):
        """Add multiple edges (recommended for initialization).

        Args:
            edges (Iterable[T_Edge]): Edges to add.
        """
        for edge in edges:
            self.add(edge)

    def remove(self, edge: T_Edge):
        """Remove an edge and clean up indices.

        Args:
            edge (T_Edge): Edge to remove.
        """
        if edge not in self._edges:
            return

        self._edges.remove(edge)
        for node in edge.participants:
            nid = node.uid
            if nid in self._incidence_index:
                self._incidence_index[nid].discard(edge)
                # If the node is now isolated, optionally clean up
                if not self._incidence_index[nid]:
                    del self._incidence_index[nid]
                    del self._node_registry[nid]

    # --- Query Interfaces ---

    def get_incident_edges(self, node_uid: Hashable) -> Set[T_Edge]:
        """Get all edges incident to a given node.

        Args:
            node_uid (Hashable): Node UID.

        Returns:
            Set[T_Edge]: Set of edges involving node with this UID.
        """
        return self._incidence_index.get(node_uid, set())

    def get_neighbors(self, node_uid: Hashable) -> Set[Node]:
        """Get 1-hop neighbors of a node via all edges.

        For hyperedges, neighbors are all other nodes in the same edge.

        Args:
            node_uid (Hashable): Node UID.

        Returns:
            Set[Node]: Set of unique neighboring nodes.
        """
        edges = self.get_incident_edges(node_uid)
        neighbors = set()
        for edge in edges:
            for p in edge.participants:
                if p.uid != node_uid:
                    neighbors.add(p)
        return neighbors

    def get_node(self, uid: Hashable) -> Optional[Node]:
        """Get node object by UID.

        Args:
            uid (Hashable): Node UID.

        Returns:
            Optional[Node]: Node object or None if not present.
        """
        return self._node_registry.get(uid)

    # --- Properties ---

    @property
    def nodes(self) -> List[Node]:
        """Returns all active nodes in the graph.

        Returns:
            List[Node]: List of nodes.
        """
        return list(self._node_registry.values())

    @property
    def edges(self) -> Set[T_Edge]:
        """Returns all edges in the relation.

        Returns:
            Set[T_Edge]: Set of edges.
        """
        return self._edges

    def __len__(self):
        """Returns the number of edges in the relation.

        Returns:
            int: Number of edges.
        """
        return len(self._edges)

    def __contains__(self, edge: T_Edge):
        """Checks if an edge is in this relation.

        Args:
            edge (T_Edge): Edge to check.

        Returns:
            bool: True if present.
        """
        return edge in self._edges

    def __repr__(self):
        return (
            f"<Relation '{self.name}': "
            f"{len(self._edges)} edges, {len(self._node_registry)} nodes>"
        )

    # --- Export Interface (Bioinformatics-optimized) ---

    def to_scipy_sparse(
        self,
        weight_key: str = 'weight',
        default_weight: float = 1.0,
        node_order: Optional[List[Hashable]] = None
    ):
        """Export relation as a Scipy sparse adjacency matrix.

        For hypergraphs, performs clique expansion (i.e., connects all pairs in each edge).

        Args:
            weight_key (str): Key in edge metadata to use for weight. Defaults to 'weight'.
            default_weight (float): Default edge weight if not present. Defaults to 1.0.
            node_order (Optional[List[Hashable]]): Optional list of node UIDs to define matrix rows/cols order.
                If not provided, uses sorted order of contained nodes.

        Returns:
            Tuple[scipy.sparse.spmatrix, List[Hashable]]: CSR matrix and node IDs in corresponding order.

        Raises:
            ImportError: If scipy or numpy is not installed.
        """
        try:
            import scipy.sparse as sp
            import numpy as np
        except ImportError:
            raise ImportError("Exporting to sparse matrix requires 'scipy' and 'numpy'.")

        # Establish mapping between UIDs and matrix indices
        if node_order is None:
            active_nodes = sorted(list(self._node_registry.keys()), key=str)
        else:
            active_nodes = node_order

        node_to_idx = {uid: i for i, uid in enumerate(active_nodes)}
        n_dim = len(active_nodes)

        # Build coordinate format arrays (COO)
        rows = []
        cols = []
        data = []

        for edge in self._edges:
            weight = edge.get(weight_key, default_weight)
            participants = [p for p in edge.participants if p.uid in node_to_idx]

            # Clique expansion: connect every pair of participants
            # Warning: Large/dense hyperedges will generate many edges
            for i in range(len(participants)):
                u_idx = node_to_idx[participants[i].uid]
                for j in range(i + 1, len(participants)):
                    v_idx = node_to_idx[participants[j].uid]

                    # Fill both (u, v) and (v, u) for undirected representation
                    rows.extend([u_idx, v_idx])
                    cols.extend([v_idx, u_idx])
                    data.extend([weight, weight])

        # Create the sparse matrix
        matrix = sp.coo_matrix((data, (rows, cols)), shape=(n_dim, n_dim))
        return matrix.tocsr(), active_nodes
