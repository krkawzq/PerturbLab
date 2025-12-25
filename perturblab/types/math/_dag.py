"""Directed Acyclic Graph (DAG) data structure.

Provides a general-purpose DAG implementation for representing hierarchical
relationships, supporting efficient queries of ancestors, descendants, and
topological properties.
"""

from collections import defaultdict, deque
from typing import Any

from perturblab.utils import get_logger

logger = get_logger()


class DAG:
    """Directed Acyclic Graph (DAG) for hierarchical relationships.

    A mathematical DAG structure that stores nodes and directed edges,
    with support for computing topological properties like level and depth.

    Attributes:
        nodes: Set of all node identifiers
        edges: List of directed edges as (parent, child) tuples
        n_nodes: Number of nodes
        n_edges: Number of edges

    Example:
        >>> # Create a simple DAG
        >>> edges = [
        ...     ('A', 'B'),  # A -> B
        ...     ('A', 'C'),  # A -> C
        ...     ('B', 'D'),  # B -> D
        ...     ('C', 'D'),  # C -> D
        ... ]
        >>> dag = DAG(edges)
        >>>
        >>> # Query relationships
        >>> parents = dag.get_parents('D')  # Returns {'B', 'C'}
        >>> children = dag.get_children('A')  # Returns {'B', 'C'}
        >>> ancestors = dag.get_all_ancestors('D')  # Returns {'A', 'B', 'C'}
        >>>
        >>> # Get topological properties
        >>> level = dag.get_level('D')  # Shortest path from root
        >>> depth = dag.get_depth('D')  # Longest path from root
    """

    def __init__(
        self,
        edges: list[tuple[Any, Any]],
        validate: bool = True,
    ):
        """Initialize DAG from edge list.

        Args:
            edges: List of directed edges as (parent, child) tuples.
                   Parent -> Child represents the direction.
            validate: Whether to validate that the graph is acyclic.

        Raises:
            ValueError: If validation is enabled and a cycle is detected.

        Example:
            >>> edges = [('root', 'A'), ('root', 'B'), ('A', 'C')]
            >>> dag = DAG(edges)
        """
        self.edges = edges

        # Extract all nodes
        self.nodes: set[Any] = set()
        for parent, child in edges:
            self.nodes.add(parent)
            self.nodes.add(child)

        # Build adjacency lists
        self._parents: dict[Any, set[Any]] = defaultdict(set)
        self._children: dict[Any, set[Any]] = defaultdict(set)

        for parent, child in edges:
            self._parents[child].add(parent)
            self._children[parent].add(child)

        # Validate acyclicity
        if validate:
            if not self._is_acyclic():
                raise ValueError("Graph contains cycles and is not a valid DAG")

        # Compute topological properties
        self._level: dict[Any, int] = {}
        self._depth: dict[Any, int] = {}
        self._compute_topological_properties()

        logger.debug(f"DAG initialized: {len(self.nodes)} nodes, {len(edges)} edges")

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the DAG."""
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        """Number of edges in the DAG."""
        return len(self.edges)

    def _is_acyclic(self) -> bool:
        """Check if the graph is acyclic using DFS.

        Returns:
            bool: True if acyclic, False if contains cycles.
        """
        # Track visited states: 0 = unvisited, 1 = visiting, 2 = visited
        state = {node: 0 for node in self.nodes}

        def has_cycle(node: Any) -> bool:
            if state[node] == 1:  # Currently visiting (back edge found)
                return True
            if state[node] == 2:  # Already visited
                return False

            state[node] = 1  # Mark as visiting

            # Check all children
            for child in self._children.get(node, set()):
                if has_cycle(child):
                    return True

            state[node] = 2  # Mark as visited
            return False

        # Check for cycles starting from each unvisited node
        for node in self.nodes:
            if state[node] == 0:
                if has_cycle(node):
                    return False

        return True

    def _compute_topological_properties(self):
        """Compute level and depth for all nodes.

        Level: Shortest distance from root (min path length)
        Depth: Longest distance from root (max path length)
        """
        # Find root nodes (nodes with no parents)
        roots = {node for node in self.nodes if not self._parents.get(node)}

        if not roots:
            logger.warning("No root nodes found in DAG")
            return

        # Initialize roots
        for root in roots:
            self._level[root] = 0
            self._depth[root] = 0

        # Compute level (BFS from roots)
        queue = deque(roots)
        visited = set(roots)

        while queue:
            node = queue.popleft()
            node_level = self._level[node]

            for child in self._children.get(node, set()):
                # Update level (minimum distance)
                if child not in self._level:
                    self._level[child] = node_level + 1
                else:
                    self._level[child] = min(self._level[child], node_level + 1)

                # Add to queue if not visited
                if child not in visited:
                    queue.append(child)
                    visited.add(child)

        # Compute depth (DFS with memoization)
        def compute_depth(node: Any) -> int:
            if node in self._depth:
                return self._depth[node]

            parents = self._parents.get(node, set())
            if not parents:
                self._depth[node] = 0
            else:
                self._depth[node] = max(compute_depth(p) for p in parents) + 1

            return self._depth[node]

        for node in self.nodes:
            if node not in self._depth:
                compute_depth(node)

    def get_parents(self, node: Any) -> set[Any]:
        """Get direct parents of a node.

        Args:
            node: Node identifier.

        Returns:
            Set[Any]: Set of parent nodes. Empty if node is a root.

        Example:
            >>> parents = dag.get_parents('D')
        """
        return self._parents.get(node, set()).copy()

    def get_children(self, node: Any) -> set[Any]:
        """Get direct children of a node.

        Args:
            node: Node identifier.

        Returns:
            Set[Any]: Set of child nodes. Empty if node is a leaf.

        Example:
            >>> children = dag.get_children('A')
        """
        return self._children.get(node, set()).copy()

    def get_all_ancestors(self, node: Any) -> set[Any]:
        """Get all ancestors of a node (transitive closure of parents).

        Args:
            node: Node identifier.

        Returns:
            Set[Any]: Set of all ancestor nodes.

        Example:
            >>> ancestors = dag.get_all_ancestors('D')
            >>> # Returns all nodes reachable by following parent edges
        """
        ancestors = set()
        queue = deque([node])
        visited = {node}

        while queue:
            current = queue.popleft()
            parents = self._parents.get(current, set())

            for parent in parents:
                if parent not in visited:
                    ancestors.add(parent)
                    visited.add(parent)
                    queue.append(parent)

        return ancestors

    def get_all_descendants(self, node: Any) -> set[Any]:
        """Get all descendants of a node (transitive closure of children).

        Args:
            node: Node identifier.

        Returns:
            Set[Any]: Set of all descendant nodes.

        Example:
            >>> descendants = dag.get_all_descendants('A')
            >>> # Returns all nodes reachable by following child edges
        """
        descendants = set()
        queue = deque([node])
        visited = {node}

        while queue:
            current = queue.popleft()
            children = self._children.get(current, set())

            for child in children:
                if child not in visited:
                    descendants.add(child)
                    visited.add(child)
                    queue.append(child)

        return descendants

    def has_path(self, source: Any, target: Any) -> bool:
        """Check if there is a path from source to target.

        Args:
            source: Source node.
            target: Target node.

        Returns:
            bool: True if path exists, False otherwise.

        Example:
            >>> has_path = dag.has_path('A', 'D')  # True if A -> ... -> D
        """
        return target in self.get_all_descendants(source)

    def get_level(self, node: Any) -> int | None:
        """Get the level of a node (shortest distance from root).

        Args:
            node: Node identifier.

        Returns:
            int | None: Level of the node, or None if node not in DAG.

        Example:
            >>> level = dag.get_level('D')
            >>> # Returns minimum number of edges from any root to D
        """
        return self._level.get(node)

    def get_depth(self, node: Any) -> int | None:
        """Get the depth of a node (longest distance from root).

        Args:
            node: Node identifier.

        Returns:
            int | None: Depth of the node, or None if node not in DAG.

        Example:
            >>> depth = dag.get_depth('D')
            >>> # Returns maximum number of edges from any root to D
        """
        return self._depth.get(node)

    def get_roots(self) -> set[Any]:
        """Get all root nodes (nodes with no parents).

        Returns:
            Set[Any]: Set of root nodes.

        Example:
            >>> roots = dag.get_roots()
        """
        return {node for node in self.nodes if not self._parents.get(node)}

    def get_leaves(self) -> set[Any]:
        """Get all leaf nodes (nodes with no children).

        Returns:
            Set[Any]: Set of leaf nodes.

        Example:
            >>> leaves = dag.get_leaves()
        """
        return {node for node in self.nodes if not self._children.get(node)}

    def get_nodes_at_level(self, level: int) -> set[Any]:
        """Get all nodes at a specific level.

        Args:
            level: Level in the hierarchy (0 = root).

        Returns:
            Set[Any]: Set of nodes at the specified level.

        Example:
            >>> nodes = dag.get_nodes_at_level(2)
        """
        return {node for node, node_level in self._level.items() if node_level == level}

    def get_topological_sort(self) -> list[Any]:
        """Get a topological sorting of the DAG.

        Returns:
            List[Any]: List of nodes in topological order.

        Example:
            >>> sorted_nodes = dag.get_topological_sort()
            >>> # Parents always appear before children
        """
        # Count in-degrees
        in_degree = {node: len(self._parents.get(node, set())) for node in self.nodes}

        # Start with nodes having no parents
        queue = deque([node for node in self.nodes if in_degree[node] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            # Reduce in-degree for children
            for child in self._children.get(node, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return result

    def subgraph(self, nodes: set[Any]) -> "DAG":
        """Create a subgraph containing only specified nodes.

        Args:
            nodes: Set of nodes to include in subgraph.

        Returns:
            DAG: New DAG containing only the specified nodes and their edges.

        Example:
            >>> subdag = dag.subgraph({'A', 'B', 'D'})
        """
        # Filter edges to only include those between specified nodes
        sub_edges = [
            (parent, child) for parent, child in self.edges if parent in nodes and child in nodes
        ]

        return DAG(sub_edges, validate=False)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the DAG.

        Returns:
            Dict: Statistics dictionary.

        Example:
            >>> stats = dag.get_stats()
            >>> print(stats['n_nodes'])
        """
        levels = [level for level in self._level.values()]
        depths = [depth for depth in self._depth.values()]

        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_roots": len(self.get_roots()),
            "n_leaves": len(self.get_leaves()),
            "max_level": max(levels) if levels else 0,
            "max_depth": max(depths) if depths else 0,
            "avg_children": self.n_edges / self.n_nodes if self.n_nodes > 0 else 0,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"DAG(nodes={self.n_nodes}, edges={self.n_edges})"

    def __contains__(self, node: Any) -> bool:
        """Check if node is in DAG."""
        return node in self.nodes
