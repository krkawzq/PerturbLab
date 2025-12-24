"""Efficient mapping and graph query kernels for bioinformatics.

This package provides high-performance operations for common mapping tasks:
- Dictionary lookups (str ↔ int conversions)
- Bipartite graph queries (e.g., gene-GO term mappings)

All functions automatically select the best available backend (Cython > Python).

Available Functions:
    - lookup_indices: Convert strings to integers using a dictionary
    - lookup_tokens: Convert integers to strings using a vocabulary
    - bipartite_graph_query: Query neighbors in a bipartite graph

Examples:
    >>> from perturblab.kernels.mapping import lookup_indices, bipartite_graph_query
    >>> import scipy.sparse as sp
    >>>
    >>> # Dictionary lookup
    >>> gene_to_id = {"BRCA1": 0, "TP53": 1, "EGFR": 2}
    >>> indices = lookup_indices(gene_to_id, ["TP53", "BRCA1"])
    >>> print(indices)  # [1, 0]
    >>>
    >>> # Bipartite graph query
    >>> graph = sp.csr_matrix([[0, 1, 1], [1, 0, 0], [0, 1, 1]])
    >>> neighbors = bipartite_graph_query(graph, [0, 2])
    >>> print([list(n) for n in neighbors])  # [[1, 2], [1, 2]]
"""

from .ops._bipartite import bipartite_graph_query
from .ops._lookup import lookup_indices, lookup_tokens

__all__ = [
    "lookup_indices",
    "lookup_tokens",
    "bipartite_graph_query",
]
