"""High-performance mapping kernels.

Provides optimized implementations for dictionary lookups, index mapping,
and bipartite graph queries. Automatically uses Cython-accelerated versions
when available, falling back to pure Python implementations.
"""

try:
    from ._fast_lookup import lookup_indices, lookup_tokens
except ImportError:
    from ._lookup import lookup_indices, lookup_tokens

try:
    from ._fast_bipartite import bipartite_graph_query
except ImportError:
    from ._bipartite import bipartite_graph_query

__all__ = [
    'lookup_indices',
    'lookup_tokens',
    'bipartite_graph_query',
]
