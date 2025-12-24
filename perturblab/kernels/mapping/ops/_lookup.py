"""Dictionary lookup operations with automatic backend selection.

This module provides high-performance dictionary lookup operations with
automatic backend selection performed at import time.

Backend priority: Cython > Python
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from perturblab.utils import get_logger

logger = get_logger()

__all__ = ["lookup_indices", "lookup_tokens"]


# =============================================================================
# Backend Selection (performed at import time)
# =============================================================================

_BACKEND_NAME = None
_lookup_indices_impl = None
_lookup_tokens_impl = None


def _select_backend():
    """Select backend at import time."""
    global _BACKEND_NAME, _lookup_indices_impl, _lookup_tokens_impl

    # Try Cython backend first
    try:
        from ..backends.cython._lookup import lookup_indices as cython_lookup_indices
        from ..backends.cython._lookup import lookup_tokens as cython_lookup_tokens

        _BACKEND_NAME = "Cython"
        _lookup_indices_impl = cython_lookup_indices
        _lookup_tokens_impl = cython_lookup_tokens
        logger.debug("Lookup backend: Cython (fast)")
        return
    except ImportError:
        pass

    # Fall back to Python
    from ..backends.python._lookup import lookup_indices as python_lookup_indices
    from ..backends.python._lookup import lookup_tokens as python_lookup_tokens

    _BACKEND_NAME = "Python"
    _lookup_indices_impl = python_lookup_indices
    _lookup_tokens_impl = python_lookup_tokens
    logger.debug("Lookup backend: Python (consider compiling Cython for better performance)")


# Select backend on module import
_select_backend()


# =============================================================================
# Public API
# =============================================================================


def lookup_indices(
    mapping: dict[str, int],
    queries: Iterable[str],
    fallback_value: int = -1,
) -> np.ndarray:
    """Perform vectorized dictionary lookup (str -> int).

    Maps string keys to integer indices using a dictionary. This is useful
    for converting gene names to IDs, converting perturbation names to indices, etc.

    Parameters
    ----------
    mapping : dict[str, int]
        Dictionary mapping strings to integers.
    queries : Iterable[str]
        Iterable of string keys to query.
    fallback_value : int, default=-1
        Value to return when key is not found.

    Returns
    -------
    np.ndarray
        int32 array of mapped integers or fallback_value.

    Examples
    --------
    >>> gene_to_id = {"BRCA1": 0, "TP53": 1, "EGFR": 2}
    >>> query_genes = ["TP53", "BRCA1", "UNKNOWN"]
    >>> indices = lookup_indices(gene_to_id, query_genes, fallback_value=-1)
    >>> print(indices)
    [1 0 -1]

    Notes
    -----
    - Time complexity: O(n) where n is the number of queries
    - Space complexity: O(n) for the output array
    - Uses Cython backend if available for ~2-3x speedup
    """
    return _lookup_indices_impl(mapping, queries, fallback_value)


def lookup_tokens(
    indices: list | np.ndarray,
    vocabulary: list[str],
    fallback_value: str = "<unk>",
) -> list[str]:
    """Perform vectorized index-to-string mapping (int -> str).

    Maps integer indices back to string tokens using a vocabulary list.
    This is the inverse operation of `lookup_indices`.

    Parameters
    ----------
    indices : list or np.ndarray
        Integer indices to query.
    vocabulary : list[str]
        List of strings to gather from.
    fallback_value : str, default='<unk>'
        String to return when index is out of bounds.

    Returns
    -------
    list[str]
        List of strings corresponding to the indices.

    Examples
    --------
    >>> vocabulary = ["BRCA1", "TP53", "EGFR"]
    >>> indices = [1, 0, 10]  # 10 is out of bounds
    >>> tokens = lookup_tokens(indices, vocabulary, fallback_value="<unk>")
    >>> print(tokens)
    ['TP53', 'BRCA1', '<unk>']

    Notes
    -----
    - Time complexity: O(n) where n is the number of indices
    - Space complexity: O(n) for the output list
    - Uses Cython backend if available for ~2-3x speedup
    """
    return _lookup_tokens_impl(indices, vocabulary, fallback_value)
