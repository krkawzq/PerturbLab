"""Type stubs for Cython-accelerated lookup operations."""

import numpy as np
from typing import Iterable


def lookup_indices(
    mapping: dict[str, int],
    queries: Iterable[str],
    fallback_value: int = -1
) -> np.ndarray:
    """Perform high-performance vectorized dictionary lookup.

    Args:
        mapping: Dictionary mapping strings to integers.
        queries: Iterable of string keys to query.
        fallback_value: Value to return when key is not found.

    Returns:
        np.ndarray: int32 array of mapped integers or fallback_value.
    """
    ...


def lookup_tokens(
    indices: list[int] | np.ndarray,
    vocabulary: list[str],
    fallback_value: str = '<unk>'
) -> list[str]:
    """Perform vectorized gather operation (int -> str).

    Args:
        indices: Integer indices to query.
        vocabulary: List of strings to gather from.
        fallback_value: String to return when index is out of bounds.

    Returns:
        list[str]: List of strings corresponding to the indices.
    """
    ...
