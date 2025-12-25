"""Type stubs for Cython lookup module."""

from collections.abc import Iterable

import numpy as np

def lookup_indices(
    mapping: dict[str, int],
    queries: Iterable[str],
    fallback_value: int = -1,
) -> np.ndarray: ...
def lookup_tokens(
    indices: list | np.ndarray,
    vocabulary: list[str],
    fallback_value: str = "<unk>",
) -> list[str]: ...
