"""Numba JIT implementation of statistics operators (optional dependency).

This module provides Numba-accelerated implementations as an optional fallback.
Numba is lazily imported to avoid making it a required dependency.

Some functions adapted from Scanpy:
https://github.com/scverse/scanpy
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
Licensed under BSD-3-Clause
"""

from ._hvg import (
    group_mean_var_numba,
    group_var_numba,
    has_numba_backend,
    sparse_clipped_moments_numba,
    sparse_mean_var_numba,
)

__all__ = [
    "has_numba_backend",
    "sparse_clipped_moments_numba",
    "sparse_mean_var_numba",
    "group_var_numba",
    "group_mean_var_numba",
]
