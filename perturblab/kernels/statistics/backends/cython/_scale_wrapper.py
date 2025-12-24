"""
Python wrapper for Cython scale functions.

Copyright (c) 2024 PerturbLab
"""

import numpy as np
import scipy.sparse

try:
    from ._scale import (
        dense_standardize_cy,
        sparse_standardize_csc_cy,
        sparse_standardize_csr_cy,
    )

    _has_cython = True
except ImportError:
    _has_cython = False


def has_cython_backend() -> bool:
    """Check if Cython backend is available."""
    return _has_cython


def sparse_standardize_cython(
    X: scipy.sparse.spmatrix,
    means: np.ndarray,
    stds: np.ndarray,
    zero_center: bool = True,
    max_value: float = 0.0,
) -> scipy.sparse.spmatrix:
    """Standardize sparse matrix by columns (Cython backend).

    Args:
        X: Sparse matrix (CSC or CSR format)
        means: Column means
        stds: Column standard deviations
        zero_center: If True, subtract mean
        max_value: Maximum absolute value for clipping (0 = no clipping)

    Returns:
        Standardized matrix (same object, modified in-place)
    """
    if not _has_cython:
        raise RuntimeError("Cython backend not available")

    X.data = X.data.astype(np.float64, copy=False)
    means = np.ascontiguousarray(means, dtype=np.float64)
    stds = np.ascontiguousarray(stds, dtype=np.float64)

    n_rows, n_cols = X.shape

    if scipy.sparse.isspmatrix_csc(X):
        sparse_standardize_csc_cy(
            X.data, X.indices, X.indptr, n_rows, n_cols, means, stds, zero_center, max_value
        )
    elif scipy.sparse.isspmatrix_csr(X):
        sparse_standardize_csr_cy(
            X.data, X.indices, X.indptr, n_rows, n_cols, means, stds, zero_center, max_value
        )
    else:
        raise ValueError("X must be CSC or CSR format")

    return X


def dense_standardize_cython(
    X: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    zero_center: bool = True,
    max_value: float = 0.0,
) -> np.ndarray:
    """Standardize dense matrix by columns (Cython backend).

    Args:
        X: Dense matrix, shape (n_obs, n_vars)
        means: Column means
        stds: Column standard deviations
        zero_center: If True, subtract mean
        max_value: Maximum absolute value for clipping (0 = no clipping)

    Returns:
        Standardized matrix (same object, modified in-place)
    """
    if not _has_cython:
        raise RuntimeError("Cython backend not available")

    if not X.flags.c_contiguous:
        X = np.ascontiguousarray(X)

    X = X.astype(np.float64, copy=False)
    means = np.ascontiguousarray(means, dtype=np.float64)
    stds = np.ascontiguousarray(stds, dtype=np.float64)

    n_rows, n_cols = X.shape

    dense_standardize_cy(X, n_rows, n_cols, means, stds, zero_center, max_value)

    return X
