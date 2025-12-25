"""
Normalization operations with automatic backend selection.

Copyright (c) 2024 PerturbLab
Portions adapted from Scanpy (https://github.com/scverse/scanpy)
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
Licensed under BSD 3-Clause License
"""

import numpy as np
import scipy.sparse

# Try to import backends in order of preference: C++ > Numba > Python
_has_cpp = False
_has_numba = False
_backend_name = "python"

try:
    from ..backends.cpp._normalization import (
        compute_median_nonzero_cpp,
        find_highly_expressed_genes_cpp,
        has_cpp_backend,
        inplace_divide_csr_rows_cpp,
        sparse_row_sum_csr_cpp,
        sparse_row_sum_csr_exclude_genes_cpp,
    )

    _has_cpp = has_cpp_backend()
    if _has_cpp:
        _backend_name = "C++"
        _sparse_row_sum_impl = sparse_row_sum_csr_cpp
        _inplace_divide_impl = inplace_divide_csr_rows_cpp
        _compute_median_impl = compute_median_nonzero_cpp
        _find_highly_expressed_impl = find_highly_expressed_genes_cpp
        _row_sum_exclude_impl = sparse_row_sum_csr_exclude_genes_cpp
except (ImportError, RuntimeError):
    pass

if not _has_cpp:
    try:
        from ..backends.python.numba._normalization import (
            compute_median_nonzero_numba,
            find_highly_expressed_genes_numba,
            inplace_divide_csr_rows_numba,
            sparse_row_sum_csr_exclude_genes_numba,
            sparse_row_sum_csr_numba,
        )

        _has_numba = True
        _backend_name = "Numba"

        def _sparse_row_sum_impl(X, n_threads=0):
            return sparse_row_sum_csr_numba(X.data, X.indptr, X.shape[0])

        def _inplace_divide_impl(X, divisors, allow_zero_divisor=False, n_threads=0):
            inplace_divide_csr_rows_numba(
                X.data, X.indptr, X.shape[0], divisors, allow_zero_divisor
            )
            return X

        def _compute_median_impl(values):
            return compute_median_nonzero_numba(values)

        def _find_highly_expressed_impl(X, row_sums, max_fraction, n_threads=0):
            return find_highly_expressed_genes_numba(
                X.data, X.indptr, X.indices, X.shape[0], X.shape[1], row_sums, max_fraction
            )

        def _row_sum_exclude_impl(X, gene_mask, n_threads=0):
            return sparse_row_sum_csr_exclude_genes_numba(
                X.data, X.indptr, X.indices, X.shape[0], gene_mask
            )

    except ImportError:
        pass

if not _has_cpp and not _has_numba:
    # Pure Python fallback
    def _sparse_row_sum_impl(X, n_threads=0):
        return np.array(X.sum(axis=1)).ravel()

    def _inplace_divide_impl(X, divisors, allow_zero_divisor=False, n_threads=0):
        divisors = divisors.copy()
        if not allow_zero_divisor:
            divisors[divisors == 0] = 1.0  # Avoid division by zero
        X.data /= np.repeat(divisors, np.diff(X.indptr))
        return X

    def _compute_median_impl(values):
        nonzero = values[values > 0]
        return float(np.median(nonzero)) if len(nonzero) > 0 else 0.0

    def _find_highly_expressed_impl(X, row_sums, max_fraction, n_threads=0):
        thresholds = row_sums * max_fraction
        highly_expressed = np.zeros(X.shape[1], dtype=bool)
        for i in range(X.shape[0]):
            row = X.getrow(i)
            cols = row.indices[row.data > thresholds[i]]
            highly_expressed[cols] = True
        return highly_expressed

    def _row_sum_exclude_impl(X, gene_mask, n_threads=0):
        X_filtered = X.copy()
        X_filtered[:, gene_mask] = 0
        return np.array(X_filtered.sum(axis=1)).ravel()


# ================================================================
# Public API
# ================================================================


def sparse_row_sum(
    X: scipy.sparse.csr_matrix,
    n_threads: int = 0,
) -> np.ndarray:
    f"""Compute row sums for sparse matrix.
    
    Args:
        X: Sparse matrix (CSR format), shape (n_obs, n_vars)
        n_threads: Number of threads (0 = auto, only used by C++ backend)
    
    Returns:
        Row sums, shape (n_obs,)
    
    Backend:
        Auto-selected at import time: {_backend_name}
    """

    if not scipy.sparse.isspmatrix_csr(X):
        X = X.tocsr()

    return _sparse_row_sum_impl(X, n_threads)


def inplace_divide_rows(
    X: scipy.sparse.csr_matrix,
    divisors: np.ndarray,
    allow_zero_divisor: bool = False,
    n_threads: int = 0,
) -> scipy.sparse.csr_matrix:
    f"""Divide each row by a scalar in-place.
    
    Args:
        X: Sparse matrix (CSR format), shape (n_obs, n_vars)
        divisors: Divisors for each row, shape (n_obs,)
        allow_zero_divisor: If False, set row to zero when divisor is zero
        n_threads: Number of threads (0 = auto, only used by C++ backend)
    
    Returns:
        Modified matrix (same object as X)
    
    Backend:
        Auto-selected at import time: {_backend_name}
    """

    if not scipy.sparse.isspmatrix_csr(X):
        raise ValueError("X must be CSR format for in-place operations")

    return _inplace_divide_impl(X, divisors, allow_zero_divisor, n_threads)


def compute_median_nonzero(values: np.ndarray) -> float:
    f"""Compute median of non-zero values.
    
    Args:
        values: Array of values
    
    Returns:
        Median of non-zero values
    
    Backend:
        Auto-selected at import time: {_backend_name}
    """

    return _compute_median_impl(values)


def find_highly_expressed_genes(
    X: scipy.sparse.csr_matrix,
    row_sums: np.ndarray,
    max_fraction: float = 0.05,
    n_threads: int = 0,
) -> np.ndarray:
    f"""Find highly expressed genes.
    
    A gene is highly expressed if it exceeds max_fraction of total counts
    in at least one cell.
    
    Args:
        X: Sparse matrix (CSR format), shape (n_obs, n_vars)
        row_sums: Total counts per cell, shape (n_obs,)
        max_fraction: Threshold fraction (e.g., 0.05)
        n_threads: Number of threads (0 = auto, only used by C++ backend)
    
    Returns:
        Boolean mask, shape (n_vars,). True = highly expressed.
    
    Backend:
        Auto-selected at import time: {_backend_name}
    """

    if not scipy.sparse.isspmatrix_csr(X):
        X = X.tocsr()

    return _find_highly_expressed_impl(X, row_sums, max_fraction, n_threads)


def sparse_row_sum_exclude_genes(
    X: scipy.sparse.csr_matrix,
    gene_mask: np.ndarray,
    n_threads: int = 0,
) -> np.ndarray:
    f"""Compute row sums excluding specific genes.
    
    Args:
        X: Sparse matrix (CSR format), shape (n_obs, n_vars)
        gene_mask: Boolean mask, shape (n_vars,). True = exclude this gene.
        n_threads: Number of threads (0 = auto, only used by C++ backend)
    
    Returns:
        Row sums, shape (n_obs,)
    
    Backend:
        Auto-selected at import time: {_backend_name}
    """

    if not scipy.sparse.isspmatrix_csr(X):
        X = X.tocsr()

    return _row_sum_exclude_impl(X, gene_mask, n_threads)
