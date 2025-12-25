"""
C++ backend for normalization operations.

Copyright (c) 2024 PerturbLab
Portions adapted from Scanpy (https://github.com/scverse/scanpy)
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
Licensed under BSD 3-Clause License
"""

import ctypes
from pathlib import Path

import numpy as np
import scipy.sparse

# Load C++ library
_lib = None
_lib_path = Path(__file__).parent.parent / "cpp" / "libmwu_kernel.so"

try:
    _lib = ctypes.CDLL(str(_lib_path))

    # sparse_row_sum_csr
    _lib.sparse_row_sum_csr_capi.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # data
        ctypes.POINTER(ctypes.c_int64),  # indptr
        ctypes.c_size_t,  # n_rows
        ctypes.POINTER(ctypes.c_double),  # out_sums
        ctypes.c_int,  # n_threads
    ]
    _lib.sparse_row_sum_csr_capi.restype = None

    # inplace_divide_csr_rows
    _lib.inplace_divide_csr_rows_capi.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # data
        ctypes.POINTER(ctypes.c_int64),  # indptr
        ctypes.c_size_t,  # n_rows
        ctypes.POINTER(ctypes.c_double),  # divisors
        ctypes.c_bool,  # allow_zero_divisor
        ctypes.c_int,  # n_threads
    ]
    _lib.inplace_divide_csr_rows_capi.restype = None

    # compute_median_nonzero
    _lib.compute_median_nonzero_capi.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # values
        ctypes.c_size_t,  # n
    ]
    _lib.compute_median_nonzero_capi.restype = ctypes.c_double

    # find_highly_expressed_genes
    _lib.find_highly_expressed_genes_capi.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # data
        ctypes.POINTER(ctypes.c_int64),  # indptr
        ctypes.POINTER(ctypes.c_int64),  # indices
        ctypes.c_size_t,  # n_rows
        ctypes.c_size_t,  # n_cols
        ctypes.POINTER(ctypes.c_double),  # row_sums
        ctypes.c_double,  # max_fraction
        ctypes.POINTER(ctypes.c_bool),  # out_gene_mask
        ctypes.c_int,  # n_threads
    ]
    _lib.find_highly_expressed_genes_capi.restype = None

    # sparse_row_sum_csr_exclude_genes
    _lib.sparse_row_sum_csr_exclude_genes_capi.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # data
        ctypes.POINTER(ctypes.c_int64),  # indptr
        ctypes.POINTER(ctypes.c_int64),  # indices
        ctypes.c_size_t,  # n_rows
        ctypes.POINTER(ctypes.c_bool),  # gene_mask
        ctypes.POINTER(ctypes.c_double),  # out_sums
        ctypes.c_int,  # n_threads
    ]
    _lib.sparse_row_sum_csr_exclude_genes_capi.restype = None

except (OSError, AttributeError):
    _lib = None


def has_cpp_backend() -> bool:
    """Check if C++ backend is available."""
    return _lib is not None


def sparse_row_sum_csr_cpp(
    X: scipy.sparse.csr_matrix,
    n_threads: int = 0,
) -> np.ndarray:
    """Compute row sums for CSR sparse matrix (C++ backend).

    Args:
        X: Sparse matrix (CSR format), shape (n_obs, n_vars)
        n_threads: Number of threads (0 = auto)

    Returns:
        Row sums, shape (n_obs,)
    """
    if _lib is None:
        raise RuntimeError("C++ backend not available")

    if not scipy.sparse.isspmatrix_csr(X):
        X = X.tocsr()

    X = X.astype(np.float64, copy=False)
    n_rows, n_cols = X.shape

    row_sums = np.zeros(n_rows, dtype=np.float64)

    _lib.sparse_row_sum_csr_capi(
        X.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        n_rows,
        row_sums.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_threads,
    )

    return row_sums


def inplace_divide_csr_rows_cpp(
    X: scipy.sparse.csr_matrix,
    divisors: np.ndarray,
    allow_zero_divisor: bool = False,
    n_threads: int = 0,
) -> scipy.sparse.csr_matrix:
    """Divide each row by a scalar in-place (C++ backend).

    Args:
        X: Sparse matrix (CSR format), shape (n_obs, n_vars)
        divisors: Divisors for each row, shape (n_obs,)
        allow_zero_divisor: If False, set row to zero when divisor is zero
        n_threads: Number of threads (0 = auto)

    Returns:
        Modified matrix (same object as X)
    """
    if _lib is None:
        raise RuntimeError("C++ backend not available")

    if not scipy.sparse.isspmatrix_csr(X):
        raise ValueError("X must be CSR format")

    X.data = X.data.astype(np.float64, copy=False)
    divisors = np.ascontiguousarray(divisors, dtype=np.float64)
    n_rows = X.shape[0]

    _lib.inplace_divide_csr_rows_capi(
        X.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        n_rows,
        divisors.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        allow_zero_divisor,
        n_threads,
    )

    return X


def compute_median_nonzero_cpp(values: np.ndarray) -> float:
    """Compute median of non-zero values (C++ backend).

    Args:
        values: Array of values

    Returns:
        Median of non-zero values
    """
    if _lib is None:
        raise RuntimeError("C++ backend not available")

    values = np.ascontiguousarray(values, dtype=np.float64)

    return _lib.compute_median_nonzero_capi(
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        values.size,
    )


def find_highly_expressed_genes_cpp(
    X: scipy.sparse.csr_matrix,
    row_sums: np.ndarray,
    max_fraction: float = 0.05,
    n_threads: int = 0,
) -> np.ndarray:
    """Find highly expressed genes (C++ backend).

    A gene is highly expressed if it exceeds max_fraction of total counts
    in at least one cell.

    Args:
        X: Sparse matrix (CSR format), shape (n_obs, n_vars)
        row_sums: Total counts per cell, shape (n_obs,)
        max_fraction: Threshold fraction (e.g., 0.05)
        n_threads: Number of threads (0 = auto)

    Returns:
        Boolean mask, shape (n_vars,). True = highly expressed.
    """
    if _lib is None:
        raise RuntimeError("C++ backend not available")

    if not scipy.sparse.isspmatrix_csr(X):
        X = X.tocsr()

    X = X.astype(np.float64, copy=False)
    row_sums = np.ascontiguousarray(row_sums, dtype=np.float64)
    n_rows, n_cols = X.shape

    gene_mask = np.zeros(n_cols, dtype=bool)

    _lib.find_highly_expressed_genes_capi(
        X.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        X.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        n_rows,
        n_cols,
        row_sums.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        max_fraction,
        gene_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        n_threads,
    )

    return gene_mask


def sparse_row_sum_csr_exclude_genes_cpp(
    X: scipy.sparse.csr_matrix,
    gene_mask: np.ndarray,
    n_threads: int = 0,
) -> np.ndarray:
    """Compute row sums excluding specific genes (C++ backend).

    Args:
        X: Sparse matrix (CSR format), shape (n_obs, n_vars)
        gene_mask: Boolean mask, shape (n_vars,). True = exclude this gene.
        n_threads: Number of threads (0 = auto)

    Returns:
        Row sums, shape (n_obs,)
    """
    if _lib is None:
        raise RuntimeError("C++ backend not available")

    if not scipy.sparse.isspmatrix_csr(X):
        X = X.tocsr()

    X = X.astype(np.float64, copy=False)
    gene_mask = np.ascontiguousarray(gene_mask, dtype=bool)
    n_rows, n_cols = X.shape

    row_sums = np.zeros(n_rows, dtype=np.float64)

    _lib.sparse_row_sum_csr_exclude_genes_capi(
        X.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        X.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        n_rows,
        gene_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
        row_sums.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_threads,
    )

    return row_sums
