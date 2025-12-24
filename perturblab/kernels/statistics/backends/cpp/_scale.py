"""
C++ backend for scaling/standardization operations.

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

    # sparse_standardize_csc
    _lib.sparse_standardize_csc_capi.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # data
        ctypes.POINTER(ctypes.c_int64),  # row_indices
        ctypes.POINTER(ctypes.c_int64),  # col_ptr
        ctypes.c_size_t,  # n_rows
        ctypes.c_size_t,  # n_cols
        ctypes.POINTER(ctypes.c_double),  # means
        ctypes.POINTER(ctypes.c_double),  # stds
        ctypes.c_bool,  # zero_center
        ctypes.c_double,  # max_value
        ctypes.c_int,  # n_threads
    ]
    _lib.sparse_standardize_csc_capi.restype = None

    # sparse_standardize_csr
    _lib.sparse_standardize_csr_capi.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # data
        ctypes.POINTER(ctypes.c_int64),  # col_indices
        ctypes.POINTER(ctypes.c_int64),  # row_ptr
        ctypes.c_size_t,  # n_rows
        ctypes.c_size_t,  # n_cols
        ctypes.POINTER(ctypes.c_double),  # means
        ctypes.POINTER(ctypes.c_double),  # stds
        ctypes.c_bool,  # zero_center
        ctypes.c_double,  # max_value
        ctypes.c_int,  # n_threads
    ]
    _lib.sparse_standardize_csr_capi.restype = None

    # dense_standardize
    _lib.dense_standardize_capi.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # data
        ctypes.c_size_t,  # n_rows
        ctypes.c_size_t,  # n_cols
        ctypes.POINTER(ctypes.c_double),  # means
        ctypes.POINTER(ctypes.c_double),  # stds
        ctypes.c_bool,  # zero_center
        ctypes.c_double,  # max_value
        ctypes.c_int,  # n_threads
    ]
    _lib.dense_standardize_capi.restype = None

except (OSError, AttributeError) as e:
    _lib = None


def has_cpp_backend() -> bool:
    """Check if C++ backend is available."""
    return _lib is not None


def sparse_standardize_cpp(
    X: scipy.sparse.spmatrix,
    means: np.ndarray,
    stds: np.ndarray,
    zero_center: bool = True,
    max_value: float = 0.0,
    n_threads: int = 0,
) -> scipy.sparse.spmatrix:
    """Standardize sparse matrix by columns (C++ backend).

    Performs in-place standardization: X[:, j] = (X[:, j] - mean[j]) / std[j]

    Args:
        X: Sparse matrix (CSC or CSR format), shape (n_obs, n_vars)
        means: Column means, shape (n_vars,)
        stds: Column standard deviations, shape (n_vars,)
        zero_center: If True, subtract mean; if False, only divide by std
        max_value: Maximum absolute value for clipping (0 = no clipping)
        n_threads: Number of threads (0 = auto)

    Returns:
        Standardized matrix (same object as X, modified in-place)
    """
    if _lib is None:
        raise RuntimeError("C++ backend not available")

    means = np.ascontiguousarray(means, dtype=np.float64)
    stds = np.ascontiguousarray(stds, dtype=np.float64)
    n_rows, n_cols = X.shape

    if scipy.sparse.isspmatrix_csc(X):
        X.data = X.data.astype(np.float64, copy=False)

        _lib.sparse_standardize_csc_capi(
            X.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            X.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            X.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            n_rows,
            n_cols,
            means.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            stds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            zero_center,
            max_value,
            n_threads,
        )
    elif scipy.sparse.isspmatrix_csr(X):
        X.data = X.data.astype(np.float64, copy=False)

        _lib.sparse_standardize_csr_capi(
            X.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            X.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            X.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            n_rows,
            n_cols,
            means.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            stds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            zero_center,
            max_value,
            n_threads,
        )
    else:
        raise ValueError("X must be CSC or CSR format")

    return X


def dense_standardize_cpp(
    X: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    zero_center: bool = True,
    max_value: float = 0.0,
    n_threads: int = 0,
) -> np.ndarray:
    """Standardize dense matrix by columns (C++ backend).

    Performs in-place standardization: X[:, j] = (X[:, j] - mean[j]) / std[j]

    Args:
        X: Dense matrix, shape (n_obs, n_vars), C-contiguous
        means: Column means, shape (n_vars,)
        stds: Column standard deviations, shape (n_vars,)
        zero_center: If True, subtract mean; if False, only divide by std
        max_value: Maximum absolute value for clipping (0 = no clipping)
        n_threads: Number of threads (0 = auto)

    Returns:
        Standardized matrix (same object as X, modified in-place)
    """
    if _lib is None:
        raise RuntimeError("C++ backend not available")

    if not X.flags.c_contiguous:
        X = np.ascontiguousarray(X)

    X = X.astype(np.float64, copy=False)
    means = np.ascontiguousarray(means, dtype=np.float64)
    stds = np.ascontiguousarray(stds, dtype=np.float64)

    n_rows, n_cols = X.shape

    _lib.dense_standardize_capi(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_rows,
        n_cols,
        means.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        stds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zero_center,
        max_value,
        n_threads,
    )

    return X
