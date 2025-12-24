"""C API interface for HVG (Highly Variable Genes) operators.

This module provides Python bindings to the high-performance C++ implementations
of HVG detection algorithms, including Seurat V3 method.
"""

import ctypes
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.sparse

logger = logging.getLogger(__name__)

__all__ = [
    "has_cpp_backend",
    "sparse_clipped_moments_cpp",
    "sparse_mean_var_cpp",
    "clip_matrix_cpp",
    "polynomial_fit_cpp",
    "loess_fit_cpp",
    "group_var_cpp",
    "group_mean_var_cpp",
]


# =============================================================================
# C API Structures
# =============================================================================

class ClippedMomentsResults(ctypes.Structure):
    """C structure for clipped moments results."""
    _fields_ = [
        ("sums", ctypes.POINTER(ctypes.c_double)),
        ("sum_squares", ctypes.POINTER(ctypes.c_double)),
        ("size", ctypes.c_size_t),
    ]


class MeanVarResults(ctypes.Structure):
    """C structure for mean/variance results."""
    _fields_ = [
        ("means", ctypes.POINTER(ctypes.c_double)),
        ("vars", ctypes.POINTER(ctypes.c_double)),
        ("size", ctypes.c_size_t),
    ]


class PolynomialFitResults(ctypes.Structure):
    """C structure for polynomial fit results."""
    _fields_ = [
        ("fitted", ctypes.POINTER(ctypes.c_double)),
        ("coeffs", ctypes.POINTER(ctypes.c_double)),
        ("n_points", ctypes.c_size_t),
        ("n_coeffs", ctypes.c_size_t),
    ]


# =============================================================================
# Load C++ Library
# =============================================================================

_lib: Optional[ctypes.CDLL] = None


def _find_library() -> Optional[Path]:
    """Find the compiled C++ library."""
    module_dir = Path(__file__).parent
    
    lib_names = [
        "libmwu_kernel.so",         # Linux
        "libmwu_kernel.dylib",      # macOS
        "mwu_kernel.dll",           # Windows
    ]
    
    for lib_name in lib_names:
        lib_path = module_dir / lib_name
        if lib_path.exists():
            logger.info(f"Found C++ library for HVG: {lib_path}")
            return lib_path
    
    return None


def _load_library() -> Optional[ctypes.CDLL]:
    """Load the C++ library and setup function signatures."""
    lib_path = _find_library()
    if lib_path is None:
        logger.warning("C++ library not found for HVG, will use fallback")
        return None
    
    try:
        lib = ctypes.CDLL(str(lib_path))
        
        # ============================================================
        # Sparse Clipped Moments
        # ============================================================
        lib.sparse_clipped_moments_csc_capi.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # data
            ctypes.POINTER(ctypes.c_int64),   # row_indices
            ctypes.POINTER(ctypes.c_int64),   # col_ptr
            ctypes.c_size_t,                  # n_cols
            ctypes.POINTER(ctypes.c_double),  # clip_vals
            ctypes.c_int,                     # n_threads
        ]
        lib.sparse_clipped_moments_csc_capi.restype = ctypes.POINTER(ClippedMomentsResults)
        
        lib.clipped_moments_free_results.argtypes = [ctypes.POINTER(ClippedMomentsResults)]
        lib.clipped_moments_free_results.restype = None
        
        # ============================================================
        # Sparse Mean/Variance
        # ============================================================
        lib.sparse_mean_var_csc_capi.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # data
            ctypes.POINTER(ctypes.c_int64),   # row_indices
            ctypes.POINTER(ctypes.c_int64),   # col_ptr
            ctypes.c_size_t,                  # n_rows
            ctypes.c_size_t,                  # n_cols
            ctypes.c_bool,                    # include_zeros
            ctypes.c_int,                     # n_threads
        ]
        lib.sparse_mean_var_csc_capi.restype = ctypes.POINTER(MeanVarResults)
        
        lib.sparse_mean_csc_capi.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_bool,
            ctypes.c_int,
        ]
        lib.sparse_mean_csc_capi.restype = ctypes.POINTER(ctypes.c_double)
        
        lib.sparse_var_csc_capi.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_bool,
            ctypes.c_int,
        ]
        lib.sparse_var_csc_capi.restype = ctypes.POINTER(ctypes.c_double)
        
        lib.mean_var_free_results.argtypes = [ctypes.POINTER(MeanVarResults)]
        lib.mean_var_free_results.restype = None
        
        # ============================================================
        # Clip Matrix (Dense)
        # ============================================================
        lib.clip_matrix_by_column_capi.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # data (in-place)
            ctypes.c_size_t,                  # n_rows
            ctypes.c_size_t,                  # n_cols
            ctypes.POINTER(ctypes.c_double),  # clip_vals
            ctypes.c_int,                     # n_threads
        ]
        lib.clip_matrix_by_column_capi.restype = None
        
        # ============================================================
        # Polynomial Fit
        # ============================================================
        lib.polynomial_fit_capi.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # x
            ctypes.POINTER(ctypes.c_double),  # y
            ctypes.c_size_t,                  # n
            ctypes.c_int,                     # degree
            ctypes.c_bool,                    # return_coeffs
        ]
        lib.polynomial_fit_capi.restype = ctypes.POINTER(PolynomialFitResults)
        
        lib.weighted_polynomial_fit_capi.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # x
            ctypes.POINTER(ctypes.c_double),  # y
            ctypes.POINTER(ctypes.c_double),  # weights
            ctypes.c_size_t,                  # n
            ctypes.c_int,                     # degree
            ctypes.c_bool,                    # return_coeffs
        ]
        lib.weighted_polynomial_fit_capi.restype = ctypes.POINTER(PolynomialFitResults)
        
        lib.loess_fit_fast_capi.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # x
            ctypes.POINTER(ctypes.c_double),  # y
            ctypes.c_size_t,                  # n
            ctypes.c_double,                  # span
            ctypes.c_int,                     # n_threads
        ]
        lib.loess_fit_fast_capi.restype = ctypes.POINTER(ctypes.c_double)
        
        lib.polynomial_fit_free_results.argtypes = [ctypes.POINTER(PolynomialFitResults)]
        lib.polynomial_fit_free_results.restype = None
        
        # ============================================================
        # Group Operations (Extended)
        # ============================================================
        lib.group_mean_var_csc_capi.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # data
            ctypes.POINTER(ctypes.c_int64),   # indices
            ctypes.POINTER(ctypes.c_int64),   # indptr
            ctypes.c_size_t,                  # n_rows
            ctypes.c_size_t,                  # n_cols
            ctypes.c_size_t,                  # nnz
            ctypes.POINTER(ctypes.c_int32),   # group_id
            ctypes.c_size_t,                  # n_groups
            ctypes.c_bool,                    # include_zeros
            ctypes.c_int,                     # threads
        ]
        lib.group_mean_var_csc_capi.restype = ctypes.POINTER(MeanVarResults)
        
        lib.group_var_csc_capi.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.c_bool,
            ctypes.c_int,
        ]
        lib.group_var_csc_capi.restype = ctypes.POINTER(ctypes.c_double)
        
        logger.info("C++ library loaded successfully for HVG operators")
        return lib
        
    except Exception as e:
        logger.warning(f"Failed to load C++ library: {e}")
        return None


# Initialize library
_lib = _load_library()


def has_cpp_backend() -> bool:
    """Check if C++ backend is available."""
    return _lib is not None


# =============================================================================
# Python Interface Functions
# =============================================================================

def sparse_clipped_moments_cpp(
    X: scipy.sparse.csc_matrix,
    clip_vals: np.ndarray,
    n_threads: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute clipped moments for sparse matrix (C++ backend).
    
    Args:
        X: Sparse matrix (CSC format), shape (n_obs, n_vars)
        clip_vals: Clipping threshold for each column, shape (n_vars,)
        n_threads: Number of threads (0 = auto)
    
    Returns:
        sums: Clipped sums, shape (n_vars,)
        sum_squares: Clipped sum of squares, shape (n_vars,)
    """
    if _lib is None:
        raise RuntimeError("C++ backend not available")
    
    if not scipy.sparse.isspmatrix_csc(X):
        X = X.tocsc()
    
    X = X.astype(np.float64, copy=False)
    clip_vals = np.ascontiguousarray(clip_vals, dtype=np.float64)
    
    n_rows, n_cols = X.shape
    
    # Call C++ function
    result_ptr = _lib.sparse_clipped_moments_csc_capi(
        X.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        X.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        n_cols,
        clip_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_threads,
    )
    
    if not result_ptr:
        raise RuntimeError("C++ function returned null pointer")
    
    # Extract results
    result = result_ptr.contents
    sums = np.ctypeslib.as_array(result.sums, shape=(result.size,)).copy()
    sum_squares = np.ctypeslib.as_array(result.sum_squares, shape=(result.size,)).copy()
    
    # Free C++ memory
    _lib.clipped_moments_free_results(result_ptr)
    
    return sums, sum_squares


def sparse_mean_var_cpp(
    X: scipy.sparse.csc_matrix,
    include_zeros: bool = True,
    n_threads: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and variance for sparse matrix columns (C++ backend).
    
    Args:
        X: Sparse matrix (CSC format), shape (n_obs, n_vars)
        include_zeros: Whether to include implicit zeros in calculation
        n_threads: Number of threads (0 = auto)
    
    Returns:
        means: Column means, shape (n_vars,)
        vars: Column variances, shape (n_vars,)
    """
    if _lib is None:
        raise RuntimeError("C++ backend not available")
    
    if not scipy.sparse.isspmatrix_csc(X):
        X = X.tocsc()
    
    X = X.astype(np.float64, copy=False)
    n_rows, n_cols = X.shape
    
    result_ptr = _lib.sparse_mean_var_csc_capi(
        X.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        X.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        n_rows,
        n_cols,
        include_zeros,
        n_threads,
    )
    
    if not result_ptr:
        raise RuntimeError("C++ function returned null pointer")
    
    result = result_ptr.contents
    means = np.ctypeslib.as_array(result.means, shape=(result.size,)).copy()
    vars = np.ctypeslib.as_array(result.vars, shape=(result.size,)).copy()
    
    _lib.mean_var_free_results(result_ptr)
    
    return means, vars


def clip_matrix_cpp(
    X: np.ndarray,
    clip_vals: np.ndarray,
    n_threads: int = 0,
) -> np.ndarray:
    """Clip dense matrix by column (C++ backend, in-place).
    
    Args:
        X: Dense matrix, shape (n_obs, n_vars), C-contiguous
        clip_vals: Clipping threshold for each column, shape (n_vars,)
        n_threads: Number of threads (0 = auto)
    
    Returns:
        Clipped matrix (same object as X, modified in-place)
    """
    if _lib is None:
        raise RuntimeError("C++ backend not available")
    
    if not X.flags.c_contiguous:
        X = np.ascontiguousarray(X)
    
    X = X.astype(np.float64, copy=False)
    clip_vals = np.ascontiguousarray(clip_vals, dtype=np.float64)
    
    n_rows, n_cols = X.shape
    
    _lib.clip_matrix_by_column_capi(
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_rows,
        n_cols,
        clip_vals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_threads,
    )
    
    return X


def polynomial_fit_cpp(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 2,
    weights: Optional[np.ndarray] = None,
    return_coeffs: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Fit polynomial regression (C++ backend).
    
    Args:
        x: Input x coordinates, shape (n,)
        y: Input y coordinates, shape (n,)
        degree: Polynomial degree
        weights: Optional weights, shape (n,)
        return_coeffs: Whether to return coefficients
    
    Returns:
        fitted: Fitted y values, shape (n,)
        coeffs: Coefficients (if return_coeffs=True), shape (degree+1,)
    """
    if _lib is None:
        raise RuntimeError("C++ backend not available")
    
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n = len(x)
    
    if weights is not None:
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        result_ptr = _lib.weighted_polynomial_fit_capi(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n,
            degree,
            return_coeffs,
        )
    else:
        result_ptr = _lib.polynomial_fit_capi(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n,
            degree,
            return_coeffs,
        )
    
    if not result_ptr:
        raise RuntimeError("C++ function returned null pointer")
    
    result = result_ptr.contents
    fitted = np.ctypeslib.as_array(result.fitted, shape=(result.n_points,)).copy()
    
    coeffs = None
    if return_coeffs and result.coeffs:
        coeffs = np.ctypeslib.as_array(result.coeffs, shape=(result.n_coeffs,)).copy()
    
    _lib.polynomial_fit_free_results(result_ptr)
    
    return fitted, coeffs


def loess_fit_cpp(
    x: np.ndarray,
    y: np.ndarray,
    span: float = 0.3,
    n_threads: int = 0,
) -> np.ndarray:
    """Fit LOESS (C++ backend).
    
    Args:
        x: Input x coordinates, shape (n,)
        y: Input y coordinates, shape (n,)
        span: Smoothing parameter (fraction of data)
        n_threads: Number of threads (0 = auto)
    
    Returns:
        fitted: Fitted y values, shape (n,)
    """
    if _lib is None:
        raise RuntimeError("C++ backend not available")
    
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n = len(x)
    
    result_ptr = _lib.loess_fit_fast_capi(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n,
        span,
        n_threads,
    )
    
    if not result_ptr:
        raise RuntimeError("C++ function returned null pointer")
    
    fitted = np.ctypeslib.as_array(result_ptr, shape=(n,)).copy()
    
    # Free C++ memory (just a double array)
    ctypes.pythonapi.free.argtypes = [ctypes.c_void_p]
    ctypes.pythonapi.free(result_ptr)
    
    return fitted


def group_var_cpp(
    X: scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_groups: int,
    include_zeros: bool = True,
    n_threads: int = 0,
) -> np.ndarray:
    """Compute group-wise variance (C++ backend).
    
    Args:
        X: Sparse matrix (CSC format), shape (n_obs, n_vars)
        group_id: Group labels, shape (n_obs,), dtype int32
        n_groups: Number of groups
        include_zeros: Whether to include implicit zeros
        n_threads: Number of threads (0 = auto)
    
    Returns:
        Group variances, shape (n_vars * n_groups,)
    """
    if _lib is None:
        raise RuntimeError("C++ backend not available")
    
    if not scipy.sparse.isspmatrix_csc(X):
        X = X.tocsc()
    
    X = X.astype(np.float64, copy=False)
    group_id = np.ascontiguousarray(group_id, dtype=np.int32)
    
    n_rows, n_cols = X.shape
    
    result_ptr = _lib.group_var_csc_capi(
        X.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        X.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        n_rows,
        n_cols,
        X.nnz,
        group_id.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        n_groups,
        include_zeros,
        n_threads,
    )
    
    if not result_ptr:
        raise RuntimeError("C++ function returned null pointer")
    
    result_size = n_cols * n_groups
    vars = np.ctypeslib.as_array(result_ptr, shape=(result_size,)).copy()
    
    ctypes.pythonapi.free.argtypes = [ctypes.c_void_p]
    ctypes.pythonapi.free(result_ptr)
    
    return vars


def group_mean_var_cpp(
    X: scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_groups: int,
    include_zeros: bool = True,
    n_threads: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute group-wise mean and variance (C++ backend).
    
    Args:
        X: Sparse matrix (CSC format), shape (n_obs, n_vars)
        group_id: Group labels, shape (n_obs,), dtype int32
        n_groups: Number of groups
        include_zeros: Whether to include implicit zeros
        n_threads: Number of threads (0 = auto)
    
    Returns:
        means: Group means, shape (n_vars * n_groups,)
        vars: Group variances, shape (n_vars * n_groups,)
    """
    if _lib is None:
        raise RuntimeError("C++ backend not available")
    
    if not scipy.sparse.isspmatrix_csc(X):
        X = X.tocsc()
    
    X = X.astype(np.float64, copy=False)
    group_id = np.ascontiguousarray(group_id, dtype=np.int32)
    
    n_rows, n_cols = X.shape
    
    result_ptr = _lib.group_mean_var_csc_capi(
        X.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        X.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        n_rows,
        n_cols,
        X.nnz,
        group_id.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        n_groups,
        include_zeros,
        n_threads,
    )
    
    if not result_ptr:
        raise RuntimeError("C++ function returned null pointer")
    
    result = result_ptr.contents
    means = np.ctypeslib.as_array(result.means, shape=(result.size,)).copy()
    vars = np.ctypeslib.as_array(result.vars, shape=(result.size,)).copy()
    
    _lib.mean_var_free_results(result_ptr)
    
    return means, vars

