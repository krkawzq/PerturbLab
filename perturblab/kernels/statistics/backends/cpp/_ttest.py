"""C API interface for t-test kernels.

Provides ctypes-based interface to the C++ t-test implementations.
"""

import ctypes
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from perturblab.utils import get_logger

logger = get_logger(__name__)

__all__ = ["has_ttest_backend", "ttest_cpp", "student_ttest_cpp", "welch_ttest_cpp", "log_fold_change_cpp"]


# =============================================================================
# C API Structures
# =============================================================================

class TTestResults(ctypes.Structure):
    """C structure for t-test results."""
    _fields_ = [
        ("t_statistic", ctypes.POINTER(ctypes.c_double)),
        ("p_value", ctypes.POINTER(ctypes.c_double)),
        ("mean_diff", ctypes.POINTER(ctypes.c_double)),
        ("log2_fc", ctypes.POINTER(ctypes.c_double)),
        ("size", ctypes.c_size_t),
    ]


# =============================================================================
# Load C++ Library
# =============================================================================

_lib: Optional[ctypes.CDLL] = None


def _find_library() -> Optional[Path]:
    """Find the compiled C++ library (shared with mannwhitneyu)."""
    module_dir = Path(__file__).parent
    
    lib_names = [
        "libmwu_kernel.so",
        "libmwu_kernel.dylib",
        "mwu_kernel.dll",
    ]
    
    for lib_name in lib_names:
        lib_path = module_dir / lib_name
        if lib_path.exists():
            return lib_path
    
    return None


def _load_library() -> Optional[ctypes.CDLL]:
    """Load the C++ library and configure function signatures."""
    lib_path = _find_library()
    
    if lib_path is None:
        logger.debug("C++ library not found for t-test")
        return None
    
    try:
        lib = ctypes.CDLL(str(lib_path))
        
        # Student's t-test
        lib.student_ttest_csc_capi.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # data
            ctypes.POINTER(ctypes.c_int64),   # indices
            ctypes.POINTER(ctypes.c_int64),   # indptr
            ctypes.POINTER(ctypes.c_int32),   # group_id
            ctypes.c_size_t,                  # n_rows
            ctypes.c_size_t,                  # n_cols
            ctypes.c_size_t,                  # nnz
            ctypes.c_int,                     # n_targets
            ctypes.c_int,                     # threads
        ]
        lib.student_ttest_csc_capi.restype = ctypes.POINTER(TTestResults)
        
        # Welch's t-test
        lib.welch_ttest_csc_capi.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # data
            ctypes.POINTER(ctypes.c_int64),   # indices
            ctypes.POINTER(ctypes.c_int64),   # indptr
            ctypes.POINTER(ctypes.c_int32),   # group_id
            ctypes.c_size_t,                  # n_rows
            ctypes.c_size_t,                  # n_cols
            ctypes.c_size_t,                  # nnz
            ctypes.c_int,                     # n_targets
            ctypes.c_int,                     # threads
        ]
        lib.welch_ttest_csc_capi.restype = ctypes.POINTER(TTestResults)
        
        # Log fold change
        lib.log_fold_change_csc_capi.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # data
            ctypes.POINTER(ctypes.c_int64),   # indices
            ctypes.POINTER(ctypes.c_int64),   # indptr
            ctypes.POINTER(ctypes.c_int32),   # group_id
            ctypes.c_size_t,                  # n_rows
            ctypes.c_size_t,                  # n_cols
            ctypes.c_size_t,                  # nnz
            ctypes.c_int,                     # n_targets
            ctypes.c_double,                  # pseudocount
            ctypes.c_int,                     # threads
        ]
        lib.log_fold_change_csc_capi.restype = ctypes.POINTER(TTestResults)
        
        # Free results
        lib.ttest_free_results.argtypes = [ctypes.POINTER(TTestResults)]
        lib.ttest_free_results.restype = None
        
        logger.info(f"C++ library loaded successfully for t-test: {lib_path}")
        return lib
        
    except Exception as e:
        logger.warning(f"Failed to load C++ library for t-test: {e}")
        return None


_lib = _load_library()


def has_ttest_backend() -> bool:
    """Check if C++ backend is available for t-test."""
    return _lib is not None


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_ttest_results(
    result_ptr: ctypes.POINTER(TTestResults),
    n_targets: int,
    n_cols: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract results from C structure."""
    result = result_ptr.contents
    size = result.size
    
    # Copy arrays
    t_statistic = np.ctypeslib.as_array(result.t_statistic, shape=(size,)).copy()
    p_value = np.ctypeslib.as_array(result.p_value, shape=(size,)).copy()
    mean_diff = np.ctypeslib.as_array(result.mean_diff, shape=(size,)).copy()
    log2_fc = np.ctypeslib.as_array(result.log2_fc, shape=(size,)).copy()
    
    # Reshape to (n_targets, n_cols)
    t_statistic = t_statistic.reshape(n_targets, n_cols)
    p_value = p_value.reshape(n_targets, n_cols)
    mean_diff = mean_diff.reshape(n_targets, n_cols)
    log2_fc = log2_fc.reshape(n_targets, n_cols)
    
    return t_statistic, p_value, mean_diff, log2_fc


# =============================================================================
# Public API
# =============================================================================

def student_ttest_cpp(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    group_id: np.ndarray,
    n_targets: int,
    threads: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Student's t-test using C++ backend."""
    if _lib is None:
        raise RuntimeError("C++ backend not available")
    
    n_rows = group_id.shape[0]
    n_cols = indptr.shape[0] - 1
    nnz = data.shape[0]
    
    # Ensure correct types
    if data.dtype != np.float64:
        data = data.astype(np.float64)
    if indices.dtype != np.int64:
        indices = indices.astype(np.int64)
    if indptr.dtype != np.int64:
        indptr = indptr.astype(np.int64)
    if group_id.dtype != np.int32:
        group_id = group_id.astype(np.int32)
    
    # Get pointers
    data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    indices_ptr = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    indptr_ptr = indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    group_id_ptr = group_id.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    
    # Call C++ function
    result_ptr = _lib.student_ttest_csc_capi(
        data_ptr, indices_ptr, indptr_ptr, group_id_ptr,
        n_rows, n_cols, nnz, n_targets, threads
    )
    
    if not result_ptr:
        raise RuntimeError("C++ student_ttest function failed")
    
    try:
        return _extract_ttest_results(result_ptr, n_targets, n_cols)
    finally:
        _lib.ttest_free_results(result_ptr)


def welch_ttest_cpp(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    group_id: np.ndarray,
    n_targets: int,
    threads: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Welch's t-test using C++ backend."""
    if _lib is None:
        raise RuntimeError("C++ backend not available")
    
    n_rows = group_id.shape[0]
    n_cols = indptr.shape[0] - 1
    nnz = data.shape[0]
    
    # Ensure correct types
    if data.dtype != np.float64:
        data = data.astype(np.float64)
    if indices.dtype != np.int64:
        indices = indices.astype(np.int64)
    if indptr.dtype != np.int64:
        indptr = indptr.astype(np.int64)
    if group_id.dtype != np.int32:
        group_id = group_id.astype(np.int32)
    
    # Get pointers
    data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    indices_ptr = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    indptr_ptr = indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    group_id_ptr = group_id.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    
    # Call C++ function
    result_ptr = _lib.welch_ttest_csc_capi(
        data_ptr, indices_ptr, indptr_ptr, group_id_ptr,
        n_rows, n_cols, nnz, n_targets, threads
    )
    
    if not result_ptr:
        raise RuntimeError("C++ welch_ttest function failed")
    
    try:
        return _extract_ttest_results(result_ptr, n_targets, n_cols)
    finally:
        _lib.ttest_free_results(result_ptr)


def log_fold_change_cpp(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    group_id: np.ndarray,
    n_targets: int,
    pseudocount: float = 1e-9,
    threads: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Log fold change using C++ backend."""
    if _lib is None:
        raise RuntimeError("C++ backend not available")
    
    n_rows = group_id.shape[0]
    n_cols = indptr.shape[0] - 1
    nnz = data.shape[0]
    
    # Ensure correct types
    if data.dtype != np.float64:
        data = data.astype(np.float64)
    if indices.dtype != np.int64:
        indices = indices.astype(np.int64)
    if indptr.dtype != np.int64:
        indptr = indptr.astype(np.int64)
    if group_id.dtype != np.int32:
        group_id = group_id.astype(np.int32)
    
    # Get pointers
    data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    indices_ptr = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    indptr_ptr = indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    group_id_ptr = group_id.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    
    # Call C++ function
    result_ptr = _lib.log_fold_change_csc_capi(
        data_ptr, indices_ptr, indptr_ptr, group_id_ptr,
        n_rows, n_cols, nnz, n_targets, pseudocount, threads
    )
    
    if not result_ptr:
        raise RuntimeError("C++ log_fold_change function failed")
    
    try:
        return _extract_ttest_results(result_ptr, n_targets, n_cols)
    finally:
        _lib.ttest_free_results(result_ptr)


# Alias for backward compatibility and consistency
ttest_cpp = welch_ttest_cpp
