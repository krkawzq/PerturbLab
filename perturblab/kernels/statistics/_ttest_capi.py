"""C API interface for t-test kernels.

Provides ctypes-based interface to the C++ t-test implementations.
"""

import ctypes
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from perturblab.utils import get_logger

logger = get_logger(__name__)

__all__ = ["has_ttest_backend", "student_ttest_cpp", "welch_ttest_cpp", "log_fold_change_cpp"]


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
    """Load the C++ library and setup function signatures."""
    lib_path = _find_library()
    if lib_path is None:
        return None
    
    try:
        lib = ctypes.CDLL(str(lib_path))
        
        # Setup function signatures for t-tests
        # student_ttest_csc_capi
        lib.student_ttest_csc_capi.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,  # n_rows
            ctypes.c_size_t,  # n_cols
            ctypes.c_size_t,  # nnz
            ctypes.c_size_t,  # n_targets
            ctypes.c_int,     # threads
        ]
        lib.student_ttest_csc_capi.restype = ctypes.POINTER(TTestResults)
        
        # welch_ttest_csc_capi
        lib.welch_ttest_csc_capi.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        lib.welch_ttest_csc_capi.restype = ctypes.POINTER(TTestResults)
        
        # log_fold_change_csc_capi
        lib.log_fold_change_csc_capi.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,  # pseudocount
            ctypes.c_int,
        ]
        lib.log_fold_change_csc_capi.restype = ctypes.POINTER(TTestResults)
        
        # ttest_free_results
        lib.ttest_free_results.argtypes = [ctypes.POINTER(TTestResults)]
        lib.ttest_free_results.restype = None
        
        return lib
    
    except Exception as e:
        logger.warning(f"Failed to load C++ library for t-test: {e}")
        return None


# Load library on module import
_lib = _load_library()


def has_ttest_backend() -> bool:
    """Check if C++ backend is available for t-test."""
    return _lib is not None


# =============================================================================
# Helper Function
# =============================================================================

def _extract_ttest_results(
    result_ptr: ctypes.POINTER(TTestResults),
    n_targets: int,
    n_cols: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract results from C structure."""
    result = result_ptr.contents
    result_size = result.size
    
    # Copy to numpy arrays
    t_stat = np.frombuffer(
        (ctypes.c_double * result_size).from_address(
            ctypes.addressof(result.t_statistic.contents)
        ),
        dtype=np.float64
    ).copy()
    
    p_val = np.frombuffer(
        (ctypes.c_double * result_size).from_address(
            ctypes.addressof(result.p_value.contents)
        ),
        dtype=np.float64
    ).copy()
    
    mean_diff = np.frombuffer(
        (ctypes.c_double * result_size).from_address(
            ctypes.addressof(result.mean_diff.contents)
        ),
        dtype=np.float64
    ).copy()
    
    log2_fc = np.frombuffer(
        (ctypes.c_double * result_size).from_address(
            ctypes.addressof(result.log2_fc.contents)
        ),
        dtype=np.float64
    ).copy()
    
    # Reshape: C++ returns flattened [C, n_targets], we want (n_targets, C)
    t_stat = t_stat.reshape(n_cols, n_targets).T
    p_val = p_val.reshape(n_cols, n_targets).T
    mean_diff = mean_diff.reshape(n_cols, n_targets).T
    log2_fc = log2_fc.reshape(n_cols, n_targets).T
    
    return t_stat, p_val, mean_diff, log2_fc


# =============================================================================
# C++ Backend Functions
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

