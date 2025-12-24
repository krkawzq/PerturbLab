"""C API interface for high-performance C++ kernels.

This module provides a ctypes-based interface to the C++ implementation,
which serves as the primary high-performance backend. Falls back to Cython
if the C++ library is not available.
"""

import ctypes
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.sparse

logger = logging.getLogger(__name__)

__all__ = ["has_cpp_backend", "mannwhitneyu_cpp", "group_mean_cpp"]


# =============================================================================
# C API Structures
# =============================================================================


class MWUResults(ctypes.Structure):
    """C structure for Mann-Whitney U results."""

    _fields_ = [
        ("U1", ctypes.POINTER(ctypes.c_double)),
        ("U2", ctypes.POINTER(ctypes.c_double)),
        ("P", ctypes.POINTER(ctypes.c_double)),
        ("size", ctypes.c_size_t),
    ]


class GroupMeanResults(ctypes.Structure):
    """C structure for group mean results."""

    _fields_ = [
        ("means", ctypes.POINTER(ctypes.c_double)),
        ("size", ctypes.c_size_t),
    ]


# =============================================================================
# Load C++ Library
# =============================================================================

_lib: Optional[ctypes.CDLL] = None


def _find_library() -> Optional[Path]:
    """Find the compiled C++ library.

    Searches in the same directory as this module.
    """
    module_dir = Path(__file__).parent

    # Possible library names
    lib_names = [
        "libmwu_kernel.so",  # Linux
        "libmwu_kernel.dylib",  # macOS
        "mwu_kernel.dll",  # Windows
    ]

    for lib_name in lib_names:
        lib_path = module_dir / lib_name
        if lib_path.exists():
            logger.info(f"Found C++ library: {lib_path}")
            return lib_path

    return None


def _load_library() -> Optional[ctypes.CDLL]:
    """Load the C++ library and setup function signatures."""
    lib_path = _find_library()
    if lib_path is None:
        logger.warning("C++ library not found, will use Cython fallback")
        return None

    try:
        lib = ctypes.CDLL(str(lib_path))

        # Setup function signatures
        # mannwhitneyu_csc_capi
        lib.mannwhitneyu_csc_capi.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # data
            ctypes.POINTER(ctypes.c_int64),  # indices
            ctypes.POINTER(ctypes.c_int64),  # indptr
            ctypes.POINTER(ctypes.c_int32),  # group_id
            ctypes.c_size_t,  # n_rows
            ctypes.c_size_t,  # n_cols
            ctypes.c_size_t,  # nnz
            ctypes.c_size_t,  # n_targets
            ctypes.c_bool,  # tie_correction
            ctypes.c_bool,  # use_continuity
            ctypes.c_int,  # threads
        ]
        lib.mannwhitneyu_csc_capi.restype = ctypes.POINTER(MWUResults)

        # mwu_free_results
        lib.mwu_free_results.argtypes = [ctypes.POINTER(MWUResults)]
        lib.mwu_free_results.restype = None

        # group_mean_csc_capi
        lib.group_mean_csc_capi.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # data
            ctypes.POINTER(ctypes.c_int64),  # indices
            ctypes.POINTER(ctypes.c_int64),  # indptr
            ctypes.POINTER(ctypes.c_int32),  # group_id
            ctypes.c_size_t,  # n_rows
            ctypes.c_size_t,  # n_cols
            ctypes.c_size_t,  # nnz
            ctypes.c_size_t,  # n_groups
            ctypes.c_bool,  # include_zeros
            ctypes.c_int,  # threads
        ]
        lib.group_mean_csc_capi.restype = ctypes.POINTER(GroupMeanResults)

        # group_mean_free_results
        lib.group_mean_free_results.argtypes = [ctypes.POINTER(GroupMeanResults)]
        lib.group_mean_free_results.restype = None

        # Float32 variants
        lib.mannwhitneyu_csc_f32_capi.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # data
            ctypes.POINTER(ctypes.c_int64),  # indices
            ctypes.POINTER(ctypes.c_int64),  # indptr
            ctypes.POINTER(ctypes.c_int32),  # group_id
            ctypes.c_size_t,  # n_rows
            ctypes.c_size_t,  # n_cols
            ctypes.c_size_t,  # nnz
            ctypes.c_size_t,  # n_targets
            ctypes.c_bool,  # tie_correction
            ctypes.c_bool,  # use_continuity
            ctypes.c_int,  # threads
        ]
        lib.mannwhitneyu_csc_f32_capi.restype = ctypes.POINTER(MWUResults)

        lib.group_mean_csc_f32_capi.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # data
            ctypes.POINTER(ctypes.c_int64),  # indices
            ctypes.POINTER(ctypes.c_int64),  # indptr
            ctypes.POINTER(ctypes.c_int32),  # group_id
            ctypes.c_size_t,  # n_rows
            ctypes.c_size_t,  # n_cols
            ctypes.c_size_t,  # nnz
            ctypes.c_size_t,  # n_groups
            ctypes.c_bool,  # include_zeros
            ctypes.c_int,  # threads
        ]
        lib.group_mean_csc_f32_capi.restype = ctypes.POINTER(GroupMeanResults)

        logger.info("C++ library loaded successfully")
        return lib

    except Exception as e:
        logger.warning(f"Failed to load C++ library: {e}")
        return None


# Load library on module import
_lib = _load_library()


def has_cpp_backend() -> bool:
    """Check if C++ backend is available."""
    return _lib is not None


# =============================================================================
# C++ Backend Functions
# =============================================================================


def mannwhitneyu_cpp(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    group_id: np.ndarray,
    n_targets: int,
    tie_correction: bool = True,
    use_continuity: bool = True,
    threads: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mann-Whitney U test using C++ backend.

    Args:
        data: CSC data array
        indices: CSC row indices
        indptr: CSC column pointers
        group_id: Group assignments (int32)
        n_targets: Number of target groups
        tie_correction: Whether to apply tie correction
        use_continuity: Whether to apply continuity correction
        threads: Number of threads (-1 for all)

    Returns:
        Tuple of (U1, U2, P) arrays of shape (n_targets, n_cols)
    """
    if _lib is None:
        raise RuntimeError("C++ backend not available")

    n_rows = group_id.shape[0]
    n_cols = indptr.shape[0] - 1
    nnz = data.shape[0]

    # Ensure correct types
    if indices.dtype != np.int64:
        indices = indices.astype(np.int64)
    if indptr.dtype != np.int64:
        indptr = indptr.astype(np.int64)
    if group_id.dtype != np.int32:
        group_id = group_id.astype(np.int32)

    # Select function based on data dtype
    if data.dtype == np.float64:
        func = _lib.mannwhitneyu_csc_capi
        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    elif data.dtype == np.float32:
        func = _lib.mannwhitneyu_csc_f32_capi
        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    else:
        # Convert to float64
        data = data.astype(np.float64)
        func = _lib.mannwhitneyu_csc_capi
        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Get pointers
    indices_ptr = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    indptr_ptr = indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    group_id_ptr = group_id.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    # Call C++ function
    result_ptr = func(
        data_ptr,
        indices_ptr,
        indptr_ptr,
        group_id_ptr,
        n_rows,
        n_cols,
        nnz,
        n_targets,
        tie_correction,
        use_continuity,
        threads,
    )

    if not result_ptr:
        raise RuntimeError("C++ function failed")

    try:
        # Extract results
        result = result_ptr.contents
        result_size = result.size

        # Copy to numpy arrays
        U1 = np.frombuffer(
            (ctypes.c_double * result_size).from_address(ctypes.addressof(result.U1.contents)),
            dtype=np.float64,
        ).copy()

        U2 = np.frombuffer(
            (ctypes.c_double * result_size).from_address(ctypes.addressof(result.U2.contents)),
            dtype=np.float64,
        ).copy()

        P = np.frombuffer(
            (ctypes.c_double * result_size).from_address(ctypes.addressof(result.P.contents)),
            dtype=np.float64,
        ).copy()

        # Reshape: C++ returns flattened [C, n_targets], we want (n_targets, C)
        U1 = U1.reshape(n_cols, n_targets).T
        U2 = U2.reshape(n_cols, n_targets).T
        P = P.reshape(n_cols, n_targets).T

        return U1, U2, P

    finally:
        # Free C++ memory
        _lib.mwu_free_results(result_ptr)


def group_mean_cpp(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    group_id: np.ndarray,
    n_groups: int,
    include_zeros: bool = True,
    threads: int = -1,
) -> np.ndarray:
    """Group mean computation using C++ backend.

    Args:
        data: CSC data array
        indices: CSC row indices
        indptr: CSC column pointers
        group_id: Group assignments (int32)
        n_groups: Total number of groups
        include_zeros: Whether to include zeros
        threads: Number of threads (-1 for all)

    Returns:
        Array of shape (n_groups, n_cols) containing group means
    """
    if _lib is None:
        raise RuntimeError("C++ backend not available")

    n_rows = group_id.shape[0]
    n_cols = indptr.shape[0] - 1
    nnz = data.shape[0]

    # Ensure correct types
    if indices.dtype != np.int64:
        indices = indices.astype(np.int64)
    if indptr.dtype != np.int64:
        indptr = indptr.astype(np.int64)
    if group_id.dtype != np.int32:
        group_id = group_id.astype(np.int32)

    # Select function based on data dtype
    if data.dtype == np.float64:
        func = _lib.group_mean_csc_capi
        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    elif data.dtype == np.float32:
        func = _lib.group_mean_csc_f32_capi
        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    else:
        data = data.astype(np.float64)
        func = _lib.group_mean_csc_capi
        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # Get pointers
    indices_ptr = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    indptr_ptr = indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    group_id_ptr = group_id.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    # Call C++ function
    result_ptr = func(
        data_ptr,
        indices_ptr,
        indptr_ptr,
        group_id_ptr,
        n_rows,
        n_cols,
        nnz,
        n_groups,
        include_zeros,
        threads,
    )

    if not result_ptr:
        raise RuntimeError("C++ function failed")

    try:
        # Extract results
        result = result_ptr.contents
        result_size = result.size

        # Copy to numpy array
        means = np.frombuffer(
            (ctypes.c_double * result_size).from_address(ctypes.addressof(result.means.contents)),
            dtype=np.float64,
        ).copy()

        # Reshape: C++ returns flattened [C, G], we want (G, C)
        means = means.reshape(n_cols, n_groups).T

        return means

    finally:
        # Free C++ memory
        _lib.group_mean_free_results(result_ptr)
