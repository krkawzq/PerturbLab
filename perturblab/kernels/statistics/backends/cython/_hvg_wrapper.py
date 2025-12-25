"""Python wrapper for Cython HVG operators."""

import logging

import numpy as np
import scipy.sparse

logger = logging.getLogger(__name__)

__all__ = [
    "has_cython_backend",
    "sparse_clipped_moments_cy",
    "sparse_mean_var_cy",
    "clip_matrix_cy",
    "polynomial_fit_cy",
    "loess_fit_cy",
    "group_var_cy",
    "group_mean_var_cy",
]

# Try to import Cython module
_has_cython = False

try:
    from ._hvg import clip_matrix_cy as _clip_matrix_cy
    from ._hvg import group_mean_var_cy as _group_mean_var_cy
    from ._hvg import group_var_cy as _group_var_cy
    from ._hvg import loess_fit_cy as _loess_fit_cy
    from ._hvg import polynomial_fit_cy as _polynomial_fit_cy
    from ._hvg import sparse_clipped_moments_cy as _sparse_clipped_moments_cy
    from ._hvg import sparse_mean_var_cy as _sparse_mean_var_cy

    _has_cython = True
    logger.info("Cython backend loaded successfully for HVG operators")
except ImportError as e:
    logger.warning(f"Cython backend not available for HVG: {e}")


def has_cython_backend() -> bool:
    """Check if Cython backend is available."""
    return _has_cython


# =============================================================================
# Wrapper Functions
# =============================================================================


def sparse_clipped_moments_cy(
    X: scipy.sparse.csc_matrix,
    clip_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute clipped moments (Cython backend)."""
    if not _has_cython:
        raise RuntimeError("Cython backend not available")

    if not scipy.sparse.isspmatrix_csc(X):
        X = X.tocsc()

    X = X.astype(np.float64, copy=False)
    clip_vals = np.ascontiguousarray(clip_vals, dtype=np.float64)

    return _sparse_clipped_moments_cy(X.data, X.indices, X.indptr, clip_vals)


def sparse_mean_var_cy(
    X: scipy.sparse.csc_matrix,
    include_zeros: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and variance (Cython backend)."""
    if not _has_cython:
        raise RuntimeError("Cython backend not available")

    if not scipy.sparse.isspmatrix_csc(X):
        X = X.tocsc()

    X = X.astype(np.float64, copy=False)

    return _sparse_mean_var_cy(X.data, X.indices, X.indptr, X.shape[0], include_zeros)


def clip_matrix_cy(
    X: np.ndarray,
    clip_vals: np.ndarray,
) -> np.ndarray:
    """Clip matrix by column (Cython backend, in-place)."""
    if not _has_cython:
        raise RuntimeError("Cython backend not available")

    if not X.flags.c_contiguous:
        X = np.ascontiguousarray(X)

    X = X.astype(np.float64, copy=False)
    clip_vals = np.ascontiguousarray(clip_vals, dtype=np.float64)

    _clip_matrix_cy(X, clip_vals)
    return X


def polynomial_fit_cy(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 2,
    weights: np.ndarray | None = None,
    return_coeffs: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit polynomial regression (Cython backend)."""
    if not _has_cython:
        raise RuntimeError("Cython backend not available")

    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)

    if weights is not None:
        weights = np.ascontiguousarray(weights, dtype=np.float64)

    fitted, coeffs = _polynomial_fit_cy(x, y, degree, weights)

    if return_coeffs:
        return fitted, coeffs
    else:
        return fitted, None


def loess_fit_cy(
    x: np.ndarray,
    y: np.ndarray,
    span: float = 0.3,
) -> np.ndarray:
    """Fit LOESS (Cython backend)."""
    if not _has_cython:
        raise RuntimeError("Cython backend not available")

    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)

    return _loess_fit_cy(x, y, span)


def group_var_cy(
    X: scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_groups: int,
    include_zeros: bool = True,
) -> np.ndarray:
    """Compute group-wise variance (Cython backend)."""
    if not _has_cython:
        raise RuntimeError("Cython backend not available")

    if not scipy.sparse.isspmatrix_csc(X):
        X = X.tocsc()

    X = X.astype(np.float64, copy=False)
    group_id = np.ascontiguousarray(group_id, dtype=np.int32)

    return _group_var_cy(X.data, X.indices, X.indptr, X.shape[0], group_id, n_groups, include_zeros)


def group_mean_var_cy(
    X: scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_groups: int,
    include_zeros: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute group-wise mean and variance (Cython backend)."""
    if not _has_cython:
        raise RuntimeError("Cython backend not available")

    if not scipy.sparse.isspmatrix_csc(X):
        X = X.tocsc()

    X = X.astype(np.float64, copy=False)
    group_id = np.ascontiguousarray(group_id, dtype=np.int32)

    return _group_mean_var_cy(
        X.data, X.indices, X.indptr, X.shape[0], group_id, n_groups, include_zeros
    )
