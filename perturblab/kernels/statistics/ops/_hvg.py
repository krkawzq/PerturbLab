"""HVG (Highly Variable Genes) operators with automatic backend selection.

This module provides high-performance operators for highly variable gene detection,
automatically selecting the best available backend at import time.

Backend priority: C++ (SIMD + OpenMP) > Python (NumPy/SciPy fallback)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import scipy.sparse

logger = logging.getLogger(__name__)

# =============================================================================
# Backend Selection (Import-time)
# Priority: C++ > Cython > Numba > Python
# =============================================================================

_backend_name: str = "unknown"
_has_cpp: bool = False

# Try C++ backend first (highest performance)
try:
    from ..backends.cpp._hvg import clip_matrix_cpp as _clip_matrix_impl
    from ..backends.cpp._hvg import group_mean_var_cpp as _group_mean_var_impl
    from ..backends.cpp._hvg import group_var_cpp as _group_var_impl
    from ..backends.cpp._hvg import (
        has_cpp_backend,
    )
    from ..backends.cpp._hvg import loess_fit_cpp as _loess_fit_impl
    from ..backends.cpp._hvg import polynomial_fit_cpp as _polynomial_fit_impl
    from ..backends.cpp._hvg import sparse_clipped_moments_cpp as _sparse_clipped_moments_impl
    from ..backends.cpp._hvg import sparse_mean_var_cpp as _sparse_mean_var_impl

    if has_cpp_backend():
        _backend_name = "cpp"
        _has_cpp = True
        logger.info("Using C++ backend for HVG operators (SIMD + OpenMP)")
    else:
        raise ImportError("C++ backend not functional")

except (ImportError, RuntimeError) as e:
    logger.warning(f"C++ backend unavailable: {e}")
    _has_cpp = False

    # Try Cython backend (good performance)
    try:
        from ..backends.cython._hvg_wrapper import clip_matrix_cy as _clip_matrix_impl
        from ..backends.cython._hvg_wrapper import group_mean_var_cy as _group_mean_var_impl
        from ..backends.cython._hvg_wrapper import group_var_cy as _group_var_impl
        from ..backends.cython._hvg_wrapper import (
            has_cython_backend,
        )
        from ..backends.cython._hvg_wrapper import loess_fit_cy as _loess_fit_impl
        from ..backends.cython._hvg_wrapper import polynomial_fit_cy as _polynomial_fit_impl
        from ..backends.cython._hvg_wrapper import (
            sparse_clipped_moments_cy as _sparse_clipped_moments_impl,
        )
        from ..backends.cython._hvg_wrapper import sparse_mean_var_cy as _sparse_mean_var_impl

        if has_cython_backend():
            _backend_name = "cython"
            logger.info("Using Cython backend for HVG operators")
        else:
            raise ImportError("Cython backend not functional")

    except (ImportError, RuntimeError) as e2:
        logger.warning(f"Cython backend unavailable: {e2}")

        # Try Numba backend (optional, JIT-compiled)
        try:
            from ..backends.python.numba import group_var_numba as _group_var_impl
            from ..backends.python.numba import (
                has_numba_backend,
            )
            from ..backends.python.numba import (
                sparse_clipped_moments_numba as _sparse_clipped_moments_impl,
            )
            from ..backends.python.numba import sparse_mean_var_numba as _sparse_mean_var_impl

            if has_numba_backend():
                _backend_name = "numba"
                logger.info("Using Numba backend for HVG operators (JIT)")

                # Numba doesn't have all operators, use Python for missing ones
                from ..backends.python._hvg import clip_matrix_py as _clip_matrix_impl
                from ..backends.python._hvg import group_mean_var_py as _group_mean_var_impl
                from ..backends.python._hvg import loess_fit_py as _loess_fit_impl
                from ..backends.python._hvg import polynomial_fit_py as _polynomial_fit_impl
            else:
                raise ImportError("Numba backend not functional")

        except (ImportError, RuntimeError) as e3:
            logger.warning(f"Numba backend unavailable: {e3}")

            # Final fallback: Pure Python/NumPy
            logger.info("Using Python/NumPy backend for HVG operators (slowest)")

            from ..backends.python._hvg import clip_matrix_py as _clip_matrix_impl
            from ..backends.python._hvg import group_mean_var_py as _group_mean_var_impl
            from ..backends.python._hvg import group_var_py as _group_var_impl
            from ..backends.python._hvg import loess_fit_py as _loess_fit_impl
            from ..backends.python._hvg import polynomial_fit_py as _polynomial_fit_impl
            from ..backends.python._hvg import (
                sparse_clipped_moments_py as _sparse_clipped_moments_impl,
            )
            from ..backends.python._hvg import sparse_mean_var_py as _sparse_mean_var_impl

            _backend_name = "python"


def get_backend() -> str:
    """Get the current backend name."""
    return _backend_name


def has_cpp_backend() -> bool:
    """Check if C++ backend is available."""
    return _has_cpp


# =============================================================================
# Public API (Backend-agnostic)
# =============================================================================

__all__ = [
    "get_backend",
    "has_cpp_backend",
    "sparse_clipped_moments",
    "sparse_mean_var",
    "clip_matrix",
    "polynomial_fit",
    "loess_fit",
    "group_var",
    "group_mean_var",
]


def sparse_clipped_moments(
    X: scipy.sparse.csc_matrix,
    clip_vals: np.ndarray,
    n_threads: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute clipped moments for sparse matrix.
    
    Core operator for Seurat V3 HVG detection. For each column (gene),
    computes:
        sum_j = Σ_i min(X_ij, clip_j)
        sum_sq_j = Σ_i min(X_ij, clip_j)²
    
    Args:
        X: Sparse matrix (CSC format), shape (n_obs, n_vars)
        clip_vals: Clipping threshold for each column, shape (n_vars,)
        n_threads: Number of threads (0 = auto, only used by C++ backend)
    
    Returns:
        sums: Clipped sums, shape (n_vars,)
        sum_squares: Clipped sum of squares, shape (n_vars,)
    
    Backend:
        Auto-selected at import time: {backend}
    
    Examples:
        >>> import scipy.sparse as sp
        >>> X = sp.random(1000, 500, density=0.1, format='csc')
        >>> clip_vals = X.mean(axis=0).A1 * 2
        >>> sums, sum_sq = sparse_clipped_moments(X, clip_vals)
    """.format(
        backend=_backend_name
    )

    if _has_cpp:
        return _sparse_clipped_moments_impl(X, clip_vals, n_threads)
    else:
        return _sparse_clipped_moments_impl(X, clip_vals)


def sparse_mean_var(
    X: scipy.sparse.csc_matrix,
    include_zeros: bool = True,
    n_threads: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and variance for sparse matrix columns.
    
    Uses Welford's online algorithm for numerical stability.
    
    Args:
        X: Sparse matrix (CSC format), shape (n_obs, n_vars)
        include_zeros: Whether to include implicit zeros in calculation
        n_threads: Number of threads (0 = auto, only used by C++ backend)
    
    Returns:
        means: Column means, shape (n_vars,)
        vars: Column variances, shape (n_vars,)
    
    Backend:
        Auto-selected at import time: {backend}
    """.format(
        backend=_backend_name
    )

    if _has_cpp:
        return _sparse_mean_var_impl(X, include_zeros, n_threads)
    else:
        return _sparse_mean_var_impl(X, include_zeros)


def clip_matrix(
    X: np.ndarray,
    clip_vals: np.ndarray,
    n_threads: int = 0,
) -> np.ndarray:
    """Clip dense matrix by column (in-place operation).
    
    Args:
        X: Dense matrix, shape (n_obs, n_vars)
        clip_vals: Clipping threshold for each column, shape (n_vars,)
        n_threads: Number of threads (0 = auto, only used by C++ backend)
    
    Returns:
        Clipped matrix (same object as X, modified in-place)
    
    Backend:
        Auto-selected at import time: {backend}
    
    Examples:
        >>> X = np.random.randn(1000, 500)
        >>> clip_vals = X.mean(axis=0) * 2
        >>> X_clipped = clip_matrix(X, clip_vals)  # X is modified in-place
    """.format(
        backend=_backend_name
    )

    if _has_cpp:
        return _clip_matrix_impl(X, clip_vals, n_threads)
    else:
        return _clip_matrix_impl(X, clip_vals)


def polynomial_fit(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 2,
    weights: Optional[np.ndarray] = None,
    return_coeffs: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Fit polynomial regression.
    
    Args:
        x: Input x coordinates, shape (n,)
        y: Input y coordinates, shape (n,)
        degree: Polynomial degree (typically 2 for HVG detection)
        weights: Optional weights, shape (n,)
        return_coeffs: Whether to return polynomial coefficients
    
    Returns:
        fitted: Fitted y values, shape (n,)
        coeffs: Coefficients [a0, a1, ..., an] if return_coeffs=True, else None
    
    Backend:
        Auto-selected at import time: {backend}
    
    Examples:
        >>> x = np.linspace(0, 10, 100)
        >>> y = 2 + 3*x + 0.5*x**2 + np.random.randn(100)
        >>> fitted, coeffs = polynomial_fit(x, y, degree=2, return_coeffs=True)
    """.format(
        backend=_backend_name
    )

    return _polynomial_fit_impl(x, y, degree, weights, return_coeffs)


def loess_fit(
    x: np.ndarray,
    y: np.ndarray,
    span: float = 0.3,
    n_threads: int = 0,
) -> np.ndarray:
    """Fit LOESS (Locally Weighted Scatterplot Smoothing).
    
    Args:
        x: Input x coordinates, shape (n,)
        y: Input y coordinates, shape (n,)
        span: Smoothing parameter (fraction of data to use for each local fit)
        n_threads: Number of threads (0 = auto, only used by C++ backend)
    
    Returns:
        fitted: Fitted y values, shape (n,)
    
    Backend:
        Auto-selected at import time: {backend}
    
    Examples:
        >>> x = np.sort(np.random.rand(100))
        >>> y = np.sin(x * 5) + np.random.randn(100) * 0.1
        >>> fitted = loess_fit(x, y, span=0.2)
    """.format(
        backend=_backend_name
    )

    if _has_cpp:
        return _loess_fit_impl(x, y, span, n_threads)
    else:
        return _loess_fit_impl(x, y, span)


def group_var(
    X: scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_groups: int,
    include_zeros: bool = True,
    n_threads: int = 0,
) -> np.ndarray:
    """Compute group-wise variance for sparse matrix.
    
    Args:
        X: Sparse matrix (CSC format), shape (n_obs, n_vars)
        group_id: Group labels for each row, shape (n_obs,), dtype int32
        n_groups: Number of groups
        include_zeros: Whether to include implicit zeros
        n_threads: Number of threads (0 = auto, only used by C++ backend)
    
    Returns:
        Group variances, shape (n_vars, n_groups), flattened in C-order
        Access as: result[col * n_groups + group]
    
    Backend:
        Auto-selected at import time: {backend}
    """.format(
        backend=_backend_name
    )

    if _has_cpp:
        return _group_var_impl(X, group_id, n_groups, include_zeros, n_threads)
    else:
        return _group_var_impl(X, group_id, n_groups, include_zeros)


def group_mean_var(
    X: scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_groups: int,
    include_zeros: bool = True,
    n_threads: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute group-wise mean and variance for sparse matrix.
    
    Uses Welford's online algorithm for single-pass computation.
    
    Args:
        X: Sparse matrix (CSC format), shape (n_obs, n_vars)
        group_id: Group labels for each row, shape (n_obs,), dtype int32
        n_groups: Number of groups
        include_zeros: Whether to include implicit zeros
        n_threads: Number of threads (0 = auto, only used by C++ backend)
    
    Returns:
        means: Group means, shape (n_vars, n_groups), flattened
        vars: Group variances, shape (n_vars, n_groups), flattened
    
    Backend:
        Auto-selected at import time: {backend}
    """.format(
        backend=_backend_name
    )

    if _has_cpp:
        return _group_mean_var_impl(X, group_id, n_groups, include_zeros, n_threads)
    else:
        return _group_mean_var_impl(X, group_id, n_groups, include_zeros)
