"""Mann-Whitney U test and group mean operations.

This module provides a unified interface with automatic backend selection
performed at import time for maximum performance.

Backend priority: C++ > Cython > Python (NumPy/SciPy)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csc_matrix, issparse

from perturblab.utils import get_logger

if TYPE_CHECKING:
    from scipy.sparse import spmatrix

logger = get_logger()

__all__ = ["mannwhitneyu", "group_mean"]


# =============================================================================
# Backend Selection (performed at import time)
# =============================================================================

_BACKEND_NAME = None
_mannwhitneyu_impl = None
_group_mean_impl = None


def _select_backend():
    """Select backend at import time."""
    global _BACKEND_NAME, _mannwhitneyu_impl, _group_mean_impl
    
    # Try C++ backend first
    try:
        from ..backends.cpp._mannwhitneyu import has_cpp_backend
        if has_cpp_backend():
            from ..backends.cpp._mannwhitneyu import mannwhitneyu_cpp, group_mean_cpp
            _BACKEND_NAME = "C++"
            _mannwhitneyu_impl = mannwhitneyu_cpp
            _group_mean_impl = group_mean_cpp
            logger.debug("Mann-Whitney U backend: C++ (highest performance)")
            return
    except (ImportError, OSError):
        pass
    
    # Try Cython backend
    try:
        from ..backends.cython.mannwhitneyu import mannwhitneyu_csc, group_mean_csc
        _BACKEND_NAME = "Cython"
        _mannwhitneyu_impl = mannwhitneyu_csc
        _group_mean_impl = group_mean_csc
        logger.debug("Mann-Whitney U backend: Cython (fast fallback)")
        return
    except ImportError:
        pass
    
    # Fall back to Python (NumPy/SciPy)
    from ..backends.python._mannwhitneyu import mannwhitneyu_scipy, group_mean_numpy
    _BACKEND_NAME = "Python"
    _mannwhitneyu_impl = mannwhitneyu_scipy
    _group_mean_impl = group_mean_numpy
    logger.warning("Mann-Whitney U backend: Python (NumPy/SciPy) - slower, consider installing compiled backends")


# Select backend on module import
_select_backend()


# =============================================================================
# Wrapper Functions (with input preprocessing)
# =============================================================================

def mannwhitneyu(
    X: np.ndarray | spmatrix,
    group_id: np.ndarray,
    n_targets: int,
    *,
    threads: int = -1,
    use_continuity: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mann-Whitney U test for differential expression.
    
    Performs Mann-Whitney U test (Wilcoxon rank-sum test) for multiple
    target groups against a reference group.
    
    Parameters
    ----------
    X : np.ndarray or sparse matrix
        Expression matrix, shape (n_cells, n_genes).
        If sparse, will be converted to CSC format.
    group_id : np.ndarray
        Group assignments for each cell, shape (n_cells,).
        - 0: reference group
        - 1 to n_targets: target groups
    n_targets : int
        Number of target groups.
    threads : int, default=-1
        Number of threads to use. -1 means use all available threads.
        Only used by C++ and Cython backends.
    use_continuity : bool, default=True
        Whether to use continuity correction.
        Only used by Python backend.
    
    Returns
    -------
    U1 : np.ndarray
        U statistic for target groups, shape (n_targets, n_genes).
    U2 : np.ndarray
        U statistic for reference group, shape (n_targets, n_genes).
    P : np.ndarray
        Two-sided p-values, shape (n_targets, n_genes).
    
    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> from perturblab.kernels.statistics import mannwhitneyu
    >>> 
    >>> # Create example data
    >>> X = sp.random(100, 50, density=0.1, format='csc')
    >>> group_id = np.array([0]*40 + [1]*30 + [2]*30, dtype=np.int32)
    >>> 
    >>> # Run test
    >>> U1, U2, P = mannwhitneyu(X, group_id, n_targets=2)
    >>> 
    >>> # Find significant genes (p < 0.05)
    >>> significant = P < 0.05
    """
    
    # Convert to CSC if needed
    if issparse(X):
        if not isinstance(X, csc_matrix):
            X = csc_matrix(X)
    else:
        X = csc_matrix(X)
    
    # Ensure group_id is int32
    group_id = np.asarray(group_id, dtype=np.int32)
    
    # Call backend-specific implementation
    if _BACKEND_NAME == "C++":
        return _mannwhitneyu_impl(
            X.data, X.indices, X.indptr, group_id, n_targets,
            tie_correction=True, use_continuity=use_continuity, threads=threads
        )
    elif _BACKEND_NAME == "Cython":
        return _mannwhitneyu_impl(X, group_id, n_targets, threads=threads)
    else:  # Python
        return _mannwhitneyu_impl(X, group_id, n_targets, use_continuity=use_continuity)


def group_mean(
    X: np.ndarray | spmatrix,
    group_id: np.ndarray,
    n_groups: int,
    *,
    threads: int = -1,
    include_zeros: bool = True,
) -> np.ndarray:
    """Compute group-wise mean expression.
    
    Calculates mean expression for each group efficiently, with support
    for sparse matrices and parallel computation.
    
    Parameters
    ----------
    X : np.ndarray or sparse matrix
        Expression matrix, shape (n_cells, n_genes).
        If sparse, will be converted to CSC format.
    group_id : np.ndarray
        Group assignments for each cell, shape (n_cells,).
        Values should be 0 to n_groups-1.
    n_groups : int
        Number of groups.
    threads : int, default=-1
        Number of threads to use. -1 means use all available threads.
        Only used by C++ and Cython backends.
    include_zeros : bool, default=True
        Whether to include zeros in mean calculation.
        If False, only non-zero values are averaged (for sparse data).
    
    Returns
    -------
    means : np.ndarray
        Group means, shape (n_groups, n_genes).
    
    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> from perturblab.kernels.statistics import group_mean
    >>> 
    >>> # Create example data
    >>> X = sp.random(100, 50, density=0.1, format='csc')
    >>> group_id = np.array([0]*40 + [1]*30 + [2]*30, dtype=np.int32)
    >>> 
    >>> # Compute group means
    >>> means = group_mean(X, group_id, n_groups=3)
    >>> print(means.shape)  # (3, 50)
    """
    
    # Convert to CSC if needed
    if issparse(X):
        if not isinstance(X, csc_matrix):
            X = csc_matrix(X)
    else:
        X = csc_matrix(X)
    
    # Ensure group_id is int32
    group_id = np.asarray(group_id, dtype=np.int32)
    
    # Call backend-specific implementation
    if _BACKEND_NAME == "C++":
        return _group_mean_impl(
            X.data, X.indices, X.indptr, group_id, n_groups,
            include_zeros=include_zeros, threads=threads
        )
    elif _BACKEND_NAME == "Cython":
        return _group_mean_impl(X, group_id, n_groups, include_zeros=include_zeros, threads=threads)
    else:  # Python
        return _group_mean_impl(X, group_id, n_groups, include_zeros=include_zeros)
