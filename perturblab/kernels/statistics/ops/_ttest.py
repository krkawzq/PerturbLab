"""T-test and log fold change operations.

This module provides Student's t-test, Welch's t-test, and log fold change
computation with automatic backend selection performed at import time.

Backend priority: C++ > Python (NumPy/SciPy)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.sparse import csc_matrix, issparse

from perturblab.utils import get_logger

if TYPE_CHECKING:
    from scipy.sparse import spmatrix

logger = get_logger()

__all__ = ["ttest", "log_fold_change"]


# =============================================================================
# Backend Selection (performed at import time)
# =============================================================================

_BACKEND_NAME = None
_student_ttest_impl = None
_welch_ttest_impl = None
_log_fold_change_impl = None


def _select_backend():
    """Select backend at import time."""
    global _BACKEND_NAME, _student_ttest_impl, _welch_ttest_impl, _log_fold_change_impl
    
    # Try C++ backend first
    try:
        from ..backends.cpp._ttest import has_ttest_backend
        if has_ttest_backend():
            from ..backends.cpp._ttest import (
                student_ttest_cpp,
                welch_ttest_cpp,
                log_fold_change_cpp,
            )
            _BACKEND_NAME = "C++"
            _student_ttest_impl = student_ttest_cpp
            _welch_ttest_impl = welch_ttest_cpp
            _log_fold_change_impl = log_fold_change_cpp
            logger.debug("T-test backend: C++ (highest performance)")
            return
    except (ImportError, OSError):
        pass
    
    # Fall back to Python (NumPy/SciPy)
    from ..backends.python._ttest import ttest_numpy, log_fold_change_numpy
    _BACKEND_NAME = "Python"
    
    # Wrapper to split ttest_numpy into student and welch
    def _student_wrapper(data, indices, indptr, group_id, n_targets, threads=-1):
        X = csc_matrix((data, indices, indptr))
        return ttest_numpy(X, group_id, n_targets, method="student")
    
    def _welch_wrapper(data, indices, indptr, group_id, n_targets, threads=-1):
        X = csc_matrix((data, indices, indptr))
        return ttest_numpy(X, group_id, n_targets, method="welch")
    
    def _logfc_wrapper(data, indices, indptr, group_id, n_targets, threads=-1, pseudocount=1e-9):
        X = csc_matrix((data, indices, indptr))
        return log_fold_change_numpy(X, group_id, n_targets, pseudocount=pseudocount)
    
    _student_ttest_impl = _student_wrapper
    _welch_ttest_impl = _welch_wrapper
    _log_fold_change_impl = _logfc_wrapper
    logger.warning("T-test backend: Python (NumPy/SciPy) - slower, consider installing compiled backends")


# Select backend on module import
_select_backend()


# =============================================================================
# Wrapper Functions (with input preprocessing)
# =============================================================================

def ttest(
    X: np.ndarray | spmatrix,
    group_id: np.ndarray,
    n_targets: int,
    *,
    method: Literal["student", "welch"] = "welch",
    threads: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Student's t-test or Welch's t-test for differential expression.
    
    Performs parametric t-tests for multiple target groups against a
    reference group.
    
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
    method : {'student', 'welch'}, default='welch'
        Test method:
        - 'student': Student's t-test (assumes equal variances)
        - 'welch': Welch's t-test (does not assume equal variances)
    threads : int, default=-1
        Number of threads to use. -1 means use all available threads.
        Only used by C++ backend.
    
    Returns
    -------
    t_stat : np.ndarray
        T-statistics, shape (n_targets, n_genes).
    p_values : np.ndarray
        Two-sided p-values, shape (n_targets, n_genes).
    mean_diff : np.ndarray
        Mean differences (target - reference), shape (n_targets, n_genes).
    log2_fc : np.ndarray
        Log2 fold changes, shape (n_targets, n_genes).
    
    Notes
    -----
    - Welch's t-test is more robust when variances are unequal
    - Student's t-test has slightly more power when variances are equal
    - Log2 fold change is computed as log2(target_mean / reference_mean)
    
    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> from perturblab.kernels.statistics import ttest
    >>> 
    >>> # Create example data
    >>> X = sp.random(100, 50, density=0.1, format='csc')
    >>> group_id = np.array([0]*40 + [1]*30 + [2]*30, dtype=np.int32)
    >>> 
    >>> # Welch's t-test
    >>> t_stat, p_val, mean_diff, log2_fc = ttest(
    ...     X, group_id, n_targets=2, method='welch'
    ... )
    >>> 
    >>> # Find significant genes (p < 0.05, |log2_fc| > 1)
    >>> significant = (p_val < 0.05) & (np.abs(log2_fc) > 1)
    """
    
    # Validate method
    if method not in ("student", "welch"):
        raise ValueError(f"method must be 'student' or 'welch', got {method!r}")
    
    # Convert to CSC if needed
    if issparse(X):
        if not isinstance(X, csc_matrix):
            X = csc_matrix(X)
    else:
        X = csc_matrix(X)
    
    # Ensure group_id is int32
    group_id = np.asarray(group_id, dtype=np.int32)
    
    # Call backend-specific implementation
    if method == "student":
        return _student_ttest_impl(
            X.data, X.indices, X.indptr, group_id, n_targets, threads=threads
        )
    else:  # welch
        return _welch_ttest_impl(
            X.data, X.indices, X.indptr, group_id, n_targets, threads=threads
        )


def log_fold_change(
    X: np.ndarray | spmatrix,
    group_id: np.ndarray,
    n_targets: int,
    *,
    threads: int = -1,
    pseudocount: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute log fold change between groups.
    
    Calculates mean differences and log2 fold changes for multiple
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
        Only used by C++ backend.
    pseudocount : float, default=1e-9
        Small value added to avoid log(0).
    
    Returns
    -------
    mean_diff : np.ndarray
        Mean differences (target - reference), shape (n_targets, n_genes).
    log2_fc : np.ndarray
        Log2 fold changes, shape (n_targets, n_genes).
    
    Notes
    -----
    - Log2 fold change: log2((target_mean + pseudocount) / (ref_mean + pseudocount))
    - Positive values indicate higher expression in target group
    - Negative values indicate higher expression in reference group
    
    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> from perturblab.kernels.statistics import log_fold_change
    >>> 
    >>> # Create example data
    >>> X = sp.random(100, 50, density=0.1, format='csc')
    >>> group_id = np.array([0]*40 + [1]*30 + [2]*30, dtype=np.int32)
    >>> 
    >>> # Compute log fold changes
    >>> mean_diff, log2_fc = log_fold_change(X, group_id, n_targets=2)
    >>> 
    >>> # Find strongly upregulated genes (log2_fc > 2)
    >>> upregulated = log2_fc > 2
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
    return _log_fold_change_impl(
        X.data, X.indices, X.indptr, group_id, n_targets,
        threads=threads, pseudocount=pseudocount
    )
