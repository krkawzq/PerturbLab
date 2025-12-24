"""T-test statistical functions with unified interface.

Provides Student's t-test, Welch's t-test, and log fold change computation
for differential expression analysis.
"""

from typing import Literal, Tuple

import numpy as np
import scipy.sparse
from scipy.stats import ttest_ind

from perturblab.utils import get_logger

from ._ttest_capi import (
    has_ttest_backend,
    student_ttest_cpp,
    welch_ttest_cpp,
    log_fold_change_cpp,
)

logger = get_logger(__name__)

__all__ = ["ttest", "log_fold_change"]


def ttest(
    sparse_matrix: scipy.sparse.csr_matrix | scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_targets: int,
    method: Literal["student", "welch"] = "welch",
    threads: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform t-test on sparse matrix (multi-group).
    
    Args:
        sparse_matrix: Sparse matrix in CSR or CSC format.
            For CSC: rows are samples, columns are features.
            For CSR: columns are samples, rows are features.
        group_id: 1D array of group assignments.
            - 0: reference group
            - 1 to n_targets: target groups
            - -1: ignored samples
        n_targets: Number of target groups to test.
        method: T-test variant:
            - "student": Student's t-test (assumes equal variances)
            - "welch": Welch's t-test (does not assume equal variances, recommended)
        threads: Number of threads to use (-1 for all available).
    
    Returns:
        Tuple of (t_statistic, p_value, mean_diff, log2_fc):
            - t_statistic: t statistics, shape (n_targets, n_features)
            - p_value: Two-sided p-values, shape (n_targets, n_features)
            - mean_diff: Mean difference (target - reference), shape (n_targets, n_features)
            - log2_fc: Log2 fold change, shape (n_targets, n_features)
    
    Examples:
        >>> import scipy.sparse as sp
        >>> from perturblab.kernels.statistics import ttest
        >>> 
        >>> X = sp.random(100, 50, density=0.1, format='csc')
        >>> group_id = np.array([0]*40 + [1]*30 + [2]*30, dtype=np.int32)
        >>> 
        >>> t_stat, p_val, mean_diff, log2_fc = ttest(
        ...     X, group_id, n_targets=2, method='welch', threads=8
        ... )
    """
    # Validate inputs
    if not isinstance(sparse_matrix, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        raise ValueError("sparse_matrix must be CSR or CSC format")
    
    if method not in ["student", "welch"]:
        raise ValueError(f"method must be 'student' or 'welch', got '{method}'")
    
    # Determine number of samples
    if isinstance(sparse_matrix, scipy.sparse.csc_matrix):
        n_samples = sparse_matrix.shape[0]
    else:  # CSR
        n_samples = sparse_matrix.shape[1]
    
    if group_id.ndim != 1 or len(group_id) != n_samples:
        raise ValueError(
            f"group_id must be 1D with length {n_samples} (number of samples)"
        )
    
    # Convert to CSC if needed
    if isinstance(sparse_matrix, scipy.sparse.csr_matrix):
        matrix_csc = sparse_matrix.T.tocsc()
    else:
        matrix_csc = sparse_matrix
    
    # Ensure group_id is int32
    if group_id.dtype != np.int32:
        group_id = group_id.astype(np.int32)
    
    # Try C++ backend first
    if has_ttest_backend():
        try:
            if method == "student":
                t_stat, p_val, mean_diff, log2_fc = student_ttest_cpp(
                    data=matrix_csc.data,
                    indices=matrix_csc.indices,
                    indptr=matrix_csc.indptr,
                    group_id=group_id,
                    n_targets=n_targets,
                    threads=threads,
                )
            else:  # welch
                t_stat, p_val, mean_diff, log2_fc = welch_ttest_cpp(
                    data=matrix_csc.data,
                    indices=matrix_csc.indices,
                    indptr=matrix_csc.indptr,
                    group_id=group_id,
                    n_targets=n_targets,
                    threads=threads,
                )
            return t_stat, p_val, mean_diff, log2_fc
        except Exception as e:
            logger.warning(f"C++ backend failed: {e}, falling back to scipy")
    
    # Fallback to scipy
    logger.warning("Using scipy fallback for t-test (slower)")
    return _ttest_scipy_fallback(matrix_csc, group_id, n_targets, method)


def log_fold_change(
    sparse_matrix: scipy.sparse.csr_matrix | scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_targets: int,
    pseudocount: float = 1e-9,
    threads: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute log fold change between groups.
    
    Args:
        sparse_matrix: Sparse matrix in CSR or CSC format.
        group_id: 1D array of group assignments (0=reference, 1..n_targets=targets, -1=ignore).
        n_targets: Number of target groups.
        pseudocount: Small value added to avoid log(0).
        threads: Number of threads to use (-1 for all available).
    
    Returns:
        Tuple of (mean_diff, log2_fc):
            - mean_diff: Mean difference, shape (n_targets, n_features)
            - log2_fc: Log2 fold change, shape (n_targets, n_features)
    """
    # Validate and convert
    if not isinstance(sparse_matrix, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        raise ValueError("sparse_matrix must be CSR or CSC format")
    
    if isinstance(sparse_matrix, scipy.sparse.csr_matrix):
        matrix_csc = sparse_matrix.T.tocsc()
    else:
        matrix_csc = sparse_matrix
    
    if group_id.dtype != np.int32:
        group_id = group_id.astype(np.int32)
    
    # Try C++ backend
    if has_ttest_backend():
        try:
            _, _, mean_diff, log2_fc = log_fold_change_cpp(
                data=matrix_csc.data,
                indices=matrix_csc.indices,
                indptr=matrix_csc.indptr,
                group_id=group_id,
                n_targets=n_targets,
                pseudocount=pseudocount,
                threads=threads,
            )
            return mean_diff, log2_fc
        except Exception as e:
            logger.warning(f"C++ backend failed: {e}, falling back to numpy")
    
    # Fallback to numpy
    logger.warning("Using numpy fallback for log fold change")
    return _log_fc_numpy_fallback(matrix_csc, group_id, n_targets, pseudocount)


# =============================================================================
# SciPy/NumPy Fallback Implementations
# =============================================================================

def _ttest_scipy_fallback(
    matrix_csc: scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_targets: int,
    method: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback t-test implementation using scipy."""
    n_cols = matrix_csc.shape[1]
    
    # Allocate outputs
    t_stat = np.zeros((n_targets, n_cols), dtype=np.float64)
    p_val = np.ones((n_targets, n_cols), dtype=np.float64)
    mean_diff = np.zeros((n_targets, n_cols), dtype=np.float64)
    log2_fc = np.zeros((n_targets, n_cols), dtype=np.float64)
    
    ref_mask = group_id == 0
    equal_var = (method == "student")
    
    for t in range(n_targets):
        tar_mask = group_id == (t + 1)
        
        for g in range(n_cols):
            ref_vals = matrix_csc[ref_mask, g].toarray().flatten()
            tar_vals = matrix_csc[tar_mask, g].toarray().flatten()
            
            if len(ref_vals) < 2 or len(tar_vals) < 2:
                continue
            
            try:
                stat, pval = ttest_ind(tar_vals, ref_vals, equal_var=equal_var)
                t_stat[t, g] = stat
                p_val[t, g] = pval
                
                ref_mean = np.mean(ref_vals)
                tar_mean = np.mean(tar_vals)
                mean_diff[t, g] = tar_mean - ref_mean
                log2_fc[t, g] = np.log2((tar_mean + 1e-9) / (ref_mean + 1e-9))
            except Exception:
                p_val[t, g] = 1.0
    
    return t_stat, p_val, mean_diff, log2_fc


def _log_fc_numpy_fallback(
    matrix_csc: scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_targets: int,
    pseudocount: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback log fold change using numpy."""
    n_cols = matrix_csc.shape[1]
    
    mean_diff = np.zeros((n_targets, n_cols), dtype=np.float64)
    log2_fc = np.zeros((n_targets, n_cols), dtype=np.float64)
    
    ref_mask = group_id == 0
    ref_data = matrix_csc[ref_mask, :]
    ref_means = np.asarray(ref_data.mean(axis=0)).flatten()
    
    for t in range(n_targets):
        tar_mask = group_id == (t + 1)
        tar_data = matrix_csc[tar_mask, :]
        tar_means = np.asarray(tar_data.mean(axis=0)).flatten()
        
        mean_diff[t, :] = tar_means - ref_means
        log2_fc[t, :] = np.log2((tar_means + pseudocount) / (ref_means + pseudocount))
    
    return mean_diff, log2_fc

