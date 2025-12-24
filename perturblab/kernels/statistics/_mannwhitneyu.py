"""Core statistical functions with unified interface.

This module provides a unified interface to statistical functions, automatically
selecting the best available backend:
1. C++ (highest performance) - if compiled library is available
2. Cython (fallback) - pure Python extension, good performance
3. SciPy (pure Python fallback) - always available but slower

The interface is transparent - users don't need to know which backend is used.
"""

from typing import Literal, Tuple

import numpy as np
import scipy.sparse
from scipy.stats import mannwhitneyu as scipy_mannwhitneyu

from perturblab.utils import get_logger

from ._mannwhitneyu_capi import has_cpp_backend, mannwhitneyu_cpp, group_mean_cpp

# Try to import Cython backend
try:
    from ._mannwhitneyu_kernel import mannwhitneyu_csc, group_mean_csc
    _has_cython = True
except ImportError:
    _has_cython = False
    mannwhitneyu_csc = None
    group_mean_csc = None

logger = get_logger(__name__)

__all__ = ["mannwhitneyu", "group_mean"]


# =============================================================================
# SciPy Fallback Implementation
# =============================================================================

def _mannwhitneyu_scipy_fallback(
    matrix_csc: scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_targets: int,
    use_continuity: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback implementation using scipy.stats.mannwhitneyu.
    
    This is slower but always available as it uses pure Python/numpy.
    """
    n_cols = matrix_csc.shape[1]
    
    # Allocate output arrays
    U1 = np.zeros((n_targets, n_cols), dtype=np.float64)
    U2 = np.zeros((n_targets, n_cols), dtype=np.float64)
    P = np.ones((n_targets, n_cols), dtype=np.float64)
    
    # Get reference group mask
    ref_mask = group_id == 0
    
    # Process each target group
    for t in range(n_targets):
        tar_mask = group_id == (t + 1)
        
        # Process each gene
        for g in range(n_cols):
            # Extract values
            ref_vals = matrix_csc[ref_mask, g].toarray().flatten()
            tar_vals = matrix_csc[tar_mask, g].toarray().flatten()
            
            if len(ref_vals) == 0 or len(tar_vals) == 0:
                continue
            
            # Call scipy (note: scipy returns statistic for the first group)
            # scipy's U is for the first argument (tar_vals)
            try:
                stat, pval = scipy_mannwhitneyu(
                    tar_vals, ref_vals,
                    alternative='two-sided',
                    use_continuity=use_continuity
                )
                
                # scipy returns U for the first group (target in our case)
                # This is U1 in our convention
                U1[t, g] = stat
                # U2 = n_ref * n_tar - U1
                U2[t, g] = len(ref_vals) * len(tar_vals) - stat
                P[t, g] = pval
            except Exception:
                # Handle edge cases (e.g., all values identical)
                P[t, g] = 1.0
    
    return U1, U2, P


def _group_mean_scipy_fallback(
    matrix_csc: scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_groups: int,
    include_zeros: bool = True,
) -> np.ndarray:
    """Fallback implementation for group mean using numpy."""
    n_cols = matrix_csc.shape[1]
    means = np.zeros((n_groups, n_cols), dtype=np.float64)
    
    for g in range(n_groups):
        mask = group_id == g
        if np.sum(mask) == 0:
            continue
        
        # Extract group data
        group_data = matrix_csc[mask, :]
        
        # Compute mean
        if include_zeros:
            # Include implicit zeros in the mean
            means[g, :] = np.asarray(group_data.mean(axis=0)).flatten()
        else:
            # Only average non-zero values
            for col in range(n_cols):
                col_data = group_data[:, col].data
                if len(col_data) > 0:
                    means[g, col] = np.mean(col_data)
    
    return means

# Log backend availability
if has_cpp_backend():
    logger.info("Using C++ backend for statistics kernels")
elif _has_cython:
    logger.info("C++ backend not available, using Cython fallback")
else:
    logger.warning("No compiled backend available! Using scipy fallback (slower).")


def mannwhitneyu(
    sparse_matrix: scipy.sparse.csr_matrix | scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_targets: int,
    tie_correction: bool = True,
    use_continuity: bool = False,
    zero_handling: Literal["none", "min", "max", "mix"] = "min",
    threads: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mann-Whitney U test on sparse matrix (multi-group).
    
    Performs Mann-Whitney U test comparing a reference group (group_id=0)
    against multiple target groups (group_id=1, 2, ..., n_targets).
    
    Args:
        sparse_matrix: Sparse matrix in CSR or CSC format.
            For CSC: rows are samples, columns are features.
            For CSR: columns are samples, rows are features.
        group_id: 1D array of group assignments.
            - 0: reference group
            - 1 to n_targets: target groups
            - -1: ignored samples
            Length must match number of samples.
        n_targets: Number of target groups to test.
        tie_correction: Whether to apply tie correction (default: True).
        use_continuity: Whether to apply continuity correction (default: False).
        zero_handling: How to handle zeros in sparse matrix (currently ignored,
            zeros are always at minimum rank as in sparse representation).
        threads: Number of threads to use (-1 for all available).
    
    Returns:
        Tuple of (U1, U2, P) where each is a 2D array of shape (n_targets, n_features):
            - U1: U statistic (sum of ranks in target group)
            - U2: Complementary U statistic (n_ref * n_tar - U1)
            - P: Two-sided p-values
    
    Examples:
        >>> import numpy as np
        >>> import scipy.sparse as sp
        >>> from perturblab.kernels.statistics import mannwhitneyu
        >>> 
        >>> # Create example data: 100 cells x 50 genes
        >>> X = sp.random(100, 50, density=0.1, format='csc')
        >>> group_id = np.array([0]*40 + [1]*30 + [2]*30, dtype=np.int32)
        >>> 
        >>> # Test 2 target groups vs reference
        >>> U1, U2, P = mannwhitneyu(X, group_id, n_targets=2)
        >>> print(U1.shape, P.shape)  # (2, 50), (2, 50)
    """
    # Validate inputs
    if not isinstance(sparse_matrix, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        raise ValueError("sparse_matrix must be CSR or CSC format")
    
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
        # For CSR, we need to transpose to get samples as rows
        matrix_csc = sparse_matrix.T.tocsc()
    else:
        matrix_csc = sparse_matrix
    
    # Ensure group_id is int32
    if group_id.dtype != np.int32:
        group_id = group_id.astype(np.int32)
    
    # Try C++ backend first, fallback to Cython
    if has_cpp_backend():
        try:
            U1, U2, P = mannwhitneyu_cpp(
                data=matrix_csc.data,
                indices=matrix_csc.indices,
                indptr=matrix_csc.indptr,
                group_id=group_id,
                n_targets=n_targets,
                tie_correction=tie_correction,
                use_continuity=use_continuity,
                threads=threads,
            )
            return U1, U2, P
        except Exception as e:
            logger.warning(f"C++ backend failed: {e}, falling back to Cython")
    
    # Fallback to Cython implementation
    if _has_cython:
        U1, U2, P = mannwhitneyu_csc(
            data=matrix_csc.data,
            indices=matrix_csc.indices,
            indptr=matrix_csc.indptr,
            group_id=group_id,
            n_targets=n_targets,
            tie_correction=tie_correction,
            use_continuity=use_continuity,
            threads=threads,
        )
        return U1, U2, P
    
    # Final fallback to scipy (pure Python)
    logger.warning(
        "No compiled backend available, using scipy fallback (significantly slower). "
        "Consider compiling C++ backend for better performance: "
        "cd perturblab/kernels/statistics/_mannwhitneyu && ./setup_deps.sh && ./build.sh"
    )
    U1, U2, P = _mannwhitneyu_scipy_fallback(
        matrix_csc, group_id, n_targets, use_continuity
    )
    return U1, U2, P


def group_mean(
    sparse_matrix: scipy.sparse.csr_matrix | scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_groups: int,
    include_zeros: bool = True,
    threads: int = -1,
) -> np.ndarray:
    """Compute group-wise mean for each feature.
    
    Efficiently computes mean expression for each group across all features.
    
    Args:
        sparse_matrix: Sparse matrix in CSR or CSC format.
            For CSC: rows are samples, columns are features.
            For CSR: columns are samples, rows are features.
        group_id: 1D array of group assignments (0 to n_groups-1).
            Length must match number of samples.
            Use -1 for samples to ignore.
        n_groups: Total number of groups.
        include_zeros: Whether to include implicit zeros in mean calculation.
            If True (default), zeros contribute to the mean.
            If False, only non-zero values are averaged.
        threads: Number of threads to use (-1 for all available).
    
    Returns:
        2D array of shape (n_groups, n_features) containing group means.
    
    Examples:
        >>> import numpy as np
        >>> import scipy.sparse as sp
        >>> from perturblab.kernels.statistics import group_mean
        >>> 
        >>> # Create example data
        >>> X = sp.random(100, 50, density=0.1, format='csc')
        >>> group_id = np.array([0]*40 + [1]*30 + [2]*30, dtype=np.int32)
        >>> 
        >>> # Compute means for 3 groups
        >>> means = group_mean(X, group_id, n_groups=3)
        >>> print(means.shape)  # (3, 50)
    """
    # Validate inputs
    if not isinstance(sparse_matrix, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        raise ValueError("sparse_matrix must be CSR or CSC format")
    
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
    
    # Try C++ backend first, fallback to Cython
    if has_cpp_backend():
        try:
            means = group_mean_cpp(
                data=matrix_csc.data,
                indices=matrix_csc.indices,
                indptr=matrix_csc.indptr,
                group_id=group_id,
                n_groups=n_groups,
                include_zeros=include_zeros,
                threads=threads,
            )
            return means
        except Exception as e:
            logger.warning(f"C++ backend failed: {e}, falling back to Cython")
    
    # Fallback to Cython implementation
    if _has_cython:
        means = group_mean_csc(
            data=matrix_csc.data,
            indices=matrix_csc.indices,
            indptr=matrix_csc.indptr,
            group_id=group_id,
            n_groups=n_groups,
            include_zeros=include_zeros,
            threads=threads,
        )
        return means
    
    # Final fallback to scipy/numpy (pure Python)
    logger.warning(
        "No compiled backend available, using numpy fallback (slower). "
        "Consider compiling C++ backend for better performance."
    )
    means = _group_mean_scipy_fallback(
        matrix_csc, group_id, n_groups, include_zeros
    )
    return means

