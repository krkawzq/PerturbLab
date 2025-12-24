"""Numba JIT implementation of HVG operators (optional dependency).

This module provides Numba-accelerated implementations as an optional fallback.
Numba is lazily imported to avoid making it a required dependency.

Adapted from Scanpy:
https://github.com/scverse/scanpy/blob/master/src/scanpy/preprocessing/_highly_variable_genes.py
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
Licensed under BSD-3-Clause

Original implementation by Scanpy authors.
Modifications by PerturbLab team.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import scipy.sparse

logger = logging.getLogger(__name__)

__all__ = [
    "has_numba_backend",
    "sparse_clipped_moments_numba",
    "sparse_mean_var_numba",
    "group_var_numba",
    "group_mean_var_numba",
]

# Lazy import of numba
_numba = None
_has_numba = False


def _import_numba():
    """Lazily import numba (not a required dependency)."""
    global _numba, _has_numba

    if _numba is not None:
        return _numba is not False

    try:
        import numba

        _numba = numba
        _has_numba = True
        logger.info("Numba backend loaded successfully for HVG operators")
        return True
    except ImportError:
        logger.debug("Numba not available, using other fallback")
        _numba = False
        _has_numba = False
        return False


def has_numba_backend() -> bool:
    """Check if Numba backend is available."""
    return _import_numba()


# =============================================================================
# Numba JIT Functions (from Scanpy)
# =============================================================================


def _get_sparse_clipped_moments_numba():
    """Get Numba JIT-compiled sparse clipped moments function.

    Adapted from Scanpy's _sum_and_sum_squares_clipped.
    """
    if not _import_numba():
        return None

    # Note: parallel=False for accuracy (from Scanpy)
    @_numba.njit(cache=True, parallel=False)
    def _sum_and_sum_squares_clipped(
        indices: np.ndarray,
        data: np.ndarray,
        indptr: np.ndarray,
        clip_vals: np.ndarray,
    ):
        """Compute clipped sums and sum of squares for CSC matrix.

        From Scanpy: https://github.com/scverse/scanpy
        Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
        """
        n_genes = len(indptr) - 1
        sum_counts = np.zeros(n_genes, dtype=data.dtype)
        sum_sq_counts = np.zeros(n_genes, dtype=data.dtype)

        for i in range(n_genes):
            start = indptr[i]
            end = indptr[i + 1]
            clip_val = clip_vals[i]

            for j in range(start, end):
                val = min(data[j], clip_val)
                sum_counts[i] += val
                sum_sq_counts[i] += val * val

        return sum_counts, sum_sq_counts

    return _sum_and_sum_squares_clipped


def _get_sparse_mean_var_numba():
    """Get Numba JIT-compiled sparse mean/var function."""
    if not _import_numba():
        return None

    @_numba.njit(cache=True, parallel=True)
    def _sparse_mean_var(
        data: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        n_rows: int,
        include_zeros: bool,
    ):
        """Compute mean and variance for CSC matrix columns."""
        n_cols = len(indptr) - 1
        means = np.zeros(n_cols, dtype=np.float64)
        vars = np.zeros(n_cols, dtype=np.float64)

        for col in _numba.prange(n_cols):
            start = indptr[col]
            end = indptr[col + 1]
            nnz = end - start

            if nnz == 0:
                continue

            # Compute mean
            col_sum = 0.0
            for i in range(start, end):
                col_sum += data[i]

            if include_zeros:
                mean = col_sum / n_rows
            else:
                mean = col_sum / nnz if nnz > 0 else 0.0

            means[col] = mean

            # Compute variance
            sq_sum = 0.0
            for i in range(start, end):
                diff = data[i] - mean
                sq_sum += diff * diff

            if include_zeros and nnz < n_rows:
                # Add implicit zeros contribution
                sq_sum += (n_rows - nnz) * mean * mean
                n = n_rows
            else:
                n = nnz

            if n > 1:
                vars[col] = sq_sum / n

        return means, vars

    return _sparse_mean_var


def _get_group_var_numba():
    """Get Numba JIT-compiled group variance function."""
    if not _import_numba():
        return None

    @_numba.njit(cache=True, parallel=True)
    def _group_var(
        data: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        n_rows: int,
        group_id: np.ndarray,
        n_groups: int,
        include_zeros: bool,
    ):
        """Compute group-wise variance."""
        n_cols = len(indptr) - 1
        result = np.zeros((n_cols, n_groups), dtype=np.float64)

        # Precompute group sizes
        group_sizes = np.zeros(n_groups, dtype=np.int64)
        for row in range(n_rows):
            group_sizes[group_id[row]] += 1

        for col in _numba.prange(n_cols):
            start = indptr[col]
            end = indptr[col + 1]

            # Compute group means first
            group_sums = np.zeros(n_groups, dtype=np.float64)
            group_counts = np.zeros(n_groups, dtype=np.int64)

            for i in range(start, end):
                row = indices[i]
                gid = group_id[row]
                group_sums[gid] += data[i]
                if not include_zeros:
                    group_counts[gid] += 1

            # Compute means
            group_means = np.zeros(n_groups, dtype=np.float64)
            for g in range(n_groups):
                if include_zeros:
                    n = group_sizes[g]
                else:
                    n = group_counts[g]

                if n > 0:
                    group_means[g] = group_sums[g] / n

            # Compute variances
            group_sq_sums = np.zeros(n_groups, dtype=np.float64)

            for i in range(start, end):
                row = indices[i]
                gid = group_id[row]
                diff = data[i] - group_means[gid]
                group_sq_sums[gid] += diff * diff

            # Add zeros contribution if needed
            if include_zeros:
                for row in range(n_rows):
                    gid = group_id[row]
                    # Check if this row is zero (not in sparse data)
                    is_zero = True
                    for i in range(start, end):
                        if indices[i] == row:
                            is_zero = False
                            break

                    if is_zero:
                        diff = 0.0 - group_means[gid]
                        group_sq_sums[gid] += diff * diff

            # Final variances
            for g in range(n_groups):
                if include_zeros:
                    n = group_sizes[g]
                else:
                    n = group_counts[g]

                if n > 1:
                    result[col, g] = group_sq_sums[g] / n

        return result.ravel()

    return _group_var


# =============================================================================
# Public API (Wrapper Functions)
# =============================================================================


def sparse_clipped_moments_numba(
    X: scipy.sparse.csc_matrix,
    clip_vals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute clipped moments (Numba backend).

    Adapted from Scanpy.
    """
    func = _get_sparse_clipped_moments_numba()
    if func is None:
        raise RuntimeError("Numba not available")

    if not scipy.sparse.isspmatrix_csc(X):
        X = X.tocsc()

    X = X.astype(np.float64, copy=False)
    clip_vals = np.ascontiguousarray(clip_vals, dtype=np.float64)

    return func(X.indices, X.data, X.indptr, clip_vals)


def sparse_mean_var_numba(
    X: scipy.sparse.csc_matrix,
    include_zeros: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and variance (Numba backend)."""
    func = _get_sparse_mean_var_numba()
    if func is None:
        raise RuntimeError("Numba not available")

    if not scipy.sparse.isspmatrix_csc(X):
        X = X.tocsc()

    X = X.astype(np.float64, copy=False)

    return func(X.data, X.indices, X.indptr, X.shape[0], include_zeros)


def group_var_numba(
    X: scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_groups: int,
    include_zeros: bool = True,
) -> np.ndarray:
    """Compute group-wise variance (Numba backend)."""
    func = _get_group_var_numba()
    if func is None:
        raise RuntimeError("Numba not available")

    if not scipy.sparse.isspmatrix_csc(X):
        X = X.tocsc()

    X = X.astype(np.float64, copy=False)
    group_id = np.ascontiguousarray(group_id, dtype=np.int32)

    return func(X.data, X.indices, X.indptr, X.shape[0], group_id, n_groups, include_zeros)


def group_mean_var_numba(
    X: scipy.sparse.csc_matrix,
    group_id: np.ndarray,
    n_groups: int,
    include_zeros: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute group-wise mean and variance (Numba backend).

    Note: This is a placeholder. For now, compute mean and var separately.
    """
    # Could implement a combined version for efficiency
    raise NotImplementedError("Use separate group_mean and group_var functions")
