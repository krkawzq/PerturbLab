"""
Scaling/standardization operations with automatic backend selection.

Copyright (c) 2024 PerturbLab
Portions adapted from Scanpy (https://github.com/scverse/scanpy)
Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
Licensed under BSD 3-Clause License
"""

import numpy as np
import scipy.sparse

# Try to import backends in order of preference: C++ > Cython > Numba > Python
_has_cpp = False
_has_cython = False
_has_numba = False
_backend_name = "python"

try:
    from ..backends.cpp._scale import dense_standardize_cpp, has_cpp_backend, sparse_standardize_cpp

    _has_cpp = has_cpp_backend()
    if _has_cpp:
        _backend_name = "C++"

        def _sparse_standardize_impl(X, means, stds, zero_center, max_value, n_threads):
            return sparse_standardize_cpp(X, means, stds, zero_center, max_value, n_threads)

        def _dense_standardize_impl(X, means, stds, zero_center, max_value, n_threads):
            return dense_standardize_cpp(X, means, stds, zero_center, max_value, n_threads)

except (ImportError, RuntimeError):
    pass

if not _has_cpp:
    try:
        from ..backends.cython._scale_wrapper import (
            dense_standardize_cython,
            has_cython_backend,
            sparse_standardize_cython,
        )

        _has_cython = has_cython_backend()
        if _has_cython:
            _backend_name = "Cython"

            def _sparse_standardize_impl(X, means, stds, zero_center, max_value, n_threads):
                return sparse_standardize_cython(X, means, stds, zero_center, max_value)

            def _dense_standardize_impl(X, means, stds, zero_center, max_value, n_threads):
                return dense_standardize_cython(X, means, stds, zero_center, max_value)

    except (ImportError, RuntimeError):
        pass

if not _has_cpp and not _has_cython:
    try:
        from ..backends.python.numba._scale import (
            dense_standardize_numba,
            sparse_standardize_csc_numba,
            sparse_standardize_csr_numba,
        )

        _has_numba = True
        _backend_name = "Numba"

        def _sparse_standardize_impl(X, means, stds, zero_center, max_value, n_threads):
            if scipy.sparse.isspmatrix_csc(X):
                sparse_standardize_csc_numba(
                    X.data,
                    X.indices,
                    X.indptr,
                    X.shape[0],
                    X.shape[1],
                    means,
                    stds,
                    zero_center,
                    max_value,
                )
            elif scipy.sparse.isspmatrix_csr(X):
                sparse_standardize_csr_numba(
                    X.data,
                    X.indices,
                    X.indptr,
                    X.shape[0],
                    X.shape[1],
                    means,
                    stds,
                    zero_center,
                    max_value,
                )
            return X

        def _dense_standardize_impl(X, means, stds, zero_center, max_value, n_threads):
            dense_standardize_numba(X, X.shape[0], X.shape[1], means, stds, zero_center, max_value)
            return X

    except ImportError:
        pass

if not _has_cpp and not _has_cython and not _has_numba:
    # Pure Python fallback
    def _sparse_standardize_impl(X, means, stds, zero_center, max_value, n_threads):
        # Column-wise standardization for sparse matrix
        for j in range(X.shape[1]):
            if stds[j] <= 0 or not np.isfinite(stds[j]):
                X[:, j] = 0
                continue

            if zero_center:
                X[:, j] = (X[:, j].toarray().ravel() - means[j]) / stds[j]
            else:
                X[:, j] = X[:, j].toarray().ravel() / stds[j]

            if max_value > 0:
                X[:, j] = np.clip(
                    X[:, j].toarray().ravel(), -max_value if zero_center else 0, max_value
                )

        return X

    def _dense_standardize_impl(X, means, stds, zero_center, max_value, n_threads):
        # Column-wise standardization
        for j in range(X.shape[1]):
            if stds[j] <= 0 or not np.isfinite(stds[j]):
                X[:, j] = 0
                continue

            if zero_center:
                X[:, j] = (X[:, j] - means[j]) / stds[j]
            else:
                X[:, j] = X[:, j] / stds[j]

            if max_value > 0:
                X[:, j] = np.clip(X[:, j], -max_value if zero_center else 0, max_value)

        return X


# ================================================================
# Public API
# ================================================================


def standardize(
    X: np.ndarray | scipy.sparse.spmatrix,
    means: np.ndarray,
    stds: np.ndarray,
    zero_center: bool = True,
    max_value: float = 0.0,
    n_threads: int = 0,
) -> np.ndarray | scipy.sparse.spmatrix:
    f"""Standardize matrix by columns (z-score normalization).
    
    Performs in-place standardization: X[:, j] = (X[:, j] - mean[j]) / std[j]
    Optionally clips values to [-max_value, max_value].
    
    Args:
        X: Data matrix, shape (n_obs, n_vars). Can be dense or sparse (CSC/CSR).
        means: Column means, shape (n_vars,)
        stds: Column standard deviations, shape (n_vars,)
        zero_center: If True, subtract mean; if False, only divide by std
        max_value: Maximum absolute value for clipping (0 = no clipping)
        n_threads: Number of threads (0 = auto, only used by C++ backend)
    
    Returns:
        Standardized matrix (same object as X, modified in-place)
    
    Backend:
        Auto-selected at import time: {_backend_name}
    
    Examples:
        >>> import numpy as np
        >>> from perturblab.kernels.statistics import standardize
        >>> X = np.random.randn(100, 50)
        >>> means = X.mean(axis=0)
        >>> stds = X.std(axis=0, ddof=1)
        >>> X_scaled = standardize(X, means, stds, zero_center=True, max_value=10)
    """

    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)

    if scipy.sparse.issparse(X):
        # Ensure CSC or CSR format
        if not (scipy.sparse.isspmatrix_csc(X) or scipy.sparse.isspmatrix_csr(X)):
            X = X.tocsr()
        return _sparse_standardize_impl(X, means, stds, zero_center, max_value, n_threads)
    else:
        # Dense array
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        return _dense_standardize_impl(X, means, stds, zero_center, max_value, n_threads)
