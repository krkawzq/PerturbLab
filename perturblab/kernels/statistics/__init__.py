"""Statistical computing kernels for single-cell analysis.

This package provides high-performance statistical algorithms optimized for
single-cell data, with C++ and Cython implementations and automatic fallback.

Available Functions:
    - mannwhitneyu: Mann-Whitney U test (Wilcoxon rank-sum test)
    - group_mean: Group-wise mean computation
    - ttest: Student's t-test and Welch's t-test
    - log_fold_change: Log fold change computation

All functions support:
    - Sparse matrix formats (CSC/CSR)
    - Multi-threading with OpenMP
    - Automatic backend selection (C++ → Cython → scipy/numpy)

Examples:
    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> from perturblab.kernels.statistics import mannwhitneyu, ttest, group_mean
    >>> 
    >>> # Create example sparse data
    >>> X = sp.random(100, 50, density=0.1, format='csc')
    >>> group_id = np.array([0]*40 + [1]*30 + [2]*30, dtype=np.int32)
    >>> 
    >>> # Mann-Whitney U test
    >>> U1, U2, P = mannwhitneyu(X, group_id, n_targets=2, threads=8)
    >>> 
    >>> # Welch's t-test
    >>> t_stat, p_val, mean_diff, log2_fc = ttest(
    ...     X, group_id, n_targets=2, method='welch', threads=8
    ... )
    >>> 
    >>> # Group means
    >>> means = group_mean(X, group_id, n_groups=3)
"""

from ._mannwhitneyu import mannwhitneyu, group_mean
from ._ttest import ttest, log_fold_change

__all__ = [
    "mannwhitneyu",
    "group_mean",
    "ttest",
    "log_fold_change",
]
