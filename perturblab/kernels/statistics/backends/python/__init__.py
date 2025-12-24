"""NumPy/SciPy backend for statistics kernels (pure Python fallback)."""

# Differential expression tests
from ._mannwhitneyu import mannwhitneyu_scipy, group_mean_numpy
from ._ttest import ttest_numpy, log_fold_change_numpy

# HVG operators
from ._hvg import (
    sparse_clipped_moments_py,
    sparse_mean_var_py,
    clip_matrix_py,
    polynomial_fit_py,
    loess_fit_py,
    group_var_py,
    group_mean_var_py,
)

__all__ = [
    # DE tests
    "mannwhitneyu_scipy",
    "ttest_numpy",
    "log_fold_change_numpy",
    "group_mean_numpy",
    # HVG operators
    "sparse_clipped_moments_py",
    "sparse_mean_var_py",
    "clip_matrix_py",
    "polynomial_fit_py",
    "loess_fit_py",
    "group_var_py",
    "group_mean_var_py",
]
