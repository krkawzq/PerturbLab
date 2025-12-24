"""NumPy/SciPy backend for statistics kernels (pure Python fallback)."""

# Differential expression tests
# HVG operators
from ._hvg import (
    clip_matrix_py,
    group_mean_var_py,
    group_var_py,
    loess_fit_py,
    polynomial_fit_py,
    sparse_clipped_moments_py,
    sparse_mean_var_py,
)
from ._mannwhitneyu import group_mean_numpy, mannwhitneyu_scipy
from ._ttest import log_fold_change_numpy, ttest_numpy

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
