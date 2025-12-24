"""C++ backend for statistics kernels (high performance)."""

# Differential expression tests
from ._mannwhitneyu import group_mean_cpp, has_cpp_backend, mannwhitneyu_cpp

try:
    from ._ttest import log_fold_change_cpp, ttest_cpp

    _has_ttest = True
except ImportError:
    _has_ttest = False
    ttest_cpp = None
    log_fold_change_cpp = None

# HVG operators
try:
    from ._hvg import (
        clip_matrix_cpp,
        group_mean_var_cpp,
        group_var_cpp,
        loess_fit_cpp,
        polynomial_fit_cpp,
        sparse_clipped_moments_cpp,
        sparse_mean_var_cpp,
    )

    _has_hvg = True
except ImportError:
    _has_hvg = False

# Normalization operators
try:
    from ._normalization import (
        compute_median_nonzero_cpp,
        find_highly_expressed_genes_cpp,
        inplace_divide_csr_rows_cpp,
        sparse_row_sum_csr_cpp,
        sparse_row_sum_csr_exclude_genes_cpp,
    )

    _has_normalization = True
except ImportError:
    _has_normalization = False

# Scale operators
try:
    from ._scale import (
        dense_standardize_cpp,
        sparse_standardize_cpp,
    )

    _has_scale = True
except ImportError:
    _has_scale = False

__all__ = [
    # Backend availability
    "has_cpp_backend",
    # DE tests
    "mannwhitneyu_cpp",
    "group_mean_cpp",
]

if _has_ttest:
    __all__.extend(["ttest_cpp", "log_fold_change_cpp"])

if _has_hvg:
    __all__.extend(
        [
            "sparse_clipped_moments_cpp",
            "sparse_mean_var_cpp",
            "clip_matrix_cpp",
            "polynomial_fit_cpp",
            "loess_fit_cpp",
            "group_var_cpp",
            "group_mean_var_cpp",
        ]
    )

if _has_normalization:
    __all__.extend(
        [
            "sparse_row_sum_csr_cpp",
            "inplace_divide_csr_rows_cpp",
            "compute_median_nonzero_cpp",
            "find_highly_expressed_genes_cpp",
            "sparse_row_sum_csr_exclude_genes_cpp",
        ]
    )

if _has_scale:
    __all__.extend(
        [
            "sparse_standardize_cpp",
            "dense_standardize_cpp",
        ]
    )
