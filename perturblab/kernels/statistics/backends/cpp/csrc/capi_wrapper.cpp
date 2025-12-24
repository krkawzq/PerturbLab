// capi_wrapper.cpp
// C API wrapper for statistical test functions
//
// This file provides a C API interface to the C++ statistical kernels,
// enabling Python integration via ctypes without pybind11 dependency.
//
// The wrapper handles:
// - Memory management (allocation and deallocation)
// - Type conversions (Python arrays -> C++ structures)
// - Error handling (returns nullptr on failure)
// - Multi-dtype support (float32, float64)
//
#include "mannwhitneyu.hpp"
#include "ttest.hpp"
#include "sparse.hpp"
#include "group_ops.hpp"
#include "sparse_clipped_moments.hpp"
#include "sparse_mean_var.hpp"
#include "clip_matrix.hpp"
#include "polynomial_fit.hpp"
#include "normalization.hpp"
#include "scale.hpp"
#include <cstring>
#include <cstdlib>

using namespace perturblab::kernel;

extern "C" {

struct MWUResults {
    double* U1;
    double* U2;
    double* P;
    size_t size;
};

struct GroupMeanResults {
    double* means;
    size_t size;
};

void mwu_free_results(MWUResults* results) {
    if (results) {
        delete[] results->U1;
        delete[] results->U2;
        delete[] results->P;
        delete results;
    }
}

void group_mean_free_results(GroupMeanResults* results) {
    if (results) {
        delete[] results->means;
        delete results;
    }
}

MWUResults* mannwhitneyu_csc_capi(
    const double* data,
    const int64_t* indices,
    const int64_t* indptr,
    const int32_t* group_id,
    size_t n_rows,
    size_t n_cols,
    size_t nnz,
    size_t n_targets,
    bool tie_correction,
    bool use_continuity,
    int threads
) {
    try {
        auto V = view::CscView<double>::from_raw(data, indices, indptr, n_rows, n_cols, nnz);
        MannWhitneyuOption opt{};
        opt.ref_sorted = false; opt.tar_sorted = false;
        opt.tie_correction = tie_correction; opt.use_continuity = use_continuity;
        opt.fast_norm = true; opt.zero_handling = MannWhitneyuOption::min;
        opt.alternative = MannWhitneyuOption::two_sided; opt.method = MannWhitneyuOption::asymptotic;
        auto result = mannwhitneyu<double>(V, group_id, n_targets, opt, threads, nullptr);
        size_t result_size = result.U1.size();
        auto* ret = new MWUResults;
        ret->U1 = new double[result_size]; ret->U2 = new double[result_size]; ret->P = new double[result_size];
        ret->size = result_size;
        std::memcpy(ret->U1, result.U1.data(), result_size * sizeof(double));
        std::memcpy(ret->U2, result.U2.data(), result_size * sizeof(double));
        std::memcpy(ret->P, result.P.data(), result_size * sizeof(double));
        return ret;
    } catch (...) { return nullptr; }
}

GroupMeanResults* group_mean_csc_capi(
    const double* data,
    const int64_t* indices,
    const int64_t* indptr,
    const int32_t* group_id,
    size_t n_rows,
    size_t n_cols,
    size_t nnz,
    size_t n_groups,
    bool include_zeros,
    int threads
) {
    try {
        auto V = view::CscView<double>::from_raw(data, indices, indptr, n_rows, n_cols, nnz);
        auto result = group_mean<double>(V, group_id, n_groups, include_zeros, threads, false);
        size_t result_size = result.size();
        auto* ret = new GroupMeanResults;
        ret->means = new double[result_size];
        ret->size = result_size;
        std::memcpy(ret->means, result.data(), result_size * sizeof(double));
        return ret;
    } catch (...) { return nullptr; }
}

MWUResults* mannwhitneyu_csc_f32_capi(
    const float* data,
    const int64_t* indices,
    const int64_t* indptr,
    const int32_t* group_id,
    size_t n_rows,
    size_t n_cols,
    size_t nnz,
    size_t n_targets,
    bool tie_correction,
    bool use_continuity,
    int threads
) {
    try {
        auto V = view::CscView<float>::from_raw(data, indices, indptr, n_rows, n_cols, nnz);
        MannWhitneyuOption opt{};
        opt.ref_sorted = false; opt.tar_sorted = false;
        opt.tie_correction = tie_correction; opt.use_continuity = use_continuity;
        opt.fast_norm = true; opt.zero_handling = MannWhitneyuOption::min;
        opt.alternative = MannWhitneyuOption::two_sided; opt.method = MannWhitneyuOption::asymptotic;
        auto result = mannwhitneyu<float>(V, group_id, n_targets, opt, threads, nullptr);
        size_t result_size = result.U1.size();
        auto* ret = new MWUResults;
        ret->U1 = new double[result_size]; ret->U2 = new double[result_size]; ret->P = new double[result_size];
        ret->size = result_size;
        std::memcpy(ret->U1, result.U1.data(), result_size * sizeof(double));
        std::memcpy(ret->U2, result.U2.data(), result_size * sizeof(double));
        std::memcpy(ret->P, result.P.data(), result_size * sizeof(double));
        return ret;
    } catch (...) { return nullptr; }
}

GroupMeanResults* group_mean_csc_f32_capi(
    const float* data,
    const int64_t* indices,
    const int64_t* indptr,
    const int32_t* group_id,
    size_t n_rows,
    size_t n_cols,
    size_t nnz,
    size_t n_groups,
    bool include_zeros,
    int threads
) {
    try {
        auto V = view::CscView<float>::from_raw(data, indices, indptr, n_rows, n_cols, nnz);
        auto result = group_mean<float>(V, group_id, n_groups, include_zeros, threads, false);
        size_t result_size = result.size();
        auto* ret = new GroupMeanResults;
        ret->means = new double[result_size];
        ret->size = result_size;
        std::memcpy(ret->means, result.data(), result_size * sizeof(double));
        return ret;
    } catch (...) { return nullptr; }
}

struct TTestResults {
    double* t_statistic;
    double* p_value;
    double* mean_diff;
    double* log2_fc;
    size_t size;
};

void ttest_free_results(TTestResults* results) {
    if (results) {
        delete[] results->t_statistic;
        delete[] results->p_value;
        delete[] results->mean_diff;
        delete[] results->log2_fc;
        delete results;
    }
}

TTestResults* student_ttest_csc_capi(
    const double* data,
    const int64_t* indices,
    const int64_t* indptr,
    const int32_t* group_id,
    size_t n_rows,
    size_t n_cols,
    size_t nnz,
    size_t n_targets,
    int threads
) {
    try {
        auto V = view::CscView<double>::from_raw(data, indices, indptr, n_rows, n_cols, nnz);
        auto result = student_ttest<double>(V, group_id, n_targets, threads, nullptr);
        size_t result_size = result.t_statistic.size();
        auto* ret = new TTestResults;
        ret->t_statistic = new double[result_size]; ret->p_value = new double[result_size];
        ret->mean_diff = new double[result_size]; ret->log2_fc = new double[result_size];
        ret->size = result_size;
        std::memcpy(ret->t_statistic, result.t_statistic.data(), result_size * sizeof(double));
        std::memcpy(ret->p_value, result.p_value.data(), result_size * sizeof(double));
        std::memcpy(ret->mean_diff, result.mean_diff.data(), result_size * sizeof(double));
        std::memcpy(ret->log2_fc, result.log2_fc.data(), result_size * sizeof(double));
        return ret;
    } catch (...) { return nullptr; }
}

TTestResults* welch_ttest_csc_capi(
    const double* data,
    const int64_t* indices,
    const int64_t* indptr,
    const int32_t* group_id,
    size_t n_rows,
    size_t n_cols,
    size_t nnz,
    size_t n_targets,
    int threads
) {
    try {
        auto V = view::CscView<double>::from_raw(data, indices, indptr, n_rows, n_cols, nnz);
        auto result = welch_ttest<double>(V, group_id, n_targets, threads, nullptr);
        size_t result_size = result.t_statistic.size();
        auto* ret = new TTestResults;
        ret->t_statistic = new double[result_size]; ret->p_value = new double[result_size];
        ret->mean_diff = new double[result_size]; ret->log2_fc = new double[result_size];
        ret->size = result_size;
        std::memcpy(ret->t_statistic, result.t_statistic.data(), result_size * sizeof(double));
        std::memcpy(ret->p_value, result.p_value.data(), result_size * sizeof(double));
        std::memcpy(ret->mean_diff, result.mean_diff.data(), result_size * sizeof(double));
        std::memcpy(ret->log2_fc, result.log2_fc.data(), result_size * sizeof(double));
        return ret;
    } catch (...) { return nullptr; }
}

TTestResults* log_fold_change_csc_capi(
    const double* data,
    const int64_t* indices,
    const int64_t* indptr,
    const int32_t* group_id,
    size_t n_rows,
    size_t n_cols,
    size_t nnz,
    size_t n_targets,
    double pseudocount,
    int threads
) {
    try {
        auto V = view::CscView<double>::from_raw(data, indices, indptr, n_rows, n_cols, nnz);
        auto result = log_fold_change<double>(V, group_id, n_targets, pseudocount, threads);
        size_t result_size = result.mean_diff.size();
        auto* ret = new TTestResults;
        ret->t_statistic = new double[result_size](); ret->p_value = new double[result_size]();
        ret->mean_diff = new double[result_size]; ret->log2_fc = new double[result_size];
        ret->size = result_size;
        std::memcpy(ret->mean_diff, result.mean_diff.data(), result_size * sizeof(double));
        std::memcpy(ret->log2_fc, result.log2_fc.data(), result_size * sizeof(double));
        return ret;
    } catch (...) { return nullptr; }
}

struct ClippedMomentsResults {
    double* sums;
    double* sum_squares;
    size_t size;
};

struct MeanVarResults {
    double* means;
    double* vars;
    size_t size;
};

struct PolynomialFitResults {
    double* fitted;
    double* coeffs;
    size_t n_points;
    size_t n_coeffs;
};

void clipped_moments_free_results(ClippedMomentsResults* results) {
    if (results) { delete[] results->sums; delete[] results->sum_squares; delete results; }
}

void mean_var_free_results(MeanVarResults* results) {
    if (results) { delete[] results->means; delete[] results->vars; delete results; }
}

void polynomial_fit_free_results(PolynomialFitResults* results) {
    if (results) { delete[] results->fitted; delete[] results->coeffs; delete results; }
}

ClippedMomentsResults* sparse_clipped_moments_csc_capi(
    const double* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_cols,
    const double* clip_vals,
    int n_threads
) {
    try {
        auto* ret = new ClippedMomentsResults;
        ret->size = n_cols; ret->sums = new double[n_cols]; ret->sum_squares = new double[n_cols];
        hvg::sparse_clipped_moments_csc(data, row_indices, col_ptr, n_cols, clip_vals, ret->sums, ret->sum_squares, n_threads);
        return ret;
    } catch (...) { return nullptr; }
}

ClippedMomentsResults* sparse_clipped_moments_csc_float_capi(
    const float* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_cols,
    const float* clip_vals,
    int n_threads
) {
    try {
        auto* ret = new ClippedMomentsResults;
        ret->size = n_cols;
        ret->sums = new double[n_cols];
        ret->sum_squares = new double[n_cols];
        
        // Convert clip_vals to double for the function call
        auto* clip_vals_d = new double[n_cols];
        for (size_t i = 0; i < n_cols; ++i) {
            clip_vals_d[i] = static_cast<double>(clip_vals[i]);
        }
        
        // sparse_clipped_moments_csc_f32 outputs to double arrays
        hvg::sparse_clipped_moments_csc_f32(
            data, row_indices, col_ptr, n_cols, clip_vals_d,
            ret->sums, ret->sum_squares, n_threads
        );
        
        delete[] clip_vals_d;
        
        return ret;
    } catch (...) {
        return nullptr;
    }
}

// ================================================================
// Sparse Mean/Variance
// ================================================================

MeanVarResults* sparse_mean_var_csc_capi(
    const double* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    bool include_zeros,
    int n_threads
) {
    try {
        auto* ret = new MeanVarResults;
        ret->size = n_cols;
        ret->means = new double[n_cols];
        ret->vars = new double[n_cols];
        
        stats::sparse_mean_var_csc(
            data, row_indices, col_ptr, n_rows, n_cols,
            ret->means, ret->vars, include_zeros, 1, n_threads
        );
        
        return ret;
    } catch (...) {
        return nullptr;
    }
}

double* sparse_mean_csc_capi(
    const double* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    bool include_zeros,
    int n_threads
) {
    try {
        auto* means = new double[n_cols];
        
        stats::sparse_mean_csc(
            data, row_indices, col_ptr, n_rows, n_cols,
            means, include_zeros, n_threads
        );
        
        return means;
    } catch (...) {
        return nullptr;
    }
}

double* sparse_var_csc_capi(
    const double* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    bool include_zeros,
    int n_threads
) {
    try {
        auto* vars = new double[n_cols];
        
        stats::sparse_var_csc(
            data, row_indices, col_ptr, n_rows, n_cols,
            vars, include_zeros, n_threads
        );
        
        return vars;
    } catch (...) {
        return nullptr;
    }
}

// ================================================================
// Clip Matrix (Dense)
// ================================================================

void clip_matrix_by_column_capi(
    double* data,
    size_t n_rows,
    size_t n_cols,
    const double* clip_vals,
    int n_threads
) {
    hvg::clip_matrix_by_column(
        data, n_rows, n_cols, clip_vals, n_threads
    );
}

void clip_matrix_by_column_float_capi(
    float* data,
    size_t n_rows,
    size_t n_cols,
    const float* clip_vals,
    int n_threads
) {
    hvg::clip_matrix_by_column_f32(
        data, n_rows, n_cols, clip_vals, n_threads
    );
}

// ================================================================
// Polynomial Fit
// ================================================================

PolynomialFitResults* polynomial_fit_capi(
    const double* x,
    const double* y,
    size_t n,
    int degree,
    bool return_coeffs
) {
    try {
        auto* ret = new PolynomialFitResults;
        ret->n_points = n;
        ret->n_coeffs = degree + 1;
        ret->fitted = new double[n];
        ret->coeffs = return_coeffs ? new double[degree + 1] : nullptr;
        
        hvg::polynomial_fit(
            x, y, n, degree, ret->fitted,
            return_coeffs ? ret->coeffs : nullptr
        );
        
        return ret;
    } catch (...) {
        return nullptr;
    }
}

PolynomialFitResults* weighted_polynomial_fit_capi(
    const double* x,
    const double* y,
    const double* weights,
    size_t n,
    int degree,
    bool return_coeffs
) {
    try {
        auto* ret = new PolynomialFitResults;
        ret->n_points = n;
        ret->n_coeffs = degree + 1;
        ret->fitted = new double[n];
        ret->coeffs = return_coeffs ? new double[degree + 1] : nullptr;
        
        hvg::weighted_polynomial_fit(
            x, y, weights, n, degree, ret->fitted,
            return_coeffs ? ret->coeffs : nullptr
        );
        
        return ret;
    } catch (...) {
        return nullptr;
    }
}

double* loess_fit_fast_capi(
    const double* x,
    const double* y,
    size_t n,
    double span,
    int n_threads
) {
    try {
        auto* fitted = new double[n];
        
        hvg::loess_fit(
            x, y, n, span, 2, fitted, n_threads
        );
        
        return fitted;
    } catch (...) {
        return nullptr;
    }
}

// ================================================================
// Group Operations (Extended)
// ================================================================

MeanVarResults* group_mean_var_csc_capi(
    const double* data,
    const int64_t* indices,
    const int64_t* indptr,
    size_t n_rows,
    size_t n_cols,
    size_t nnz,
    const int32_t* group_id,
    size_t n_groups,
    bool include_zeros,
    int threads
) {
    try {
        auto V = view::CscView<double>::from_raw(
            data, indices, indptr, n_rows, n_cols, nnz
        );
        
        size_t result_size = n_cols * n_groups;
        auto* ret = new MeanVarResults;
        ret->size = result_size;
        ret->means = new double[result_size];
        ret->vars = new double[result_size];
        
        auto result = group_mean_var<double>(
            V, group_id, n_groups, include_zeros, threads, 1
        );
        std::memcpy(ret->means, result.first.data(), result_size * sizeof(double));
        std::memcpy(ret->vars, result.second.data(), result_size * sizeof(double));
        
        return ret;
    } catch (...) {
        return nullptr;
    }
}

double* group_var_csc_capi(
    const double* data,
    const int64_t* indices,
    const int64_t* indptr,
    size_t n_rows,
    size_t n_cols,
    size_t nnz,
    const int32_t* group_id,
    size_t n_groups,
    bool include_zeros,
    int threads
) {
    try {
        auto V = view::CscView<double>::from_raw(
            data, indices, indptr, n_rows, n_cols, nnz
        );
        
        size_t result_size = n_cols * n_groups;
        auto* vars = new double[result_size];
        
        auto result = group_var<double>(
            V, group_id, n_groups, include_zeros, threads, 1
        );
        std::memcpy(vars, result.data(), result_size * sizeof(double));
        
        return vars;
    } catch (...) {
        return nullptr;
    }
}

size_t* group_count_nonzero_csc_capi(
    const double* data,
    const int64_t* indices,
    const int64_t* indptr,
    size_t n_rows,
    size_t n_cols,
    size_t nnz,
    const int32_t* group_id,
    size_t n_groups,
    int threads
) {
    try {
        auto V = view::CscView<double>::from_raw(
            data, indices, indptr, n_rows, n_cols, nnz
        );
        
        size_t result_size = n_cols * n_groups;
        auto* counts = new size_t[result_size];
        
        auto result = group_count_nonzero<double>(
            V, group_id, n_groups, threads
        );
        std::memcpy(counts, result.data(), result_size * sizeof(size_t));
        
        return counts;
    } catch (...) {
        return nullptr;
    }
}

// ================================================================
// Normalization C API
// ================================================================

void sparse_row_sum_csr_capi(
    const double* data,
    const int64_t* indptr,
    size_t n_rows,
    double* out_sums,
    int n_threads
) {
    perturblab::kernel::normalization::sparse_row_sum_csr(
        data, indptr, n_rows, out_sums, n_threads
    );
}

void sparse_row_sum_csr_f32_capi(
    const float* data,
    const int64_t* indptr,
    size_t n_rows,
    float* out_sums,
    int n_threads
) {
    perturblab::kernel::normalization::sparse_row_sum_csr_f32(
        data, indptr, n_rows, out_sums, n_threads
    );
}

void inplace_divide_csr_rows_capi(
    double* data,
    const int64_t* indptr,
    size_t n_rows,
    const double* divisors,
    bool allow_zero_divisor,
    int n_threads
) {
    perturblab::kernel::normalization::inplace_divide_csr_rows(
        data, indptr, n_rows, divisors, allow_zero_divisor, n_threads
    );
}

void inplace_divide_csr_rows_f32_capi(
    float* data,
    const int64_t* indptr,
    size_t n_rows,
    const float* divisors,
    bool allow_zero_divisor,
    int n_threads
) {
    perturblab::kernel::normalization::inplace_divide_csr_rows_f32(
        data, indptr, n_rows, divisors, allow_zero_divisor, n_threads
    );
}

double compute_median_nonzero_capi(
    const double* values,
    size_t n
) {
    return perturblab::kernel::normalization::compute_median_nonzero(values, n);
}

void find_highly_expressed_genes_capi(
    const double* data,
    const int64_t* indptr,
    const int64_t* indices,
    size_t n_rows,
    size_t n_cols,
    const double* row_sums,
    double max_fraction,
    bool* out_gene_mask,
    int n_threads
) {
    perturblab::kernel::normalization::find_highly_expressed_genes(
        data, indptr, indices, n_rows, n_cols,
        row_sums, max_fraction, out_gene_mask, n_threads
    );
}

void sparse_row_sum_csr_exclude_genes_capi(
    const double* data,
    const int64_t* indptr,
    const int64_t* indices,
    size_t n_rows,
    const bool* gene_mask,
    double* out_sums,
    int n_threads
) {
    perturblab::kernel::normalization::sparse_row_sum_csr_exclude_genes(
        data, indptr, indices, n_rows, gene_mask, out_sums, n_threads
    );
}

// ================================================================
// Scale C API
// ================================================================

void sparse_standardize_csc_capi(
    double* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    const double* means,
    const double* stds,
    bool zero_center,
    double max_value,
    int n_threads
) {
    perturblab::kernel::scale::sparse_standardize_csc(
        data, row_indices, col_ptr, n_rows, n_cols,
        means, stds, zero_center, max_value, n_threads
    );
}

void sparse_standardize_csc_f32_capi(
    float* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    const float* means,
    const float* stds,
    bool zero_center,
    float max_value,
    int n_threads
) {
    perturblab::kernel::scale::sparse_standardize_csc_f32(
        data, row_indices, col_ptr, n_rows, n_cols,
        means, stds, zero_center, max_value, n_threads
    );
}

void sparse_standardize_csr_capi(
    double* data,
    const int64_t* col_indices,
    const int64_t* row_ptr,
    size_t n_rows,
    size_t n_cols,
    const double* means,
    const double* stds,
    bool zero_center,
    double max_value,
    int n_threads
) {
    perturblab::kernel::scale::sparse_standardize_csr(
        data, col_indices, row_ptr, n_rows, n_cols,
        means, stds, zero_center, max_value, n_threads
    );
}

void sparse_standardize_csr_f32_capi(
    float* data,
    const int64_t* col_indices,
    const int64_t* row_ptr,
    size_t n_rows,
    size_t n_cols,
    const float* means,
    const float* stds,
    bool zero_center,
    float max_value,
    int n_threads
) {
    perturblab::kernel::scale::sparse_standardize_csr_f32(
        data, col_indices, row_ptr, n_rows, n_cols,
        means, stds, zero_center, max_value, n_threads
    );
}

void dense_standardize_capi(
    double* data,
    size_t n_rows,
    size_t n_cols,
    const double* means,
    const double* stds,
    bool zero_center,
    double max_value,
    int n_threads
) {
    perturblab::kernel::scale::dense_standardize(
        data, n_rows, n_cols, means, stds,
        zero_center, max_value, n_threads
    );
}

void dense_standardize_f32_capi(
    float* data,
    size_t n_rows,
    size_t n_cols,
    const float* means,
    const float* stds,
    bool zero_center,
    float max_value,
    int n_threads
) {
    perturblab::kernel::scale::dense_standardize_f32(
        data, n_rows, n_cols, means, stds,
        zero_center, max_value, n_threads
    );
}

} // extern "C"
