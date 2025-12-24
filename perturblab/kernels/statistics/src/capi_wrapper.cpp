// capi_wrapper.cpp
// C API wrapper for statistical test functions
//
// 🤖 AI-GENERATED CODE (Cursor AI Assistant)
//
// This file provides a C API interface to the C++ statistical kernels,
// enabling Python integration via ctypes without pybind11 dependency.
//
// The wrapper handles:
// - Memory management (allocation and deallocation)
// - Type conversions (Python arrays → C++ structures)
// - Error handling (returns nullptr on failure)
// - Multi-dtype support (float32, float64)
//
// Generated: 2025-12-24
//
// Provides a C interface to the C++ implementation for use with ctypes

#include "include/mannwhitneyu.hpp"
#include "include/ttest.hpp"
#include "include/sparse.hpp"
#include <cstring>
#include <cstdlib>

extern "C" {

// Structure to hold results
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

// Free results
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

// Mann-Whitney U test for CSC matrix
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
        // Create CSC view
        auto V = hpdex::view::CscView<double>::from_raw(
            data, indices, indptr,
            n_rows, n_cols, nnz
        );
        
        // Setup options
        hpdex::MannWhitneyuOption opt{};
        opt.ref_sorted = false;
        opt.tar_sorted = false;
        opt.tie_correction = tie_correction;
        opt.use_continuity = use_continuity;
        opt.fast_norm = true;
        opt.zero_handling = hpdex::MannWhitneyuOption::min;
        opt.alternative = hpdex::MannWhitneyuOption::two_sided;
        opt.method = hpdex::MannWhitneyuOption::asymptotic;
        
        // Call C++ implementation
        auto result = hpdex::mannwhitneyu<double>(
            V, group_id, n_targets, opt, threads, nullptr
        );
        
        // Allocate and copy results
        size_t result_size = result.U1.size();
        auto* ret = new MWUResults;
        ret->U1 = new double[result_size];
        ret->U2 = new double[result_size];
        ret->P = new double[result_size];
        ret->size = result_size;
        
        std::memcpy(ret->U1, result.U1.data(), result_size * sizeof(double));
        std::memcpy(ret->U2, result.U2.data(), result_size * sizeof(double));
        std::memcpy(ret->P, result.P.data(), result_size * sizeof(double));
        
        return ret;
    } catch (...) {
        return nullptr;
    }
}

// Group mean for CSC matrix
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
        // Create CSC view
        auto V = hpdex::view::CscView<double>::from_raw(
            data, indices, indptr,
            n_rows, n_cols, nnz
        );
        
        // Call C++ implementation
        auto result = hpdex::group_mean<double>(
            V, group_id, n_groups, include_zeros, threads, false
        );
        
        // Allocate and copy results
        size_t result_size = result.size();
        auto* ret = new GroupMeanResults;
        ret->means = new double[result_size];
        ret->size = result_size;
        
        std::memcpy(ret->means, result.data(), result_size * sizeof(double));
        
        return ret;
    } catch (...) {
        return nullptr;
    }
}

// Float32 variants
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
        auto V = hpdex::view::CscView<float>::from_raw(
            data, indices, indptr,
            n_rows, n_cols, nnz
        );
        
        hpdex::MannWhitneyuOption opt{};
        opt.ref_sorted = false;
        opt.tar_sorted = false;
        opt.tie_correction = tie_correction;
        opt.use_continuity = use_continuity;
        opt.fast_norm = true;
        opt.zero_handling = hpdex::MannWhitneyuOption::min;
        opt.alternative = hpdex::MannWhitneyuOption::two_sided;
        opt.method = hpdex::MannWhitneyuOption::asymptotic;
        
        auto result = hpdex::mannwhitneyu<float>(
            V, group_id, n_targets, opt, threads, nullptr
        );
        
        size_t result_size = result.U1.size();
        auto* ret = new MWUResults;
        ret->U1 = new double[result_size];
        ret->U2 = new double[result_size];
        ret->P = new double[result_size];
        ret->size = result_size;
        
        std::memcpy(ret->U1, result.U1.data(), result_size * sizeof(double));
        std::memcpy(ret->U2, result.U2.data(), result_size * sizeof(double));
        std::memcpy(ret->P, result.P.data(), result_size * sizeof(double));
        
        return ret;
    } catch (...) {
        return nullptr;
    }
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
        auto V = hpdex::view::CscView<float>::from_raw(
            data, indices, indptr,
            n_rows, n_cols, nnz
        );
        
        auto result = hpdex::group_mean<float>(
            V, group_id, n_groups, include_zeros, threads, false
        );
        
        size_t result_size = result.size();
        auto* ret = new GroupMeanResults;
        ret->means = new double[result_size];
        ret->size = result_size;
        
        std::memcpy(ret->means, result.data(), result_size * sizeof(double));
        
        return ret;
    } catch (...) {
        return nullptr;
    }
}

// ============================================================================
// T-Test APIs
// ============================================================================

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

// Student's t-test
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
        auto V = hpdex::view::CscView<double>::from_raw(
            data, indices, indptr,
            n_rows, n_cols, nnz
        );
        
        auto result = hpdex::student_ttest<double>(
            V, group_id, n_targets, threads, nullptr
        );
        
        size_t result_size = result.t_statistic.size();
        auto* ret = new TTestResults;
        ret->t_statistic = new double[result_size];
        ret->p_value = new double[result_size];
        ret->mean_diff = new double[result_size];
        ret->log2_fc = new double[result_size];
        ret->size = result_size;
        
        std::memcpy(ret->t_statistic, result.t_statistic.data(), result_size * sizeof(double));
        std::memcpy(ret->p_value, result.p_value.data(), result_size * sizeof(double));
        std::memcpy(ret->mean_diff, result.mean_diff.data(), result_size * sizeof(double));
        std::memcpy(ret->log2_fc, result.log2_fc.data(), result_size * sizeof(double));
        
        return ret;
    } catch (...) {
        return nullptr;
    }
}

// Welch's t-test
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
        auto V = hpdex::view::CscView<double>::from_raw(
            data, indices, indptr,
            n_rows, n_cols, nnz
        );
        
        auto result = hpdex::welch_ttest<double>(
            V, group_id, n_targets, threads, nullptr
        );
        
        size_t result_size = result.t_statistic.size();
        auto* ret = new TTestResults;
        ret->t_statistic = new double[result_size];
        ret->p_value = new double[result_size];
        ret->mean_diff = new double[result_size];
        ret->log2_fc = new double[result_size];
        ret->size = result_size;
        
        std::memcpy(ret->t_statistic, result.t_statistic.data(), result_size * sizeof(double));
        std::memcpy(ret->p_value, result.p_value.data(), result_size * sizeof(double));
        std::memcpy(ret->mean_diff, result.mean_diff.data(), result_size * sizeof(double));
        std::memcpy(ret->log2_fc, result.log2_fc.data(), result_size * sizeof(double));
        
        return ret;
    } catch (...) {
        return nullptr;
    }
}

// Log fold change
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
        auto V = hpdex::view::CscView<double>::from_raw(
            data, indices, indptr,
            n_rows, n_cols, nnz
        );
        
        auto result = hpdex::log_fold_change<double>(
            V, group_id, n_targets, pseudocount, threads
        );
        
        size_t result_size = result.mean_diff.size();
        auto* ret = new TTestResults;
        ret->t_statistic = new double[result_size](); // Zero initialize
        ret->p_value = new double[result_size]();
        ret->mean_diff = new double[result_size];
        ret->log2_fc = new double[result_size];
        ret->size = result_size;
        
        std::memcpy(ret->mean_diff, result.mean_diff.data(), result_size * sizeof(double));
        std::memcpy(ret->log2_fc, result.log2_fc.data(), result_size * sizeof(double));
        
        return ret;
    } catch (...) {
        return nullptr;
    }
}

} // extern "C"

