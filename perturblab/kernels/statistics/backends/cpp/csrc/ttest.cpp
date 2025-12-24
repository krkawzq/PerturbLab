// ttest.cpp - T-test implementations
//
// This file provides t-test functionality for single-cell differential expression analysis.
//
// Algorithm: Student's t-test, Welch's t-test, Log fold change
//
#include "ttest.hpp"
#include "macro.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_max_threads() { return 1; }
static inline int omp_get_thread_num() { return 0; }
#endif

// Highway SIMD support
#include <hwy/highway.h>
namespace hn = hwy::HWY_NAMESPACE;

namespace perturblab {
namespace kernel {

// ============================================================================
// Data Structures & Helpers
// ============================================================================

struct GroupStats {
    double sum = 0.0;
    double sum_sq = 0.0;
    size_t nnz = 0; // count of non-zero elements encountered
};

/**
 * @brief SIMD-accelerated accumulation for contiguous data blocks.
 */
template<typename T>
force_inline_ void simd_accumulate_group(
    const T* data,
    size_t count,
    GroupStats& stats
) {
    if (count == 0) return;
    using D = HWY_FULL(double);
    const D d;
    const size_t lanes = hn::Lanes(d);
    auto vec_sum = hn::Zero(d);
    auto vec_sum_sq = hn::Zero(d);
    size_t i = 0;
    if (count >= lanes) {
        for (; i + lanes <= count; i += lanes) {
            alignas(32) double temp_data[32];
            for (size_t j = 0; j < lanes && (i + j) < count; ++j) {
                temp_data[j] = static_cast<double>(data[i + j]);
            }
            auto vec_data = hn::LoadU(d, temp_data);
            vec_sum = hn::Add(vec_sum, vec_data);
            vec_sum_sq = hn::MulAdd(vec_data, vec_data, vec_sum_sq);
        }
        stats.sum += hn::ReduceSum(d, vec_sum);
        stats.sum_sq += hn::ReduceSum(d, vec_sum_sq);
    }
    for (; i < count; ++i) {
        double val = static_cast<double>(data[i]);
        stats.sum += val;
        stats.sum_sq += val * val;
    }
    stats.nnz += count;
}

struct FinalStats {
    double mean;
    double var;
    size_t n; // total count (including zeros)
};

/**
 * @brief Compute mean and variance from sufficient statistics.
 */
static inline FinalStats finish_stats(const GroupStats& raw, size_t total_count) {
    FinalStats res;
    res.n = total_count;
    if (total_count == 0) {
        res.mean = 0.0;
        res.var = 0.0;
        return res;
    }
    res.mean = raw.sum / static_cast<double>(total_count);
    if (total_count > 1) {
        double mean_sq = raw.sum_sq / static_cast<double>(total_count);
        double m2 = mean_sq - res.mean * res.mean;
        if (m2 < 0) m2 = 0.0;
        res.var = m2 * static_cast<double>(total_count) / static_cast<double>(total_count - 1);
    } else {
        res.var = 0.0;
    }
    return res;
}

/**
 * @brief Fast approximation of T-distribution CDF.
 */
static inline double t_distribution_cdf_fast(double t, double df) {
    if (std::abs(t) > 30.0) return (t > 0) ? 1.0 : 0.0;
    if (df > 30.0) {
         double z = t / std::sqrt(1.0 + t * t / df);
         return 0.5 * std::erfc(-z * 0.7071067811865475);
    }
    double prob = 0.5;
    if (t != 0.0) {
        prob = 0.5 * (1.0 + t / std::sqrt(df + t * t));
    }
    return prob;
}

static inline double t_to_pvalue_fast(double t_stat, double df) {
    if (!std::isfinite(t_stat) || df <= 0) return 1.0;
    double abs_t = std::abs(t_stat);
    return 2.0 * (1.0 - t_distribution_cdf_fast(abs_t, df));
}

// ============================================================================
// Unified Kernel for T-Tests
// ============================================================================

enum TestType {
    STUDENT_T,
    WELCH_T
};

template<typename T, TestType TType>
TTestResult run_ttest_optimized(
    const view::CscView<T>& A,
    const int32_t* group_id,
    size_t n_targets,
    int threads,
    size_t* progress_ptr
) {
    const size_t n_rows = A.rows();
    const size_t n_cols = A.cols();
    std::vector<size_t> global_group_counts(n_targets + 1, 0);
    for (size_t i = 0; i < n_rows; ++i) {
        int32_t gid = group_id[i];
        if (gid >= 0 && gid <= static_cast<int32_t>(n_targets)) {
            global_group_counts[gid]++;
        }
    }
    const size_t n_ref_total = global_group_counts[0];

    TTestResult result;
    size_t total_out = n_targets * n_cols;
    result.t_statistic.resize(total_out);
    result.p_value.resize(total_out);
    result.mean_diff.resize(total_out);
    result.log2_fc.resize(total_out);

    int num_threads = threads > 0 ? threads : omp_get_max_threads();
    
    #pragma omp parallel num_threads(num_threads)
    {
        std::vector<GroupStats> local_stats(n_targets + 1);
        #pragma omp for schedule(dynamic, 32)
        for (size_t col = 0; col < n_cols; ++col) {
            std::fill(local_stats.begin(), local_stats.end(), GroupStats{0.0, 0.0, 0});
            int64_t start = A.indptr()[col];
            int64_t end = A.indptr()[col + 1];
            const auto* indices = A.indices();
            const auto* data = A.data();
            
            int64_t idx = start;
            while (idx < end) {
                int32_t gid = group_id[indices[idx]];
                if (gid < 0 || gid > static_cast<int32_t>(n_targets)) {
                    ++idx; continue;
                }
                int64_t batch_start = idx;
                int64_t batch_end = idx + 1;
                constexpr int64_t MAX_LOOKAHEAD = 16;
                while (batch_end < end && (batch_end - batch_start) < MAX_LOOKAHEAD && group_id[indices[batch_end]] == gid) {
                    ++batch_end;
                }
                size_t batch_size = batch_end - batch_start;
                if (batch_size >= 4) { simd_accumulate_group(data + batch_start, batch_size, local_stats[gid]); }
                else {
                    for (int64_t i = batch_start; i < batch_end; ++i) {
                        double val = static_cast<double>(data[i]);
                        local_stats[gid].sum += val;
                        local_stats[gid].sum_sq += val * val;
                        local_stats[gid].nnz++;
                    }
                }
                idx = batch_end;
            }

            FinalStats ref = finish_stats(local_stats[0], n_ref_total);
            for (size_t t = 0; t < n_targets; ++t) {
                int32_t target_idx = t + 1;
                size_t n_tar_total = global_group_counts[target_idx];
                size_t out_idx = t * n_cols + col;
                if (n_tar_total == 0 || n_ref_total == 0) {
                    result.p_value[out_idx] = 1.0;
                    result.t_statistic[out_idx] = 0.0;
                    result.mean_diff[out_idx] = 0.0;
                    result.log2_fc[out_idx] = 0.0;
                    continue;
                }
                FinalStats tar = finish_stats(local_stats[target_idx], n_tar_total);
                double t_stat = 0.0, df = 0.0, pooled_se = 0.0;
                if (TType == STUDENT_T) {
                    double pooled_var = 0.0;
                    if (ref.n + tar.n > 2) {
                        pooled_var = ((ref.n - 1) * ref.var + (tar.n - 1) * tar.var) / (ref.n + tar.n - 2);
                    }
                    pooled_se = std::sqrt(pooled_var * (1.0 / ref.n + 1.0 / tar.n));
                    df = static_cast<double>(ref.n + tar.n - 2);
                } else {
                    double se_sq = ref.var / ref.n + tar.var / tar.n;
                    pooled_se = std::sqrt(se_sq);
                    df = 1.0;
                    if (se_sq > 1e-12 && ref.n > 1 && tar.n > 1) {
                        double v1 = ref.var / ref.n;
                        double v2 = tar.var / tar.n;
                        df = (v1 + v2) * (v1 + v2) / (v1 * v1 / (ref.n - 1) + v2 * v2 / (tar.n - 1));
                    }
                }
                if (pooled_se > 1e-12) { t_stat = (tar.mean - ref.mean) / pooled_se; }
                result.t_statistic[out_idx] = t_stat;
                result.p_value[out_idx] = t_to_pvalue_fast(t_stat, df);
                result.mean_diff[out_idx] = tar.mean - ref.mean;
                constexpr double pseudocount = 1e-9;
                result.log2_fc[out_idx] = std::log2((tar.mean + pseudocount) / (ref.mean + pseudocount));
            }
            if (progress_ptr != nullptr && (col & 0xFF) == 0) {
                 #pragma omp atomic
                 (*progress_ptr) += 256;
            }
        }
    }
    return result;
}

// ============================================================================
// Public Interface Wrappers
// ============================================================================

template<typename T>
TTestResult student_ttest(const view::CscView<T>& A, const int32_t* group_id, size_t n_targets, int threads, size_t* progress_ptr) {
    return run_ttest_optimized<T, STUDENT_T>(A, group_id, n_targets, threads, progress_ptr);
}

template<typename T>
TTestResult welch_ttest(const view::CscView<T>& A, const int32_t* group_id, size_t n_targets, int threads, size_t* progress_ptr) {
    return run_ttest_optimized<T, WELCH_T>(A, group_id, n_targets, threads, progress_ptr);
}

template<typename T>
TTestResult log_fold_change(const view::CscView<T>& A, const int32_t* group_id, size_t n_targets, double pseudocount, int threads) {
    const size_t n_rows = A.rows();
    const size_t n_cols = A.cols();
    std::vector<size_t> group_counts(n_targets + 1, 0);
    for (size_t i = 0; i < n_rows; ++i) {
        int32_t gid = group_id[i];
        if (gid >= 0 && gid <= static_cast<int32_t>(n_targets)) { group_counts[gid]++; }
    }
    size_t n_ref = group_counts[0];
    TTestResult result;
    size_t total_size = n_targets * n_cols;
    result.t_statistic.assign(total_size, 0.0);
    result.p_value.assign(total_size, 1.0);
    result.mean_diff.resize(total_size);
    result.log2_fc.resize(total_size);
    int num_threads = threads > 0 ? threads : omp_get_max_threads();
    #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (size_t col = 0; col < n_cols; ++col) {
        int64_t start = A.indptr()[col], end = A.indptr()[col + 1];
        double ref_sum = 0.0;
        for (int64_t idx = start; idx < end; ++idx) {
            if (group_id[A.indices()[idx]] == 0) { ref_sum += static_cast<double>(A.data()[idx]); }
        }
        double ref_mean = (n_ref > 0) ? (ref_sum / n_ref) : 0.0;
        for (size_t t = 0; t < n_targets; ++t) {
            int32_t target_group = static_cast<int32_t>(t + 1);
            size_t tar_n = group_counts[t + 1];
            double tar_sum = 0.0;
            for (int64_t idx = start; idx < end; ++idx) {
                if (group_id[A.indices()[idx]] == target_group) { tar_sum += static_cast<double>(A.data()[idx]); }
            }
            double tar_mean = (tar_n > 0) ? (tar_sum / tar_n) : 0.0;
            size_t idx_out = t * n_cols + col;
            result.mean_diff[idx_out] = tar_mean - ref_mean;
            result.log2_fc[idx_out] = std::log2((tar_mean + pseudocount) / (ref_mean + pseudocount));
        }
    }
    return result;
}

// Explicit Instantiations
template TTestResult student_ttest<float>(const view::CscView<float>&, const int32_t*, size_t, int, size_t*);
template TTestResult student_ttest<double>(const view::CscView<double>&, const int32_t*, size_t, int, size_t*);
template TTestResult welch_ttest<float>(const view::CscView<float>&, const int32_t*, size_t, int, size_t*);
template TTestResult welch_ttest<double>(const view::CscView<double>&, const int32_t*, size_t, int, size_t*);
template TTestResult log_fold_change<float>(const view::CscView<float>&, const int32_t*, size_t, double, int);
template TTestResult log_fold_change<double>(const view::CscView<double>&, const int32_t*, size_t, double, int);

} // namespace kernel
} // namespace perturblab
