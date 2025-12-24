// group_ops.cpp
// Group-wise operations on sparse matrices
//
// This file provides high-performance group aggregation functions (mean, variance, etc.)
// for sparse matrices in CSC format.
//
// Key optimizations:
// - OpenMP column-wise parallelization (dynamic scheduling for load balancing)
// - Thread-local buffers to avoid allocations
// - Kahan summation option for numerical stability
// - Welford algorithm for single-pass mean+variance computation
//
#include "group_ops.hpp"
#include "common.hpp"
#include "macro.hpp"
#include <cstring>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_max_threads() { return 1; }
static inline int omp_get_thread_num() { return 0; }
#endif

namespace perturblab {
namespace kernel {

/**
 * @brief Helper: Compute row counts per group.
 */
static inline void compute_group_rows(
    const int32_t* group_id,
    size_t n_rows,
    size_t n_groups,
    std::vector<size_t>& group_rows
) {
    group_rows.assign(n_groups, 0);
    for (size_t r = 0; r < n_rows; ++r) {
        int g = group_id[r];
        if (g >= 0 && static_cast<size_t>(g) < n_groups) {
            ++group_rows[static_cast<size_t>(g)];
        }
    }
}

// ================================================================
// Group Mean - Standard version
// ================================================================

template<class T>
static force_inline_ std::vector<double>
group_mean_core(
    const T*           data,
    const int64_t*     indices,
    const int64_t*     indptr,
    size_t             n_rows,
    size_t             n_cols,
    const int32_t*     group_id,
    size_t             n_groups,
    bool               include_zeros,
    int                threads
) {
    const size_t G = n_groups;
    const size_t C = n_cols;
    const size_t R = n_rows;
    if (G == 0 || C == 0) return {};
    std::vector<size_t> group_rows;
    compute_group_rows(group_id, R, G, group_rows);
    if (threads <= 0) threads = omp_get_max_threads();
    std::vector<double> mean_out(C * G, 0.0);

    #pragma omp parallel num_threads(threads)
    {
        std::vector<double> sum(G);
        std::vector<size_t> valid_cnt(G);
        std::vector<size_t> invalid_cnt(G);
        #pragma omp for schedule(dynamic, 64)
        for (std::ptrdiff_t cc = 0; cc < static_cast<std::ptrdiff_t>(C); ++cc) {
            const size_t c = static_cast<size_t>(cc);
            std::memset(sum.data(), 0, G * sizeof(double));
            std::memset(valid_cnt.data(), 0, G * sizeof(size_t));
            std::memset(invalid_cnt.data(), 0, G * sizeof(size_t));
            const int64_t p0 = indptr[c], p1 = indptr[c + 1];
            for (int64_t p = p0; p < p1; ++p) {
                const int64_t r = indices[p];
                if unlikely_(r < 0 || static_cast<size_t>(r) >= R) continue;
                const int gi = group_id[static_cast<size_t>(r)];
                if unlikely_(gi < 0 || static_cast<size_t>(gi) >= G) continue;
                const double v = static_cast<double>(data[p]);
                if likely_(is_valid_value(v)) {
                    sum[static_cast<size_t>(gi)] += v;
                    ++valid_cnt[static_cast<size_t>(gi)];
                } else { ++invalid_cnt[static_cast<size_t>(gi)]; }
            }
            double* out_col = mean_out.data() + c * G;
            for (size_t g = 0; g < G; ++g) {
                size_t denom = include_zeros ? (group_rows[g] - invalid_cnt[g]) : valid_cnt[g];
                out_col[g] = (denom > 0) ? (sum[g] / static_cast<double>(denom)) : 0.0;
            }
        }
    }
    return mean_out;
}

// ================================================================
// Group Mean - Kahan summation version
// ================================================================

template<class T>
static force_inline_ std::vector<double>
group_mean_core_kahan(
    const T*           data,
    const int64_t*     indices,
    const int64_t*     indptr,
    size_t             n_rows,
    size_t             n_cols,
    const int32_t*     group_id,
    size_t             n_groups,
    bool               include_zeros,
    int                threads
) {
    const size_t G = n_groups;
    const size_t C = n_cols;
    const size_t R = n_rows;
    if (G == 0 || C == 0) return {};
    std::vector<size_t> group_rows;
    compute_group_rows(group_id, R, G, group_rows);
    if (threads <= 0) threads = omp_get_max_threads();
    std::vector<double> mean_out(C * G, 0.0);

    #pragma omp parallel num_threads(threads)
    {
        std::vector<double> sum(G, 0.0);
        std::vector<double> comp(G, 0.0);
        std::vector<size_t> valid_cnt(G);
        std::vector<size_t> invalid_cnt(G);
        #pragma omp for schedule(dynamic, 64)
        for (std::ptrdiff_t cc = 0; cc < static_cast<std::ptrdiff_t>(C); ++cc) {
            const size_t c = static_cast<size_t>(cc);
            std::fill(sum.begin(), sum.end(), 0.0);
            std::fill(comp.begin(), comp.end(), 0.0);
            std::fill(valid_cnt.begin(), valid_cnt.end(), 0);
            std::fill(invalid_cnt.begin(), invalid_cnt.end(), 0);
            const int64_t p0 = indptr[c], p1 = indptr[c + 1];
            for (int64_t p = p0; p < p1; ++p) {
                const int64_t r = indices[p];
                if unlikely_(r < 0 || static_cast<size_t>(r) >= R) continue;
                const int gi = group_id[static_cast<size_t>(r)];
                if unlikely_(gi < 0 || static_cast<size_t>(gi) >= G) continue;
                const double v = static_cast<double>(data[p]);
                if likely_(is_valid_value(v)) {
                    const size_t g = static_cast<size_t>(gi);
                    double y = v - comp[g];
                    double t = sum[g] + y;
                    comp[g] = (t - sum[g]) - y;
                    sum[g] = t;
                    ++valid_cnt[g];
                } else { ++invalid_cnt[static_cast<size_t>(gi)]; }
            }
            double* out_col = mean_out.data() + c * G;
            for (size_t g = 0; g < G; ++g) {
                size_t denom = include_zeros ? (group_rows[g] - invalid_cnt[g]) : valid_cnt[g];
                out_col[g] = (denom > 0) ? (sum[g] / static_cast<double>(denom)) : 0.0;
            }
        }
    }
    return mean_out;
}

// ================================================================
// Group Variance - Welford online algorithm
// ================================================================

template<class T>
static force_inline_ std::vector<double>
group_var_core(
    const T*           data,
    const int64_t*     indices,
    const int64_t*     indptr,
    size_t             n_rows,
    size_t             n_cols,
    const int32_t*     group_id,
    size_t             n_groups,
    bool               include_zeros,
    int                threads,
    int                ddof
) {
    const size_t G = n_groups;
    const size_t C = n_cols;
    const size_t R = n_rows;
    if (G == 0 || C == 0) return {};
    std::vector<size_t> group_rows;
    compute_group_rows(group_id, R, G, group_rows);
    if (threads <= 0) threads = omp_get_max_threads();
    std::vector<double> var_out(C * G, 0.0);

    #pragma omp parallel num_threads(threads)
    {
        std::vector<double> mean(G);
        std::vector<double> M2(G);
        std::vector<size_t> count(G);
        std::vector<size_t> invalid_cnt(G);
        #pragma omp for schedule(dynamic, 64)
        for (std::ptrdiff_t cc = 0; cc < static_cast<std::ptrdiff_t>(C); ++cc) {
            const size_t c = static_cast<size_t>(cc);
            std::fill(mean.begin(), mean.end(), 0.0);
            std::fill(M2.begin(), M2.end(), 0.0);
            std::fill(count.begin(), count.end(), 0);
            std::fill(invalid_cnt.begin(), invalid_cnt.end(), 0);
            const int64_t p0 = indptr[c], p1 = indptr[c + 1];
            for (int64_t p = p0; p < p1; ++p) {
                const int64_t r = indices[p];
                if unlikely_(r < 0 || static_cast<size_t>(r) >= R) continue;
                const int gi = group_id[static_cast<size_t>(r)];
                if unlikely_(gi < 0 || static_cast<size_t>(gi) >= G) continue;
                const double v = static_cast<double>(data[p]);
                if likely_(is_valid_value(v)) {
                    const size_t g = static_cast<size_t>(gi);
                    ++count[g];
                    double delta = v - mean[g];
                    mean[g] += delta / static_cast<double>(count[g]);
                    double delta2 = v - mean[g];
                    M2[g] += delta * delta2;
                } else { ++invalid_cnt[static_cast<size_t>(gi)]; }
            }
            if (include_zeros) {
                for (size_t g = 0; g < G; ++g) {
                    size_t n_zeros = group_rows[g] - count[g] - invalid_cnt[g];
                    if (n_zeros > 0) {
                        for (size_t i = 0; i < n_zeros; ++i) {
                            ++count[g];
                            double delta = -mean[g];
                            mean[g] += delta / static_cast<double>(count[g]);
                            double delta2 = -mean[g];
                            M2[g] += delta * delta2;
                        }
                    }
                }
            }
            double* out_col = var_out.data() + c * G;
            for (size_t g = 0; g < G; ++g) {
                int64_t n = static_cast<int64_t>(count[g]) - ddof;
                out_col[g] = (n > 0) ? (M2[g] / static_cast<double>(n)) : 0.0;
            }
        }
    }
    return var_out;
}

// ================================================================
// Group Mean and Variance - Single pass
// ================================================================

template<class T>
static force_inline_ std::pair<std::vector<double>, std::vector<double>>
group_mean_var_core(
    const T*           data,
    const int64_t*     indices,
    const int64_t*     indptr,
    size_t             n_rows,
    size_t             n_cols,
    const int32_t*     group_id,
    size_t             n_groups,
    bool               include_zeros,
    int                threads,
    int                ddof
) {
    const size_t G = n_groups;
    const size_t C = n_cols;
    const size_t R = n_rows;
    if (G == 0 || C == 0) return {{}, {}};
    std::vector<size_t> group_rows;
    compute_group_rows(group_id, R, G, group_rows);
    if (threads <= 0) threads = omp_get_max_threads();
    std::vector<double> mean_out(C * G, 0.0);
    std::vector<double> var_out(C * G, 0.0);

    #pragma omp parallel num_threads(threads)
    {
        std::vector<double> mean(G);
        std::vector<double> M2(G);
        std::vector<size_t> count(G);
        std::vector<size_t> invalid_cnt(G);
        #pragma omp for schedule(dynamic, 64)
        for (std::ptrdiff_t cc = 0; cc < static_cast<std::ptrdiff_t>(C); ++cc) {
            const size_t c = static_cast<size_t>(cc);
            std::fill(mean.begin(), mean.end(), 0.0);
            std::fill(M2.begin(), M2.end(), 0.0);
            std::fill(count.begin(), count.end(), 0);
            std::fill(invalid_cnt.begin(), invalid_cnt.end(), 0);
            const int64_t p0 = indptr[c], p1 = indptr[c + 1];
            for (int64_t p = p0; p < p1; ++p) {
                const int64_t r = indices[p];
                if unlikely_(r < 0 || static_cast<size_t>(r) >= R) continue;
                const int gi = group_id[static_cast<size_t>(r)];
                if unlikely_(gi < 0 || static_cast<size_t>(gi) >= G) continue;
                const double v = static_cast<double>(data[p]);
                if likely_(is_valid_value(v)) {
                    const size_t g = static_cast<size_t>(gi);
                    ++count[g];
                    double delta = v - mean[g];
                    mean[g] += delta / static_cast<double>(count[g]);
                    double delta2 = v - mean[g];
                    M2[g] += delta * delta2;
                } else { ++invalid_cnt[static_cast<size_t>(gi)]; }
            }
            if (include_zeros) {
                for (size_t g = 0; g < G; ++g) {
                    size_t n_zeros = group_rows[g] - count[g] - invalid_cnt[g];
                    if (n_zeros > 0) {
                        for (size_t i = 0; i < n_zeros; ++i) {
                            ++count[g];
                            double delta = -mean[g];
                            mean[g] += delta / static_cast<double>(count[g]);
                            double delta2 = -mean[g];
                            M2[g] += delta * delta2;
                        }
                    }
                }
            }
            double* out_mean_col = mean_out.data() + c * G;
            double* out_var_col = var_out.data() + c * G;
            for (size_t g = 0; g < G; ++g) {
                out_mean_col[g] = mean[g];
                int64_t n = static_cast<int64_t>(count[g]) - ddof;
                out_var_col[g] = (n > 0) ? (M2[g] / static_cast<double>(n)) : 0.0;
            }
        }
    }
    return {mean_out, var_out};
}

// ================================================================
// Group Count Nonzero
// ================================================================

template<class T>
static force_inline_ std::vector<size_t>
group_count_nonzero_core(
    const T*           data,
    const int64_t*     indices,
    const int64_t*     indptr,
    size_t             n_rows,
    size_t             n_cols,
    const int32_t*     group_id,
    size_t             n_groups,
    int                threads
) {
    const size_t G = n_groups;
    const size_t C = n_cols;
    const size_t R = n_rows;
    if (G == 0 || C == 0) return {};
    if (threads <= 0) threads = omp_get_max_threads();
    std::vector<size_t> count_out(C * G, 0);
    #pragma omp parallel num_threads(threads)
    {
        std::vector<size_t> count(G);
        #pragma omp for schedule(dynamic, 64)
        for (std::ptrdiff_t cc = 0; cc < static_cast<std::ptrdiff_t>(C); ++cc) {
            const size_t c = static_cast<size_t>(cc);
            std::fill(count.begin(), count.end(), 0);
            const int64_t p0 = indptr[c], p1 = indptr[c + 1];
            for (int64_t p = p0; p < p1; ++p) {
                const int64_t r = indices[p];
                if unlikely_(r < 0 || static_cast<size_t>(r) >= R) continue;
                const int gi = group_id[static_cast<size_t>(r)];
                if unlikely_(gi < 0 || static_cast<size_t>(gi) >= G) continue;
                const double v = static_cast<double>(data[p]);
                if likely_(is_valid_value(v)) { ++count[static_cast<size_t>(gi)]; }
            }
            size_t* out_col = count_out.data() + c * G;
            std::memcpy(out_col, count.data(), G * sizeof(size_t));
        }
    }
    return count_out;
}

// ================================================================
// External wrappers
// ================================================================

template<class T>
std::vector<double> group_mean(
    const view::CscView<T>& A,
    const int32_t* group_id,
    const size_t& n_groups,
    bool include_zeros,
    int threads,
    bool use_kahan
) {
    if (use_kahan) {
        return group_mean_core_kahan<T>(A.data(), A.indices(), A.indptr(), A.rows(), A.cols(), group_id, n_groups, include_zeros, threads);
    } else {
        return group_mean_core<T>(A.data(), A.indices(), A.indptr(), A.rows(), A.cols(), group_id, n_groups, include_zeros, threads);
    }
}

template<class T>
std::vector<double> group_var(
    const view::CscView<T>& A,
    const int32_t* group_id,
    const size_t& n_groups,
    bool include_zeros,
    int threads,
    int ddof
) {
    return group_var_core<T>(A.data(), A.indices(), A.indptr(), A.rows(), A.cols(), group_id, n_groups, include_zeros, threads, ddof);
}

template<class T>
std::pair<std::vector<double>, std::vector<double>> group_mean_var(
    const view::CscView<T>& A,
    const int32_t* group_id,
    const size_t& n_groups,
    bool include_zeros,
    int threads,
    int ddof
) {
    return group_mean_var_core<T>(A.data(), A.indices(), A.indptr(), A.rows(), A.cols(), group_id, n_groups, include_zeros, threads, ddof);
}

template<class T>
std::vector<size_t> group_count_nonzero(
    const view::CscView<T>& A,
    const int32_t* group_id,
    const size_t& n_groups,
    int threads
) {
    return group_count_nonzero_core<T>(A.data(), A.indices(), A.indptr(), A.rows(), A.cols(), group_id, n_groups, threads);
}

#define GROUP_OPS_INSTANTIATE(T) \
    template std::vector<double> group_mean<T>( \
        const view::CscView<T>&, \
        const int32_t*, \
        const size_t&, \
        bool, \
        int, \
        bool \
    ); \
    template std::vector<double> group_var<T>( \
        const view::CscView<T>&, \
        const int32_t*, \
        const size_t&, \
        bool, \
        int, \
        int \
    ); \
    template std::pair<std::vector<double>, std::vector<double>> group_mean_var<T>( \
        const view::CscView<T>&, \
        const int32_t*, \
        const size_t&, \
        bool, \
        int, \
        int \
    ); \
    template std::vector<size_t> group_count_nonzero<T>( \
        const view::CscView<T>&, \
        const int32_t*, \
        const size_t&, \
        int \
    );

TYPE_DISPATCH(GROUP_OPS_INSTANTIATE);
#undef GROUP_OPS_INSTANTIATE

} // namespace kernel
} // namespace perturblab
