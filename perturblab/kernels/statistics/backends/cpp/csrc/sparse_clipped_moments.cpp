// sparse_clipped_moments.cpp
// Sparse Clipped Moments - Core operator for Seurat V3 HVG detection
//
// This file implements the most performance-critical operator for highly variable gene (HVG)
// detection using the Seurat V3 algorithm. The operator computes clipped moments:
//   sum_j = Σ_i min(X_ij, clip_j)
//   sum_sq_j = Σ_i min(X_ij, clip_j)²
//
// Key optimizations:
// - CSC format for lock-free column-wise parallelization
// - Highway SIMD library for vectorization of min/mul/add operations
// - OpenMP dynamic scheduling for load balancing (columns have varying nnz)
// - Thread-local accumulators to avoid false sharing
//
#include "sparse_clipped_moments.hpp"
#include "common.hpp"
#include "macro.hpp"
#include "simd.hpp"
#include <cstring>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_max_threads() { return 1; }
#endif

namespace perturblab {
namespace kernel {
namespace hvg {

// ================================================================
// CSC Version - Double Precision
// ================================================================

void sparse_clipped_moments_csc(
    const double* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_cols,
    const double* clip_vals,
    double* out_sums,
    double* out_sum_sq,
    int n_threads
) {
    if (n_cols == 0) return;
    if (n_threads <= 0) n_threads = omp_get_max_threads();
    #pragma omp parallel num_threads(n_threads)
    {
        using D = HWY_FULL(double);
        D d;
        const size_t vec_lanes = Lanes(d);
        #pragma omp for schedule(dynamic, 64)
        for (std::ptrdiff_t cc = 0; cc < static_cast<std::ptrdiff_t>(n_cols); ++cc) {
            const size_t col = static_cast<size_t>(cc);
            const int64_t p_start = col_ptr[col], p_end = col_ptr[col + 1];
            const int64_t nnz_col = p_end - p_start;
            if (nnz_col == 0) { out_sums[col] = 0.0; out_sum_sq[col] = 0.0; continue; }
            const double clip = clip_vals[col];
            auto v_sum = Zero(d); auto v_sum_sq = Zero(d); auto v_clip = Set(d, clip);
            int64_t p = p_start;
            for (; p + static_cast<int64_t>(vec_lanes) <= p_end; p += vec_lanes) {
                auto v_data = Load(d, data + p);
                auto v_clipped = Min(v_data, v_clip);
                v_sum = Add(v_sum, v_clipped);
                v_sum_sq = Add(v_sum_sq, Mul(v_clipped, v_clipped));
            }
            double sum_scalar = ReduceSum(d, v_sum);
            double sum_sq_scalar = ReduceSum(d, v_sum_sq);
            for (; p < p_end; ++p) {
                double val = data[p];
                double clipped = (val < clip) ? val : clip;
                sum_scalar += clipped;
                sum_sq_scalar += clipped * clipped;
            }
            out_sums[col] = sum_scalar;
            out_sum_sq[col] = sum_sq_scalar;
        }
    }
}

// ================================================================
// CSC Version - Single Precision (Float32)
// ================================================================

void sparse_clipped_moments_csc_f32(
    const float* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_cols,
    const double* clip_vals,
    double* out_sums,
    double* out_sum_sq,
    int n_threads
) {
    if (n_cols == 0) return;
    if (n_threads <= 0) n_threads = omp_get_max_threads();
    #pragma omp parallel num_threads(n_threads)
    {
        using D = HWY_FULL(float);
        D d;
        const size_t vec_lanes = Lanes(d);
        #pragma omp for schedule(dynamic, 64)
        for (std::ptrdiff_t cc = 0; cc < static_cast<std::ptrdiff_t>(n_cols); ++cc) {
            const size_t col = static_cast<size_t>(cc);
            const int64_t p_start = col_ptr[col], p_end = col_ptr[col + 1];
            if (p_start == p_end) { out_sums[col] = 0.0; out_sum_sq[col] = 0.0; continue; }
            const float clip_f = static_cast<float>(clip_vals[col]);
            auto v_sum = Zero(d); auto v_sum_sq = Zero(d); auto v_clip = Set(d, clip_f);
            int64_t p = p_start;
            for (; p + static_cast<int64_t>(vec_lanes) <= p_end; p += vec_lanes) {
                auto v_data = Load(d, data + p);
                auto v_clipped = Min(v_data, v_clip);
                v_sum = Add(v_sum, v_clipped);
                v_sum_sq = Add(v_sum_sq, Mul(v_clipped, v_clipped));
            }
            double sum_scalar = static_cast<double>(ReduceSum(d, v_sum));
            double sum_sq_scalar = static_cast<double>(ReduceSum(d, v_sum_sq));
            for (; p < p_end; ++p) {
                float val = data[p];
                float clipped = (val < clip_f) ? val : clip_f;
                sum_scalar += static_cast<double>(clipped);
                sum_sq_scalar += static_cast<double>(clipped * clipped);
            }
            out_sums[col] = sum_scalar;
            out_sum_sq[col] = sum_sq_scalar;
        }
    }
}

// ================================================================
// CSR Version - Double Precision
// ================================================================

void sparse_clipped_moments_csr(
    const double* data,
    const int64_t* col_indices,
    const int64_t* row_ptr,
    size_t n_rows,
    size_t n_cols,
    size_t nnz,
    const double* clip_vals,
    double* out_sums,
    double* out_sum_sq,
    int n_threads
) {
    if (n_cols == 0 || n_rows == 0) return;
    if (n_threads <= 0) n_threads = omp_get_max_threads();
    std::memset(out_sums, 0, n_cols * sizeof(double));
    std::memset(out_sum_sq, 0, n_cols * sizeof(double));
    #pragma omp parallel num_threads(n_threads)
    {
        std::vector<double> local_sums(n_cols, 0.0);
        std::vector<double> local_sum_sq(n_cols, 0.0);
        #pragma omp for schedule(dynamic, 128)
        for (std::ptrdiff_t rr = 0; rr < static_cast<std::ptrdiff_t>(n_rows); ++rr) {
            const size_t row = static_cast<size_t>(rr);
            const int64_t p_start = row_ptr[row], p_end = row_ptr[row + 1];
            for (int64_t p = p_start; p < p_end; ++p) {
                const int64_t col_idx = col_indices[p];
                if unlikely_(col_idx < 0 || static_cast<size_t>(col_idx) >= n_cols) continue;
                const size_t col = static_cast<size_t>(col_idx);
                const double val = data[p], clip = clip_vals[col];
                const double clipped = (val < clip) ? val : clip;
                local_sums[col] += clipped;
                local_sum_sq[col] += clipped * clipped;
            }
        }
        #pragma omp critical
        {
            for (size_t c = 0; c < n_cols; ++c) {
                out_sums[c] += local_sums[c];
                out_sum_sq[c] += local_sum_sq[c];
            }
        }
    }
}

// ================================================================
// CSR Version - Single Precision
// ================================================================

void sparse_clipped_moments_csr_f32(
    const float* data,
    const int64_t* col_indices,
    const int64_t* row_ptr,
    size_t n_rows,
    size_t n_cols,
    size_t nnz,
    const double* clip_vals,
    double* out_sums,
    double* out_sum_sq,
    int n_threads
) {
    if (n_cols == 0 || n_rows == 0) return;
    if (n_threads <= 0) n_threads = omp_get_max_threads();
    std::memset(out_sums, 0, n_cols * sizeof(double));
    std::memset(out_sum_sq, 0, n_cols * sizeof(double));
    #pragma omp parallel num_threads(n_threads)
    {
        std::vector<double> local_sums(n_cols, 0.0);
        std::vector<double> local_sum_sq(n_cols, 0.0);
        #pragma omp for schedule(dynamic, 128)
        for (std::ptrdiff_t rr = 0; rr < static_cast<std::ptrdiff_t>(n_rows); ++rr) {
            const size_t row = static_cast<size_t>(rr);
            const int64_t p_start = row_ptr[row], p_end = row_ptr[row + 1];
            for (int64_t p = p_start; p < p_end; ++p) {
                const int64_t col_idx = col_indices[p];
                if unlikely_(col_idx < 0 || static_cast<size_t>(col_idx) >= n_cols) continue;
                const size_t col = static_cast<size_t>(col_idx);
                const float val = data[p], clip = static_cast<float>(clip_vals[col]);
                const float clipped = (val < clip) ? val : clip;
                local_sums[col] += static_cast<double>(clipped);
                local_sum_sq[col] += static_cast<double>(clipped * clipped);
            }
        }
        #pragma omp critical
        {
            for (size_t c = 0; c < n_cols; ++c) {
                out_sums[c] += local_sums[c];
                out_sum_sq[c] += local_sum_sq[c];
            }
        }
    }
}

// ================================================================
// High-level C++ Interface - CSC View
// ================================================================

template<class T>
std::pair<std::vector<double>, std::vector<double>>
sparse_clipped_moments(
    const view::CscView<T>& A,
    const double* clip_vals,
    int n_threads
) {
    const size_t n_cols = A.cols();
    std::vector<double> sums(n_cols), sum_sq(n_cols);
    if constexpr (std::is_same_v<T, double>) {
        sparse_clipped_moments_csc(A.data(), A.indices(), A.indptr(), n_cols, clip_vals, sums.data(), sum_sq.data(), n_threads);
    } else if constexpr (std::is_same_v<T, float>) {
        sparse_clipped_moments_csc_f32(A.data(), A.indices(), A.indptr(), n_cols, clip_vals, sums.data(), sum_sq.data(), n_threads);
    } else {
        std::vector<double> data_d(A.nnz());
        for (size_t i = 0; i < A.nnz(); ++i) { data_d[i] = static_cast<double>(A.data()[i]); }
        sparse_clipped_moments_csc(data_d.data(), A.indices(), A.indptr(), n_cols, clip_vals, sums.data(), sum_sq.data(), n_threads);
    }
    return {sums, sum_sq};
}

// Explicit Instantiations
template std::pair<std::vector<double>, std::vector<double>> sparse_clipped_moments<double>(const view::CscView<double>&, const double*, int);
template std::pair<std::vector<double>, std::vector<double>> sparse_clipped_moments<float>(const view::CscView<float>&, const double*, int);

} // namespace hvg
} // namespace kernel
} // namespace perturblab
