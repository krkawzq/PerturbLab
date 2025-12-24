// clip_matrix.cpp
// Dense matrix clipping by column
//
// Implements column-wise clipping for dense matrices using SIMD instructions.
// Useful for Seurat V3 HVG detection on dense data.
//
// Key optimizations:
// - Highway SIMD for vectorized min operations
// - Row-wise processing with SIMD (better cache locality)
// - OpenMP parallelization
//
#include "clip_matrix.hpp"
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
// Clip Matrix - Double Precision
// ================================================================

void clip_matrix_by_column(
    double* data,
    size_t n_rows,
    size_t n_cols,
    const double* clip_vals,
    int n_threads
) {
    if (n_rows == 0 || n_cols == 0) return;

    if (n_threads <= 0) n_threads = omp_get_max_threads();
    n_threads = std::min(n_threads, omp_get_max_threads());

    #pragma omp parallel num_threads(n_threads)
    {
        using D = HWY_FULL(double);
        D d;
        const size_t vec_lanes = Lanes(d);
        #pragma omp for schedule(static)
        for (std::ptrdiff_t rr = 0; rr < static_cast<std::ptrdiff_t>(n_rows); ++rr) {
            const size_t row = static_cast<size_t>(rr);
            double* row_data = data + row * n_cols;
            size_t col = 0;
            for (; col + vec_lanes <= n_cols; col += vec_lanes) {
                auto v_data = Load(d, row_data + col);
                auto v_clip = Load(d, clip_vals + col);
                auto v_clipped = Min(v_data, v_clip);
                Store(v_clipped, d, row_data + col);
            }
            for (; col < n_cols; ++col) {
                double val = row_data[col];
                double clip = clip_vals[col];
                row_data[col] = (val < clip) ? val : clip;
            }
        }
    }
}

void clip_matrix_by_column_f32(
    float* data,
    size_t n_rows,
    size_t n_cols,
    const float* clip_vals,
    int n_threads
) {
    if (n_rows == 0 || n_cols == 0) return;

    if (n_threads <= 0) n_threads = omp_get_max_threads();
    n_threads = std::min(n_threads, omp_get_max_threads());

    #pragma omp parallel num_threads(n_threads)
    {
        using D = HWY_FULL(float);
        D d;
        const size_t vec_lanes = Lanes(d);
        #pragma omp for schedule(static)
        for (std::ptrdiff_t rr = 0; rr < static_cast<std::ptrdiff_t>(n_rows); ++rr) {
            const size_t row = static_cast<size_t>(rr);
            float* row_data = data + row * n_cols;
            size_t col = 0;
            for (; col + vec_lanes <= n_cols; col += vec_lanes) {
                auto v_data = Load(d, row_data + col);
                auto v_clip = Load(d, clip_vals + col);
                auto v_clipped = Min(v_data, v_clip);
                Store(v_clipped, d, row_data + col);
            }
            for (; col < n_cols; ++col) {
                float val = row_data[col];
                float clip = clip_vals[col];
                row_data[col] = (val < clip) ? val : clip;
            }
        }
    }
}

// ================================================================
// Clip and Sum - Double Precision
// ================================================================

void clip_matrix_and_sum(
    const double* data,
    size_t n_rows,
    size_t n_cols,
    const double* clip_vals,
    double* out_sums,
    double* out_sum_sq,
    int n_threads
) {
    if (n_rows == 0 || n_cols == 0) return;
    if (n_threads <= 0) n_threads = omp_get_max_threads();
    std::memset(out_sums, 0, n_cols * sizeof(double));
    std::memset(out_sum_sq, 0, n_cols * sizeof(double));
    #pragma omp parallel num_threads(n_threads)
    {
        using D = HWY_FULL(double);
        D d;
        const size_t vec_lanes = Lanes(d);
        std::vector<double> local_sums(n_cols, 0.0);
        std::vector<double> local_sum_sq(n_cols, 0.0);
        #pragma omp for schedule(static)
        for (std::ptrdiff_t rr = 0; rr < static_cast<std::ptrdiff_t>(n_rows); ++rr) {
            const size_t row = static_cast<size_t>(rr);
            const double* row_data = data + row * n_cols;
            size_t col = 0;
            for (; col + vec_lanes <= n_cols; col += vec_lanes) {
                auto v_data = Load(d, row_data + col);
                auto v_clip = Load(d, clip_vals + col);
                auto v_clipped = Min(v_data, v_clip);
                auto v_sums = Load(d, local_sums.data() + col);
                auto v_sum_sq = Load(d, local_sum_sq.data() + col);
                v_sums = Add(v_sums, v_clipped);
                v_sum_sq = Add(v_sum_sq, Mul(v_clipped, v_clipped));
                Store(v_sums, d, local_sums.data() + col);
                Store(v_sum_sq, d, local_sum_sq.data() + col);
            }
            for (; col < n_cols; ++col) {
                double val = row_data[col], clip = clip_vals[col];
                double clipped = (val < clip) ? val : clip;
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

void clip_matrix_and_sum_f32(
    const float* data,
    size_t n_rows,
    size_t n_cols,
    const float* clip_vals,
    double* out_sums,
    double* out_sum_sq,
    int n_threads
) {
    if (n_rows == 0 || n_cols == 0) return;
    if (n_threads <= 0) n_threads = omp_get_max_threads();
    std::memset(out_sums, 0, n_cols * sizeof(double));
    std::memset(out_sum_sq, 0, n_cols * sizeof(double));
    #pragma omp parallel num_threads(n_threads)
    {
        using D = HWY_FULL(float);
        D d;
        const size_t vec_lanes = Lanes(d);
        std::vector<double> local_sums(n_cols, 0.0);
        std::vector<double> local_sum_sq(n_cols, 0.0);
        #pragma omp for schedule(static)
        for (std::ptrdiff_t rr = 0; rr < static_cast<std::ptrdiff_t>(n_rows); ++rr) {
            const size_t row = static_cast<size_t>(rr);
            const float* row_data = data + row * n_cols;
            size_t col = 0;
            for (; col + vec_lanes <= n_cols; col += vec_lanes) {
                auto v_data = Load(d, row_data + col);
                auto v_clip = Load(d, clip_vals + col);
                auto v_clipped = Min(v_data, v_clip);
                alignas(64) float clipped_arr[16];
                Store(v_clipped, d, clipped_arr);
                for (size_t i = 0; i < vec_lanes && (col + i) < n_cols; ++i) {
                    double val_d = static_cast<double>(clipped_arr[i]);
                    local_sums[col + i] += val_d;
                    local_sum_sq[col + i] += val_d * val_d;
                }
            }
            for (; col < n_cols; ++col) {
                float val = row_data[col], clip = clip_vals[col];
                double clipped = static_cast<double>((val < clip) ? val : clip);
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

} // namespace hvg
} // namespace kernel
} // namespace perturblab
