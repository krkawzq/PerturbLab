// clip_matrix.hpp - Dense matrix clipping by column
// SIMD-optimized column-wise clipping
#pragma once

#include "config.hpp"
#include "macro.hpp"
#include <cstddef>
#include <cstdint>

namespace perturblab {
namespace kernel {
namespace hvg {

/**
 * @brief Column-wise clipping of a dense matrix: X_ij' = min(X_ij, clip_j)
 * 
 * Uses Highway SIMD for acceleration. Supports in-place operations.
 * 
 * @param data Input/output matrix data (row-major).
 * @param n_rows Number of rows.
 * @param n_cols Number of columns.
 * @param clip_vals Clipping thresholds per column [n_cols].
 * @param n_threads Number of threads.
 */
void clip_matrix_by_column(
    double* data,
    size_t n_rows,
    size_t n_cols,
    const double* clip_vals,
    int n_threads = 0
);

/**
 * @brief Float32 version of clip_matrix_by_column.
 */
void clip_matrix_by_column_f32(
    float* data,
    size_t n_rows,
    size_t n_cols,
    const float* clip_vals,
    int n_threads = 0
);

/**
 * @brief Clip a dense matrix and calculate sums and squared sums per column.
 * 
 * Combined operation to minimize data passes during HVG detection.
 * 
 * @param data Input matrix data.
 * @param n_rows Number of rows.
 * @param n_cols Number of columns.
 * @param clip_vals Clipping thresholds per column.
 * @param out_sums Output sums [n_cols].
 * @param out_sum_sq Output squared sums [n_cols].
 * @param n_threads Number of threads.
 */
void clip_matrix_and_sum(
    const double* data,
    size_t n_rows,
    size_t n_cols,
    const double* clip_vals,
    double* out_sums,
    double* out_sum_sq,
    int n_threads = 0
);

/**
 * @brief Float32 version of clip_matrix_and_sum.
 */
void clip_matrix_and_sum_f32(
    const float* data,
    size_t n_rows,
    size_t n_cols,
    const float* clip_vals,
    double* out_sums,
    double* out_sum_sq,
    int n_threads = 0
);

} // namespace hvg
} // namespace kernel
} // namespace perturblab
