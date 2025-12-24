// sparse_clipped_moments.hpp - Sparse Clipped Moments operator
// Core operator for Seurat V3 HVG detection
#pragma once

#include "config.hpp"
#include "macro.hpp"
#include "sparse.hpp"
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace perturblab {
namespace kernel {
namespace hvg {

/**
 * @brief Compute clipped moments for a sparse CSC matrix.
 * 
 * This is the core operator for Seurat V3 highly variable gene (HVG) detection.
 * For each column (gene) j, it calculates:
 *   sum_j = Σ_i min(X_ij, clip_j)
 *   sum_sq_j = Σ_i (min(X_ij, clip_j))^2
 * 
 * Algorithm:
 * 1. Parallelize over columns to avoid race conditions.
 * 2. Vectorize the clipping operation using Highway SIMD.
 * 
 * @param data Non-zero values (nnz,).
 * @param row_indices Row indices (nnz,).
 * @param col_ptr Column pointers (n_cols + 1,).
 * @param n_cols Number of columns.
 * @param clip_vals Clipping thresholds per column (n_cols,).
 * @param out_sums Output: Clipped sums per column (n_cols,).
 * @param out_sum_sq Output: Clipped squared sums per column (n_cols,).
 * @param n_threads Number of threads.
 */
void sparse_clipped_moments_csc(
    const double* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_cols,
    const double* clip_vals,
    double* out_sums,
    double* out_sum_sq,
    int n_threads = 0
);

/**
 * @brief Float32 version of sparse_clipped_moments_csc.
 */
void sparse_clipped_moments_csc_f32(
    const float* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_cols,
    const double* clip_vals,
    double* out_sums,
    double* out_sum_sq,
    int n_threads = 0
);

/**
 * @brief Compute clipped moments for a sparse CSR matrix.
 * 
 * Strategy:
 * - If nnz is small: Transpose to CSC and use the CSC version.
 * - If nnz is large: Use thread-local buffers to avoid atomic contention.
 * 
 * @param data Non-zero values.
 * @param col_indices Column indices.
 * @param row_ptr Row pointers.
 * @param n_rows Number of rows.
 * @param n_cols Number of columns.
 * @param nnz Number of non-zero elements.
 * @param clip_vals Clipping thresholds per column.
 * @param out_sums Output: Clipped sums per column.
 * @param out_sum_sq Output: Clipped squared sums per column.
 * @param n_threads Number of threads.
 */
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
    int n_threads = 0
);

/**
 * @brief Float32 version of sparse_clipped_moments_csr.
 */
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
    int n_threads = 0
);

/**
 * @brief High-level C++ interface using CSC views.
 * 
 * @tparam T Numeric type.
 * @return pair<sums, sum_squares>
 */
template<class T>
std::pair<std::vector<double>, std::vector<double>>
sparse_clipped_moments(
    const view::CscView<T>& A,
    const double* clip_vals,
    int n_threads = 0
);

} // namespace hvg
} // namespace kernel
} // namespace perturblab
