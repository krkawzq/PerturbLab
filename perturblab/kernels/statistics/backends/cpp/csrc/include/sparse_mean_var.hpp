// sparse_mean_var.hpp - Sparse matrix mean and variance computation
// Efficient algorithms for single-pass computation
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
namespace stats {

/**
 * @brief Compute mean of each column for a sparse CSC matrix.
 * 
 * Algorithm:
 * mean = (sum of non-zero elements) / n_rows (if include_zeros is true)
 * 
 * @param data Non-zero values.
 * @param row_indices Row indices.
 * @param col_ptr Column pointers.
 * @param n_rows Total number of rows.
 * @param n_cols Total number of columns.
 * @param out_means Output array [n_cols].
 * @param include_zeros Whether to include implicit zeros in denominator.
 * @param n_threads Number of threads.
 */
void sparse_mean_csc(
    const double* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    double* out_means,
    bool include_zeros = true,
    int n_threads = 0
);

/**
 * @brief Float32 version of sparse_mean_csc.
 */
void sparse_mean_csc_f32(
    const float* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    double* out_means,
    bool include_zeros = true,
    int n_threads = 0
);

/**
 * @brief Compute variance of each column for a sparse CSC matrix.
 * 
 * Algorithm: Uses Welford's online algorithm or two-pass approach for stability.
 * var = (sum(x^2) - (sum(x)^2 / n)) / (n - ddof)
 * 
 * @param data Non-zero values.
 * @param row_indices Row indices.
 * @param col_ptr Column pointers.
 * @param n_rows Total number of rows.
 * @param n_cols Total number of columns.
 * @param out_vars Output array [n_cols].
 * @param include_zeros Whether to include implicit zeros.
 * @param ddof Delta Degrees of Freedom (0 for population, 1 for sample).
 * @param n_threads Number of threads.
 */
void sparse_var_csc(
    const double* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    double* out_vars,
    bool include_zeros = true,
    int ddof = 1,
    int n_threads = 0
);

/**
 * @brief Float32 version of sparse_var_csc.
 */
void sparse_var_csc_f32(
    const float* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    double* out_vars,
    bool include_zeros = true,
    int ddof = 1,
    int n_threads = 0
);

/**
 * @brief Compute both mean and variance of each column in a single pass.
 * 
 * This is the most efficient method for calculating both statistics simultaneously.
 * 
 * @param data Non-zero values.
 * @param row_indices Row indices.
 * @param col_ptr Column pointers.
 * @param n_rows Total number of rows.
 * @param n_cols Total number of columns.
 * @param out_means Output array [n_cols].
 * @param out_vars Output array [n_cols].
 * @param include_zeros Whether to include implicit zeros.
 * @param ddof Delta Degrees of Freedom.
 * @param n_threads Number of threads.
 */
void sparse_mean_var_csc(
    const double* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    double* out_means,
    double* out_vars,
    bool include_zeros = true,
    int ddof = 1,
    int n_threads = 0
);

/**
 * @brief Float32 version of sparse_mean_var_csc.
 */
void sparse_mean_var_csc_f32(
    const float* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    double* out_means,
    double* out_vars,
    bool include_zeros = true,
    int ddof = 1,
    int n_threads = 0
);

/**
 * @brief High-level C++ interface using CSC views.
 * 
 * @tparam T Numeric type.
 * @return pair<means, vars>
 */
template<class T>
std::pair<std::vector<double>, std::vector<double>>
sparse_mean_var(
    const view::CscView<T>& A,
    bool include_zeros = true,
    int ddof = 1,
    int n_threads = 0
);

} // namespace stats
} // namespace kernel
} // namespace perturblab
