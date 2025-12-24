// scale.hpp - Header for scaling and standardization operations
#pragma once

#include "config.hpp"
#include <cstddef>
#include <cstdint>

namespace perturblab {
namespace kernel {
namespace scale {

// ================================================================
// CSC Standardization (Column-wise)
// ================================================================

/**
 * @brief Standardize sparse CSC matrix by columns (genes).
 * 
 * Performs: X[:, j] = (X[:, j] - mean[j]) / std[j]
 * Optionally clips values to [-max_value, max_value].
 * 
 * @param data CSC data array (modified in-place).
 * @param row_indices CSC row indices array.
 * @param col_ptr CSC column pointer array (length: n_cols + 1).
 * @param n_rows Number of rows (cells).
 * @param n_cols Number of columns (genes).
 * @param means Array of column means (length: n_cols).
 * @param stds Array of column standard deviations (length: n_cols).
 * @param zero_center If true, subtract mean; if false, only divide by std.
 * @param max_value Maximum absolute value for clipping (0 = no clipping).
 * @param n_threads Number of threads (0 = auto).
 */
void sparse_standardize_csc(
    double* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    const double* means,
    const double* stds,
    bool zero_center = true,
    double max_value = 0.0,
    int n_threads = 0
);

/**
 * @brief Float32 version of sparse_standardize_csc.
 */
void sparse_standardize_csc_f32(
    float* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    const float* means,
    const float* stds,
    bool zero_center = true,
    float max_value = 0.0f,
    int n_threads = 0
);

// ================================================================
// CSR Standardization (Row-wise access, but column-wise standardization)
// ================================================================

/**
 * @brief Standardize sparse CSR matrix by columns (genes).
 * 
 * Same as sparse_standardize_csc but for CSR format.
 * Note: This is less efficient than CSC for column-wise operations.
 * 
 * @param data CSR data array (modified in-place).
 * @param col_indices CSR column indices array.
 * @param row_ptr CSR row pointer array (length: n_rows + 1).
 * @param n_rows Number of rows (cells).
 * @param n_cols Number of columns (genes).
 * @param means Array of column means (length: n_cols).
 * @param stds Array of column standard deviations (length: n_cols).
 * @param zero_center If true, subtract mean; if false, only divide by std.
 * @param max_value Maximum absolute value for clipping (0 = no clipping).
 * @param n_threads Number of threads (0 = auto).
 */
void sparse_standardize_csr(
    double* data,
    const int64_t* col_indices,
    const int64_t* row_ptr,
    size_t n_rows,
    size_t n_cols,
    const double* means,
    const double* stds,
    bool zero_center = true,
    double max_value = 0.0,
    int n_threads = 0
);

/**
 * @brief Float32 version of sparse_standardize_csr.
 */
void sparse_standardize_csr_f32(
    float* data,
    const int64_t* col_indices,
    const int64_t* row_ptr,
    size_t n_rows,
    size_t n_cols,
    const float* means,
    const float* stds,
    bool zero_center = true,
    float max_value = 0.0f,
    int n_threads = 0
);

// ================================================================
// Dense Matrix Standardization
// ================================================================

/**
 * @brief Standardize dense matrix by columns.
 * 
 * Assumes row-major layout: data[i * n_cols + j] = cell i, gene j.
 * 
 * @param data Dense data array (modified in-place).
 * @param n_rows Number of rows (cells).
 * @param n_cols Number of columns (genes).
 * @param means Array of column means (length: n_cols).
 * @param stds Array of column standard deviations (length: n_cols).
 * @param zero_center If true, subtract mean; if false, only divide by std.
 * @param max_value Maximum absolute value for clipping (0 = no clipping).
 * @param n_threads Number of threads (0 = auto).
 */
void dense_standardize(
    double* data,
    size_t n_rows,
    size_t n_cols,
    const double* means,
    const double* stds,
    bool zero_center = true,
    double max_value = 0.0,
    int n_threads = 0
);

/**
 * @brief Float32 version of dense_standardize.
 */
void dense_standardize_f32(
    float* data,
    size_t n_rows,
    size_t n_cols,
    const float* means,
    const float* stds,
    bool zero_center = true,
    float max_value = 0.0f,
    int n_threads = 0
);

} // namespace scale
} // namespace kernel
} // namespace perturblab

