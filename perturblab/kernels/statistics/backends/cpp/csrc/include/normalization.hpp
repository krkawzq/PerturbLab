// normalization.hpp - Header for normalization operations
#pragma once

#include "config.hpp"
#include <cstddef>
#include <cstdint>

namespace perturblab {
namespace kernel {
namespace normalization {

// ================================================================
// CSR Row Sum Operations
// ================================================================

/**
 * @brief Compute sum of each row in CSR sparse matrix (SIMD optimized).
 * 
 * @param data CSR data array.
 * @param indptr CSR index pointer array (length: n_rows + 1).
 * @param n_rows Number of rows.
 * @param out_sums Output array for row sums (length: n_rows).
 * @param n_threads Number of threads (0 = auto).
 */
void sparse_row_sum_csr(
    const double* data,
    const int64_t* indptr,
    size_t n_rows,
    double* out_sums,
    int n_threads = 0
);

/**
 * @brief Float32 version of sparse_row_sum_csr.
 */
void sparse_row_sum_csr_f32(
    const float* data,
    const int64_t* indptr,
    size_t n_rows,
    float* out_sums,
    int n_threads = 0
);

// ================================================================
// In-place Division Operations
// ================================================================

/**
 * @brief Divide each row by a scalar in-place (CSR format).
 * 
 * Performs: data[row_i] /= divisors[i] for each row i.
 * 
 * @param data CSR data array (modified in-place).
 * @param indptr CSR index pointer array (length: n_rows + 1).
 * @param n_rows Number of rows.
 * @param divisors Array of divisors for each row (length: n_rows).
 * @param allow_zero_divisor If true, skip division for zero divisors; if false, set to 0.
 * @param n_threads Number of threads (0 = auto).
 */
void inplace_divide_csr_rows(
    double* data,
    const int64_t* indptr,
    size_t n_rows,
    const double* divisors,
    bool allow_zero_divisor = false,
    int n_threads = 0
);

/**
 * @brief Float32 version of inplace_divide_csr_rows.
 */
void inplace_divide_csr_rows_f32(
    float* data,
    const int64_t* indptr,
    size_t n_rows,
    const float* divisors,
    bool allow_zero_divisor = false,
    int n_threads = 0
);

// ================================================================
// Utility Functions
// ================================================================

/**
 * @brief Compute median of non-zero values.
 * 
 * @param values Array of values.
 * @param n Length of array.
 * @return Median of non-zero values.
 */
double compute_median_nonzero(
    const double* values,
    size_t n
);

// ================================================================
// Exclude Highly Expressed Genes
// ================================================================

/**
 * @brief Find genes that are highly expressed in at least one cell.
 * 
 * A gene is considered highly expressed in a cell if its expression
 * is greater than max_fraction * total_counts_in_cell.
 * 
 * @param data CSR data array.
 * @param indptr CSR index pointer array (length: n_rows + 1).
 * @param indices CSR indices array (column indices).
 * @param n_rows Number of rows (cells).
 * @param n_cols Number of columns (genes).
 * @param row_sums Total counts per cell (length: n_rows).
 * @param max_fraction Threshold fraction (e.g., 0.05).
 * @param out_gene_mask Output boolean mask (length: n_cols). True = highly expressed.
 * @param n_threads Number of threads (0 = auto).
 */
void find_highly_expressed_genes(
    const double* data,
    const int64_t* indptr,
    const int64_t* indices,
    size_t n_rows,
    size_t n_cols,
    const double* row_sums,
    double max_fraction,
    bool* out_gene_mask,
    int n_threads = 0
);

/**
 * @brief Compute row sums excluding specific genes.
 * 
 * @param data CSR data array.
 * @param indptr CSR index pointer array (length: n_rows + 1).
 * @param indices CSR indices array (column indices).
 * @param n_rows Number of rows.
 * @param gene_mask Boolean mask indicating genes to exclude (length: n_cols).
 * @param out_sums Output array for row sums (length: n_rows).
 * @param n_threads Number of threads (0 = auto).
 */
void sparse_row_sum_csr_exclude_genes(
    const double* data,
    const int64_t* indptr,
    const int64_t* indices,
    size_t n_rows,
    const bool* gene_mask,
    double* out_sums,
    int n_threads = 0
);

} // namespace normalization
} // namespace kernel
} // namespace perturblab

