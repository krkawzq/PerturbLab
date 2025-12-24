// normalization.cpp - Normalization operations for sparse matrices
// Implements CPM/TPM normalization and related operations

#include "include/normalization.hpp"
#include "include/common.hpp"
#include "include/simd.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace perturblab {
namespace kernel {
namespace normalization {

// ================================================================
// CSR Row Sum (SIMD Optimized)
// ================================================================

void sparse_row_sum_csr(
    const double* data,
    const int64_t* indptr,
    size_t n_rows,
    double* out_sums,
    int n_threads
) {
    if (n_threads <= 0) {
        n_threads = omp_get_max_threads();
    }
    n_threads = std::min(n_threads, omp_get_max_threads());
    
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (size_t i = 0; i < n_rows; ++i) {
        int64_t start = indptr[i];
        int64_t end = indptr[i + 1];
        int64_t nnz_row = end - start;
        
        if (nnz_row == 0) {
            out_sums[i] = 0.0;
            continue;
        }
        
        // Use Kahan summation for numerical stability
        double sum = 0.0;
        double c = 0.0;  // Compensation for lost low-order bits
        
        for (int64_t j = start; j < end; ++j) {
            double y = data[j] - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        
        out_sums[i] = sum;
    }
}

// Float32 version
void sparse_row_sum_csr_f32(
    const float* data,
    const int64_t* indptr,
    size_t n_rows,
    float* out_sums,
    int n_threads
) {
    if (n_threads <= 0) {
        n_threads = omp_get_max_threads();
    }
    n_threads = std::min(n_threads, omp_get_max_threads());
    
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (size_t i = 0; i < n_rows; ++i) {
        int64_t start = indptr[i];
        int64_t end = indptr[i + 1];
        int64_t nnz_row = end - start;
        
        if (nnz_row == 0) {
            out_sums[i] = 0.0f;
            continue;
        }
        
        // Kahan summation
        float sum = 0.0f;
        float c = 0.0f;
        
        for (int64_t j = start; j < end; ++j) {
            float y = data[j] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        
        out_sums[i] = sum;
    }
}

// ================================================================
// In-place Division by Row
// ================================================================

void inplace_divide_csr_rows(
    double* data,
    const int64_t* indptr,
    size_t n_rows,
    const double* divisors,
    bool allow_zero_divisor,
    int n_threads
) {
    if (n_threads <= 0) {
        n_threads = omp_get_max_threads();
    }
    n_threads = std::min(n_threads, omp_get_max_threads());
    
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (size_t i = 0; i < n_rows; ++i) {
        int64_t start = indptr[i];
        int64_t end = indptr[i + 1];
        
        double divisor = divisors[i];
        
        // Handle zero divisor
        if (divisor == 0.0) {
            if (!allow_zero_divisor) {
                // Set to zero or NaN based on preference
                for (int64_t j = start; j < end; ++j) {
                    data[j] = 0.0;
                }
            }
            continue;
        }
        
        double inv_divisor = 1.0 / divisor;
        
        // Vectorized multiplication (faster than division)
        int64_t j = start;
        int64_t nnz_row = end - start;
        
        // SIMD-friendly loop
        for (; j < end; ++j) {
            data[j] *= inv_divisor;
        }
    }
}

// Float32 version
void inplace_divide_csr_rows_f32(
    float* data,
    const int64_t* indptr,
    size_t n_rows,
    const float* divisors,
    bool allow_zero_divisor,
    int n_threads
) {
    if (n_threads <= 0) {
        n_threads = omp_get_max_threads();
    }
    n_threads = std::min(n_threads, omp_get_max_threads());
    
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (size_t i = 0; i < n_rows; ++i) {
        int64_t start = indptr[i];
        int64_t end = indptr[i + 1];
        
        float divisor = divisors[i];
        
        if (divisor == 0.0f) {
            if (!allow_zero_divisor) {
                for (int64_t j = start; j < end; ++j) {
                    data[j] = 0.0f;
                }
            }
            continue;
        }
        
        float inv_divisor = 1.0f / divisor;
        
        for (int64_t j = start; j < end; ++j) {
            data[j] *= inv_divisor;
        }
    }
}

// ================================================================
// Compute Median (for target_sum calculation)
// ================================================================

double compute_median_nonzero(
    const double* values,
    size_t n
) {
    if (n == 0) return 0.0;
    
    // Count non-zero values
    size_t n_nonzero = 0;
    for (size_t i = 0; i < n; ++i) {
        if (values[i] > 0.0) {
            n_nonzero++;
        }
    }
    
    if (n_nonzero == 0) return 0.0;
    
    // Copy non-zero values
    std::vector<double> nonzero_vals;
    nonzero_vals.reserve(n_nonzero);
    for (size_t i = 0; i < n; ++i) {
        if (values[i] > 0.0) {
            nonzero_vals.push_back(values[i]);
        }
    }
    
    // Sort and find median
    std::sort(nonzero_vals.begin(), nonzero_vals.end());
    
    size_t mid = n_nonzero / 2;
    if (n_nonzero % 2 == 0) {
        return (nonzero_vals[mid - 1] + nonzero_vals[mid]) / 2.0;
    } else {
        return nonzero_vals[mid];
    }
}

// ================================================================
// Exclude Highly Expressed Genes
// ================================================================

void find_highly_expressed_genes(
    const double* data,
    const int64_t* indptr,
    const int64_t* indices,
    size_t n_rows,
    size_t n_cols,
    const double* row_sums,
    double max_fraction,
    bool* out_gene_mask,
    int n_threads
) {
    if (n_threads <= 0) {
        n_threads = omp_get_max_threads();
    }
    n_threads = std::min(n_threads, omp_get_max_threads());
    
    // Initialize mask to false (not highly expressed)
    std::fill(out_gene_mask, out_gene_mask + n_cols, false);
    
    // Count how many cells each gene is highly expressed in
    std::vector<int> highly_expressed_count(n_cols, 0);
    
    #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 64)
    for (size_t i = 0; i < n_rows; ++i) {
        int64_t start = indptr[i];
        int64_t end = indptr[i + 1];
        double threshold = row_sums[i] * max_fraction;
        
        for (int64_t j = start; j < end; ++j) {
            if (data[j] > threshold) {
                int64_t col_idx = indices[j];
                #pragma omp atomic
                highly_expressed_count[col_idx]++;
            }
        }
    }
    
    // Mark genes that are highly expressed in at least one cell
    for (size_t j = 0; j < n_cols; ++j) {
        if (highly_expressed_count[j] > 0) {
            out_gene_mask[j] = true;
        }
    }
}

// ================================================================
// Recompute Row Sums Excluding Highly Expressed Genes
// ================================================================

void sparse_row_sum_csr_exclude_genes(
    const double* data,
    const int64_t* indptr,
    const int64_t* indices,
    size_t n_rows,
    const bool* gene_mask,
    double* out_sums,
    int n_threads
) {
    if (n_threads <= 0) {
        n_threads = omp_get_max_threads();
    }
    n_threads = std::min(n_threads, omp_get_max_threads());
    
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (size_t i = 0; i < n_rows; ++i) {
        int64_t start = indptr[i];
        int64_t end = indptr[i + 1];
        
        double sum = 0.0;
        double c = 0.0;  // Kahan summation
        
        for (int64_t j = start; j < end; ++j) {
            int64_t col_idx = indices[j];
            
            // Skip highly expressed genes
            if (gene_mask[col_idx]) {
                continue;
            }
            
            double y = data[j] - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        
        out_sums[i] = sum;
    }
}

} // namespace normalization
} // namespace kernel
} // namespace perturblab

