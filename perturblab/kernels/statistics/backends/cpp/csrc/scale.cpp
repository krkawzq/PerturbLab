// scale.cpp - Scaling and standardization operations for sparse matrices
// Implements z-score standardization and clipping

#include "include/scale.hpp"
#include "include/common.hpp"
#include "include/simd.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace perturblab {
namespace kernel {
namespace scale {

// ================================================================
// CSC Standardization (Column-wise)
// ================================================================

void sparse_standardize_csc(
    double* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    const double* means,
    const double* stds,
    bool zero_center,
    double max_value,
    int n_threads
) {
    if (n_threads <= 0) {
        n_threads = omp_get_max_threads();
    }
    n_threads = std::min(n_threads, omp_get_max_threads());
    
    #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 64)
    for (size_t j = 0; j < n_cols; ++j) {
        int64_t start = col_ptr[j];
        int64_t end = col_ptr[j + 1];
        
        double mean = means[j];
        double std = stds[j];
        
        // Skip if std is zero or invalid
        if (std <= 0.0 || !std::isfinite(std)) {
            // Set all values to zero
            for (int64_t idx = start; idx < end; ++idx) {
                data[idx] = 0.0;
            }
            continue;
        }
        
        double inv_std = 1.0 / std;
        
        // Standardize: (x - mean) / std
        for (int64_t idx = start; idx < end; ++idx) {
            double val = data[idx];
            
            if (zero_center) {
                val = (val - mean) * inv_std;
            } else {
                val = val * inv_std;
            }
            
            // Apply clipping if specified
            if (max_value > 0.0) {
                if (val > max_value) {
                    val = max_value;
                } else if (zero_center && val < -max_value) {
                    val = -max_value;
                }
            }
            
            data[idx] = val;
        }
    }
}

// Float32 version
void sparse_standardize_csc_f32(
    float* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    const float* means,
    const float* stds,
    bool zero_center,
    float max_value,
    int n_threads
) {
    if (n_threads <= 0) {
        n_threads = omp_get_max_threads();
    }
    n_threads = std::min(n_threads, omp_get_max_threads());
    
    #pragma omp parallel for num_threads(n_threads) schedule(dynamic, 64)
    for (size_t j = 0; j < n_cols; ++j) {
        int64_t start = col_ptr[j];
        int64_t end = col_ptr[j + 1];
        
        float mean = means[j];
        float std = stds[j];
        
        if (std <= 0.0f || !std::isfinite(std)) {
            for (int64_t idx = start; idx < end; ++idx) {
                data[idx] = 0.0f;
            }
            continue;
        }
        
        float inv_std = 1.0f / std;
        
        for (int64_t idx = start; idx < end; ++idx) {
            float val = data[idx];
            
            if (zero_center) {
                val = (val - mean) * inv_std;
            } else {
                val = val * inv_std;
            }
            
            if (max_value > 0.0f) {
                if (val > max_value) {
                    val = max_value;
                } else if (zero_center && val < -max_value) {
                    val = -max_value;
                }
            }
            
            data[idx] = val;
        }
    }
}

// ================================================================
// CSR Standardization (Row-wise)
// ================================================================

void sparse_standardize_csr(
    double* data,
    const int64_t* col_indices,
    const int64_t* row_ptr,
    size_t n_rows,
    size_t n_cols,
    const double* means,
    const double* stds,
    bool zero_center,
    double max_value,
    int n_threads
) {
    if (n_threads <= 0) {
        n_threads = omp_get_max_threads();
    }
    n_threads = std::min(n_threads, omp_get_max_threads());
    
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (size_t i = 0; i < n_rows; ++i) {
        int64_t start = row_ptr[i];
        int64_t end = row_ptr[i + 1];
        
        for (int64_t idx = start; idx < end; ++idx) {
            int64_t col_idx = col_indices[idx];
            double val = data[idx];
            
            double mean = means[col_idx];
            double std = stds[col_idx];
            
            if (std <= 0.0 || !std::isfinite(std)) {
                data[idx] = 0.0;
                continue;
            }
            
            double inv_std = 1.0 / std;
            
            if (zero_center) {
                val = (val - mean) * inv_std;
            } else {
                val = val * inv_std;
            }
            
            if (max_value > 0.0) {
                if (val > max_value) {
                    val = max_value;
                } else if (zero_center && val < -max_value) {
                    val = -max_value;
                }
            }
            
            data[idx] = val;
        }
    }
}

// Float32 version
void sparse_standardize_csr_f32(
    float* data,
    const int64_t* col_indices,
    const int64_t* row_ptr,
    size_t n_rows,
    size_t n_cols,
    const float* means,
    const float* stds,
    bool zero_center,
    float max_value,
    int n_threads
) {
    if (n_threads <= 0) {
        n_threads = omp_get_max_threads();
    }
    n_threads = std::min(n_threads, omp_get_max_threads());
    
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (size_t i = 0; i < n_rows; ++i) {
        int64_t start = row_ptr[i];
        int64_t end = row_ptr[i + 1];
        
        for (int64_t idx = start; idx < end; ++idx) {
            int64_t col_idx = col_indices[idx];
            float val = data[idx];
            
            float mean = means[col_idx];
            float std = stds[col_idx];
            
            if (std <= 0.0f || !std::isfinite(std)) {
                data[idx] = 0.0f;
                continue;
            }
            
            float inv_std = 1.0f / std;
            
            if (zero_center) {
                val = (val - mean) * inv_std;
            } else {
                val = val * inv_std;
            }
            
            if (max_value > 0.0f) {
                if (val > max_value) {
                    val = max_value;
                } else if (zero_center && val < -max_value) {
                    val = -max_value;
                }
            }
            
            data[idx] = val;
        }
    }
}

// ================================================================
// Dense Matrix Standardization
// ================================================================

void dense_standardize(
    double* data,
    size_t n_rows,
    size_t n_cols,
    const double* means,
    const double* stds,
    bool zero_center,
    double max_value,
    int n_threads
) {
    if (n_threads <= 0) {
        n_threads = omp_get_max_threads();
    }
    n_threads = std::min(n_threads, omp_get_max_threads());
    
    // Column-major standardization
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (size_t j = 0; j < n_cols; ++j) {
        double mean = means[j];
        double std = stds[j];
        
        if (std <= 0.0 || !std::isfinite(std)) {
            for (size_t i = 0; i < n_rows; ++i) {
                data[i * n_cols + j] = 0.0;
            }
            continue;
        }
        
        double inv_std = 1.0 / std;
        
        for (size_t i = 0; i < n_rows; ++i) {
            size_t idx = i * n_cols + j;
            double val = data[idx];
            
            if (zero_center) {
                val = (val - mean) * inv_std;
            } else {
                val = val * inv_std;
            }
            
            if (max_value > 0.0) {
                if (val > max_value) {
                    val = max_value;
                } else if (zero_center && val < -max_value) {
                    val = -max_value;
                }
            }
            
            data[idx] = val;
        }
    }
}

// Float32 version
void dense_standardize_f32(
    float* data,
    size_t n_rows,
    size_t n_cols,
    const float* means,
    const float* stds,
    bool zero_center,
    float max_value,
    int n_threads
) {
    if (n_threads <= 0) {
        n_threads = omp_get_max_threads();
    }
    n_threads = std::min(n_threads, omp_get_max_threads());
    
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (size_t j = 0; j < n_cols; ++j) {
        float mean = means[j];
        float std = stds[j];
        
        if (std <= 0.0f || !std::isfinite(std)) {
            for (size_t i = 0; i < n_rows; ++i) {
                data[i * n_cols + j] = 0.0f;
            }
            continue;
        }
        
        float inv_std = 1.0f / std;
        
        for (size_t i = 0; i < n_rows; ++i) {
            size_t idx = i * n_cols + j;
            float val = data[idx];
            
            if (zero_center) {
                val = (val - mean) * inv_std;
            } else {
                val = val * inv_std;
            }
            
            if (max_value > 0.0f) {
                if (val > max_value) {
                    val = max_value;
                } else if (zero_center && val < -max_value) {
                    val = -max_value;
                }
            }
            
            data[idx] = val;
        }
    }
}

} // namespace scale
} // namespace kernel
} // namespace perturblab

