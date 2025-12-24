// sparse_mean_var.cpp
// Sparse matrix mean and variance computation
//
// Implements Welford's online algorithm for numerically stable computation
// of mean and variance in a single pass over the data.
//
// Algorithm: Welford's online algorithm
//   mean_n = mean_{n-1} + (x_n - mean_{n-1}) / n
//   M2_n = M2_{n-1} + (x_n - mean_{n-1}) * (x_n - mean_n)
//   var = M2_n / (n - ddof)
//
#include "sparse_mean_var.hpp"
#include "common.hpp"
#include "macro.hpp"
#include <cstring>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_max_threads() { return 1; }
#endif

namespace perturblab {
namespace kernel {
namespace stats {

// ================================================================
// Mean Only - Double Precision
// ================================================================

void sparse_mean_csc(
    const double* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    double* out_means,
    bool include_zeros,
    int n_threads
) {
    if (n_cols == 0) return;

    if (n_threads <= 0) n_threads = omp_get_max_threads();
    n_threads = std::min(n_threads, omp_get_max_threads());

    #pragma omp parallel num_threads(n_threads)
    {
        #pragma omp for schedule(dynamic, 64)
        for (std::ptrdiff_t cc = 0; cc < static_cast<std::ptrdiff_t>(n_cols); ++cc) {
            const size_t col = static_cast<size_t>(cc);
            const int64_t p_start = col_ptr[col], p_end = col_ptr[col + 1];
            const int64_t nnz_col = p_end - p_start;
            if (nnz_col == 0) { out_means[col] = 0.0; continue; }
            double sum = 0.0;
            for (int64_t p = p_start; p < p_end; ++p) {
                const double val = data[p];
                if likely_(is_valid_value(val)) { sum += val; }
            }
            if (include_zeros) { out_means[col] = sum / static_cast<double>(n_rows); }
            else { out_means[col] = (nnz_col > 0) ? (sum / static_cast<double>(nnz_col)) : 0.0; }
        }
    }
}

void sparse_mean_csc_f32(
    const float* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    double* out_means,
    bool include_zeros,
    int n_threads
) {
    if (n_cols == 0) return;

    if (n_threads <= 0) n_threads = omp_get_max_threads();
    n_threads = std::min(n_threads, omp_get_max_threads());

    #pragma omp parallel num_threads(n_threads)
    {
        #pragma omp for schedule(dynamic, 64)
        for (std::ptrdiff_t cc = 0; cc < static_cast<std::ptrdiff_t>(n_cols); ++cc) {
            const size_t col = static_cast<size_t>(cc);
            const int64_t p_start = col_ptr[col], p_end = col_ptr[col + 1];
            const int64_t nnz_col = p_end - p_start;
            if (nnz_col == 0) { out_means[col] = 0.0; continue; }
            double sum = 0.0;
            for (int64_t p = p_start; p < p_end; ++p) {
                const float val = data[p];
                if likely_(is_valid_value(static_cast<double>(val))) { sum += static_cast<double>(val); }
            }
            if (include_zeros) { out_means[col] = sum / static_cast<double>(n_rows); }
            else { out_means[col] = (nnz_col > 0) ? (sum / static_cast<double>(nnz_col)) : 0.0; }
        }
    }
}

// ================================================================
// Variance Only - Double Precision
// ================================================================

void sparse_var_csc(
    const double* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    double* out_vars,
    bool include_zeros,
    int ddof,
    int n_threads
) {
    if (n_cols == 0) return;

    if (n_threads <= 0) n_threads = omp_get_max_threads();
    n_threads = std::min(n_threads, omp_get_max_threads());

    #pragma omp parallel num_threads(n_threads)
    {
        #pragma omp for schedule(dynamic, 64)
        for (std::ptrdiff_t cc = 0; cc < static_cast<std::ptrdiff_t>(n_cols); ++cc) {
            const size_t col = static_cast<size_t>(cc);
            const int64_t p_start = col_ptr[col], p_end = col_ptr[col + 1];
            if (p_start == p_end) { out_vars[col] = 0.0; continue; }
            double mean = 0.0, M2 = 0.0; size_t count = 0;
            for (int64_t p = p_start; p < p_end; ++p) {
                const double val = data[p];
                if likely_(is_valid_value(val)) {
                    ++count;
                    double delta = val - mean;
                    mean += delta / static_cast<double>(count);
                    double delta2 = val - mean;
                    M2 += delta * delta2;
                }
            }
            if (include_zeros) {
                size_t n_zeros = n_rows - count;
                for (size_t i = 0; i < n_zeros; ++i) {
                    ++count;
                    double delta = -mean;
                    mean += delta / static_cast<double>(count);
                    double delta2 = -mean;
                    M2 += delta * delta2;
                }
            }
            int64_t n = static_cast<int64_t>(count) - ddof;
            out_vars[col] = (n > 0) ? (M2 / static_cast<double>(n)) : 0.0;
        }
    }
}

void sparse_var_csc_f32(
    const float* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    double* out_vars,
    bool include_zeros,
    int ddof,
    int n_threads
) {
    if (n_cols == 0) return;

    if (n_threads <= 0) n_threads = omp_get_max_threads();
    n_threads = std::min(n_threads, omp_get_max_threads());

    #pragma omp parallel num_threads(n_threads)
    {
        #pragma omp for schedule(dynamic, 64)
        for (std::ptrdiff_t cc = 0; cc < static_cast<std::ptrdiff_t>(n_cols); ++cc) {
            const size_t col = static_cast<size_t>(cc);
            const int64_t p_start = col_ptr[col], p_end = col_ptr[col + 1];
            if (p_start == p_end) { out_vars[col] = 0.0; continue; }
            double mean = 0.0, M2 = 0.0; size_t count = 0;
            for (int64_t p = p_start; p < p_end; ++p) {
                const double val = static_cast<double>(data[p]);
                if likely_(is_valid_value(val)) {
                    ++count;
                    double delta = val - mean;
                    mean += delta / static_cast<double>(count);
                    double delta2 = val - mean;
                    M2 += delta * delta2;
                }
            }
            if (include_zeros) {
                size_t n_zeros = n_rows - count;
                for (size_t i = 0; i < n_zeros; ++i) {
                    ++count;
                    double delta = -mean;
                    mean += delta / static_cast<double>(count);
                    double delta2 = -mean;
                    M2 += delta * delta2;
                }
            }
            int64_t n = static_cast<int64_t>(count) - ddof;
            out_vars[col] = (n > 0) ? (M2 / static_cast<double>(n)) : 0.0;
        }
    }
}

// ================================================================
// Mean and Variance - Double Precision (Single Pass)
// ================================================================

void sparse_mean_var_csc(
    const double* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    double* out_means,
    double* out_vars,
    bool include_zeros,
    int ddof,
    int n_threads
) {
    if (n_cols == 0) return;

    if (n_threads <= 0) n_threads = omp_get_max_threads();
    n_threads = std::min(n_threads, omp_get_max_threads());

    #pragma omp parallel num_threads(n_threads)
    {
        #pragma omp for schedule(dynamic, 64)
        for (std::ptrdiff_t cc = 0; cc < static_cast<std::ptrdiff_t>(n_cols); ++cc) {
            const size_t col = static_cast<size_t>(cc);
            const int64_t p_start = col_ptr[col], p_end = col_ptr[col + 1];
            if (p_start == p_end) { out_means[col] = 0.0; out_vars[col] = 0.0; continue; }
            double mean = 0.0, M2 = 0.0; size_t count = 0;
            for (int64_t p = p_start; p < p_end; ++p) {
                const double val = data[p];
                if likely_(is_valid_value(val)) {
                    ++count;
                    double delta = val - mean;
                    mean += delta / static_cast<double>(count);
                    double delta2 = val - mean;
                    M2 += delta * delta2;
                }
            }
            if (include_zeros) {
                size_t n_zeros = n_rows - count;
                for (size_t i = 0; i < n_zeros; ++i) {
                    ++count;
                    double delta = -mean;
                    mean += delta / static_cast<double>(count);
                    double delta2 = -mean;
                    M2 += delta * delta2;
                }
            }
            out_means[col] = mean;
            int64_t n = static_cast<int64_t>(count) - ddof;
            out_vars[col] = (n > 0) ? (M2 / static_cast<double>(n)) : 0.0;
        }
    }
}

void sparse_mean_var_csc_f32(
    const float* data,
    const int64_t* row_indices,
    const int64_t* col_ptr,
    size_t n_rows,
    size_t n_cols,
    double* out_means,
    double* out_vars,
    bool include_zeros,
    int ddof,
    int n_threads
) {
    if (n_cols == 0) return;

    if (n_threads <= 0) n_threads = omp_get_max_threads();
    n_threads = std::min(n_threads, omp_get_max_threads());

    #pragma omp parallel num_threads(n_threads)
    {
        #pragma omp for schedule(dynamic, 64)
        for (std::ptrdiff_t cc = 0; cc < static_cast<std::ptrdiff_t>(n_cols); ++cc) {
            const size_t col = static_cast<size_t>(cc);
            const int64_t p_start = col_ptr[col], p_end = col_ptr[col + 1];
            if (p_start == p_end) { out_means[col] = 0.0; out_vars[col] = 0.0; continue; }
            double mean = 0.0, M2 = 0.0; size_t count = 0;
            for (int64_t p = p_start; p < p_end; ++p) {
                const double val = static_cast<double>(data[p]);
                if likely_(is_valid_value(val)) {
                    ++count;
                    double delta = val - mean;
                    mean += delta / static_cast<double>(count);
                    double delta2 = val - mean;
                    M2 += delta * delta2;
                }
            }
            if (include_zeros) {
                size_t n_zeros = n_rows - count;
                for (size_t i = 0; i < n_zeros; ++i) {
                    ++count;
                    double delta = -mean;
                    mean += delta / static_cast<double>(count);
                    double delta2 = -mean;
                    M2 += delta * delta2;
                }
            }
            out_means[col] = mean;
            int64_t n = static_cast<int64_t>(count) - ddof;
            out_vars[col] = (n > 0) ? (M2 / static_cast<double>(n)) : 0.0;
        }
    }
}

// ================================================================
// High-level C++ Interface
// ================================================================

template<class T>
std::pair<std::vector<double>, std::vector<double>>
sparse_mean_var(
    const view::CscView<T>& A,
    bool include_zeros,
    int ddof,
    int n_threads
) {
    const size_t n_cols = A.cols();
    const size_t n_rows = A.rows();
    std::vector<double> means(n_cols);
    std::vector<double> vars(n_cols);
    if constexpr (std::is_same_v<T, double>) {
        sparse_mean_var_csc(A.data(), A.indices(), A.indptr(), n_rows, n_cols, means.data(), vars.data(), include_zeros, ddof, n_threads);
    } else if constexpr (std::is_same_v<T, float>) {
        sparse_mean_var_csc_f32(A.data(), A.indices(), A.indptr(), n_rows, n_cols, means.data(), vars.data(), include_zeros, ddof, n_threads);
    } else {
        std::vector<double> data_d(A.nnz());
        for (size_t i = 0; i < A.nnz(); ++i) { data_d[i] = static_cast<double>(A.data()[i]); }
        sparse_mean_var_csc(data_d.data(), A.indices(), A.indptr(), n_rows, n_cols, means.data(), vars.data(), include_zeros, ddof, n_threads);
    }
    return {means, vars};
}

// Explicit Instantiations
template std::pair<std::vector<double>, std::vector<double>> sparse_mean_var<double>(const view::CscView<double>&, bool, int, int);
template std::pair<std::vector<double>, std::vector<double>> sparse_mean_var<float>(const view::CscView<float>&, bool, int, int);

} // namespace stats
} // namespace kernel
} // namespace perturblab
