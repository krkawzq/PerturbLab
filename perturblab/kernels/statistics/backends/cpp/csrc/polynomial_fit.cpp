// polynomial_fit.cpp
// Polynomial regression for HVG detection
//
// Implements polynomial regression as a fast alternative to LOESS.
// For single-cell data, a 2nd-degree polynomial is typically sufficient.
//
// Algorithm: Ordinary Least Squares (OLS) and Locally Weighted Scatterplot Smoothing (LOESS)
//
#include "polynomial_fit.hpp"
#include "common.hpp"
#include "macro.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_max_threads() { return 1; }
#endif

namespace perturblab {
namespace kernel {
namespace hvg {

// ================================================================
// Linear Algebra Helpers
// ================================================================

namespace linalg {

/**
 * @brief Matrix multiplication: C = A * B (m×k) × (k×n) = (m×n).
 */
static inline void matmul(
    const double* A, const double* B, double* C,
    size_t m, size_t k, size_t n
) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/**
 * @brief Matrix transpose: B = A^T (m×n) -> (n×m).
 */
static inline void transpose(
    const double* A, double* B,
    size_t m, size_t n
) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            B[j * m + i] = A[i * n + j];
        }
    }
}

/**
 * @brief Cholesky decomposition: A = L * L^T.
 */
static inline bool cholesky(double* A, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = A[i * n + j];
            for (size_t k = 0; k < j; ++k) {
                sum -= A[i * n + k] * A[j * n + k];
            }
            if (i == j) {
                if (sum <= 0.0) return false;
                A[i * n + i] = std::sqrt(sum);
            } else {
                A[i * n + j] = sum / A[j * n + j];
            }
        }
    }
    return true;
}

static inline void solve_lower(
    const double* L, const double* b, double* y,
    size_t n
) {
    for (size_t i = 0; i < n; ++i) {
        double sum = b[i];
        for (size_t j = 0; j < i; ++j) {
            sum -= L[i * n + j] * y[j];
        }
        y[i] = sum / L[i * n + i];
    }
}

static inline void solve_upper_transpose(
    const double* L, const double* y, double* x,
    size_t n
) {
    for (int64_t i = n - 1; i >= 0; --i) {
        double sum = y[i];
        for (size_t j = i + 1; j < n; ++j) {
            sum -= L[j * n + i] * x[j];
        }
        x[i] = sum / L[i * n + i];
    }
}

/**
 * @brief Solve linear system A * x = b using Cholesky decomposition.
 */
static inline bool solve_cholesky(
    const double* A, const double* b, double* x,
    size_t n
) {
    std::vector<double> L(n * n);
    std::copy(A, A + n * n, L.begin());
    if (!cholesky(L.data(), n)) { return false; }
    std::vector<double> y(n);
    solve_lower(L.data(), b, y.data(), n);
    solve_upper_transpose(L.data(), y.data(), x, n);
    return true;
}

} // namespace linalg

// ================================================================
// Polynomial Regression
// ================================================================

void polynomial_fit(
    const double* x,
    const double* y,
    size_t n,
    int degree,
    double* out_fitted,
    double* out_coeffs
) {
    if (n == 0 || degree < 0) return;
    const size_t p = static_cast<size_t>(degree) + 1;
    if (n < p) {
        for (size_t i = 0; i < n; ++i) { out_fitted[i] = y[i]; }
        return;
    }
    std::vector<double> X(n * p);
    for (size_t i = 0; i < n; ++i) {
        double xi = x[i];
        double power = 1.0;
        for (size_t j = 0; j < p; ++j) {
            X[i * p + j] = power;
            power *= xi;
        }
    }
    std::vector<double> XtX(p * p), Xt(p * n);
    linalg::transpose(X.data(), Xt.data(), n, p);
    linalg::matmul(Xt.data(), X.data(), XtX.data(), p, n, p);
    std::vector<double> Xty(p);
    for (size_t i = 0; i < p; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < n; ++j) { sum += Xt[i * n + j] * y[j]; }
        Xty[i] = sum;
    }
    std::vector<double> coeffs(p);
    if (!linalg::solve_cholesky(XtX.data(), Xty.data(), coeffs.data(), p)) {
        double mean_y = 0.0;
        for (size_t i = 0; i < n; ++i) mean_y += y[i];
        mean_y /= static_cast<double>(n);
        for (size_t i = 0; i < n; ++i) { out_fitted[i] = mean_y; }
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        double val = 0.0;
        for (size_t j = 0; j < p; ++j) { val += X[i * p + j] * coeffs[j]; }
        out_fitted[i] = val;
    }
    if (out_coeffs) { std::copy(coeffs.begin(), coeffs.end(), out_coeffs); }
}

void weighted_polynomial_fit(
    const double* x,
    const double* y,
    const double* weights,
    size_t n,
    int degree,
    double* out_fitted,
    double* out_coeffs
) {
    if (n == 0 || degree < 0) return;
    const size_t p = static_cast<size_t>(degree) + 1;
    if (n < p) {
        for (size_t i = 0; i < n; ++i) { out_fitted[i] = y[i]; }
        return;
    }
    std::vector<double> X(n * p);
    for (size_t i = 0; i < n; ++i) {
        double xi = x[i], power = 1.0;
        for (size_t j = 0; j < p; ++j) { X[i * p + j] = power; power *= xi; }
    }
    std::vector<double> XtWX(p * p, 0.0);
    for (size_t i = 0; i < p; ++i) {
        for (size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) { sum += X[k * p + i] * weights[k] * X[k * p + j]; }
            XtWX[i * p + j] = sum;
        }
    }
    std::vector<double> XtWy(p, 0.0);
    for (size_t i = 0; i < p; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < n; ++j) { sum += X[j * p + i] * weights[j] * y[j]; }
        XtWy[i] = sum;
    }
    std::vector<double> coeffs(p);
    if (!linalg::solve_cholesky(XtWX.data(), XtWy.data(), coeffs.data(), p)) {
        double sum_wy = 0.0, sum_w = 0.0;
        for (size_t i = 0; i < n; ++i) { sum_wy += weights[i] * y[i]; sum_w += weights[i]; }
        double mean_y = (sum_w > 0.0) ? (sum_wy / sum_w) : 0.0;
        for (size_t i = 0; i < n; ++i) { out_fitted[i] = mean_y; }
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        double val = 0.0;
        for (size_t j = 0; j < p; ++j) { val += X[i * p + j] * coeffs[j]; }
        out_fitted[i] = val;
    }
    if (out_coeffs) { std::copy(coeffs.begin(), coeffs.end(), out_coeffs); }
}

// ================================================================
// LOESS Implementation
// ================================================================

static inline double tricube_weight(double u) {
    if (std::abs(u) >= 1.0) return 0.0;
    double t = 1.0 - std::abs(u * u * u);
    return t * t * t;
}

void loess_fit(
    const double* x,
    const double* y,
    size_t n,
    double span,
    int degree,
    double* out_fitted,
    int n_threads
) {
    if (n == 0) return;
    if (n_threads <= 0) n_threads = omp_get_max_threads();
    size_t window_size = static_cast<size_t>(std::ceil(span * static_cast<double>(n)));
    window_size = std::max(window_size, static_cast<size_t>(degree) + 1);
    window_size = std::min(window_size, n);
    #pragma omp parallel num_threads(n_threads)
    {
        std::vector<double> local_x(n), local_y(n), local_weights(n), local_fitted(n);
        #pragma omp for schedule(dynamic, 16)
        for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n); ++ii) {
            const size_t i = static_cast<size_t>(ii), xi = x[i];
            std::vector<std::pair<double, size_t>> distances;
            distances.reserve(n);
            for (size_t j = 0; j < n; ++j) { distances.push_back({std::abs(x[j] - xi), j}); }
            std::partial_sort(distances.begin(), distances.begin() + window_size, distances.end());
            double h = distances[window_size - 1].first;
            if (h == 0.0) h = 1e-10;
            size_t n_local = 0;
            for (size_t k = 0; k < window_size; ++k) {
                size_t j = distances[k].second;
                double u = distances[k].first / h, w = tricube_weight(u);
                if (w > 0.0) { local_x[n_local] = x[j]; local_y[n_local] = y[j]; local_weights[n_local] = w; ++n_local; }
            }
            if (n_local > 0) {
                weighted_polynomial_fit(local_x.data(), local_y.data(), local_weights.data(), n_local, degree, local_fitted.data(), nullptr);
                out_fitted[i] = local_fitted[0];
            } else { out_fitted[i] = y[i]; }
        }
    }
}

// ================================================================
// High-level Interfaces
// ================================================================

std::pair<std::vector<double>, std::vector<double>>
polynomial_regression(
    const std::vector<double>& x,
    const std::vector<double>& y,
    int degree
) {
    const size_t n = x.size(), p = static_cast<size_t>(degree) + 1;
    std::vector<double> fitted(n), coeffs(p);
    polynomial_fit(x.data(), y.data(), n, degree, fitted.data(), coeffs.data());
    return {fitted, coeffs};
}

std::vector<double> loess_regression(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double span,
    int degree,
    int n_threads
) {
    const size_t n = x.size();
    std::vector<double> fitted(n);
    loess_fit(x.data(), y.data(), n, span, degree, fitted.data(), n_threads);
    return fitted;
}

} // namespace hvg
} // namespace kernel
} // namespace perturblab
