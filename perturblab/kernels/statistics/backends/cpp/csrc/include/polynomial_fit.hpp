// polynomial_fit.hpp - Polynomial regression for HVG detection
// Fast alternative to LOESS regression
#pragma once

#include "config.hpp"
#include "macro.hpp"
#include <cstddef>
#include <vector>

namespace perturblab {
namespace kernel {
namespace hvg {

/**
 * @brief Polynomial regression fit using Ordinary Least Squares (OLS).
 * 
 * Fits y = a0 + a1*x + a2*x^2 + ... + an*x^n.
 * Fast alternative to LOESS, typically degree 2 is sufficient for single-cell data.
 * 
 * Algorithm: Solves the normal equation (X^T X)^{-1} X^T y.
 * 
 * @param x Input x coordinates [n].
 * @param y Input y coordinates [n].
 * @param n Number of data points.
 * @param degree Polynomial degree.
 * @param out_fitted Output: Fitted values [n].
 * @param out_coeffs Output: Coefficients [degree+1] (optional).
 */
void polynomial_fit(
    const double* x,
    const double* y,
    size_t n,
    int degree,
    double* out_fitted,
    double* out_coeffs = nullptr
);

/**
 * @brief Weighted polynomial regression fit.
 * 
 * Minimizes Σ w_i (y_i - ŷ_i)^2.
 * 
 * @param x Input x coordinates.
 * @param y Input y coordinates.
 * @param weights weights [n].
 * @param n Number of data points.
 * @param degree Polynomial degree.
 * @param out_fitted Output: Fitted values.
 * @param out_coeffs Output: Coefficients.
 */
void weighted_polynomial_fit(
    const double* x,
    const double* y,
    const double* weights,
    size_t n,
    int degree,
    double* out_fitted,
    double* out_coeffs = nullptr
);

/**
 * @brief Locally Weighted Polynomial Regression (Simplified LOESS).
 * 
 * For each point x_i, performs a local polynomial fit using tricube weights.
 * 
 * Algorithm:
 * - Tricube weight: w(d) = (1 - |d|^3)^3 for |d| < 1.
 * - Local window size = span * n.
 * - Fits each point independently (parallelized).
 * 
 * @param x Input x coordinates.
 * @param y Input y coordinates.
 * @param n Number of data points.
 * @param span Smoothing parameter (0-1), controlling local window size.
 * @param degree Polynomial degree.
 * @param out_fitted Output: Fitted values.
 * @param n_threads Number of threads.
 */
void loess_fit(
    const double* x,
    const double* y,
    size_t n,
    double span,
    int degree,
    double* out_fitted,
    int n_threads = 0
);

/**
 * @brief High-level C++ interface for polynomial regression.
 * 
 * @return pair<fitted_values, coefficients>
 */
std::pair<std::vector<double>, std::vector<double>>
polynomial_regression(
    const std::vector<double>& x,
    const std::vector<double>& y,
    int degree = 2
);

/**
 * @brief High-level C++ interface for LOESS regression.
 * 
 * @return fitted_values
 */
std::vector<double> loess_regression(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double span = 0.3,
    int degree = 2,
    int n_threads = 0
);

} // namespace hvg
} // namespace kernel
} // namespace perturblab
