// ttest.hpp - T-test implementations for differential expression
#pragma once

#include "sparse.hpp"
#include <cmath>
#include <vector>
#include <cstddef>

namespace perturblab {
namespace kernel {

/**
 * @brief Result structure for t-test computations.
 */
struct TTestResult {
    std::vector<double> t_statistic;  /**< Calculated t-statistics */
    std::vector<double> p_value;      /**< Two-sided p-values */
    std::vector<double> mean_diff;    /**< Mean difference (target - reference) */
    std::vector<double> log2_fc;      /**< Log2 fold change */
};

/**
 * @brief Student's t-test (assumes equal variances).
 * 
 * Algorithm:
 * t = (x1 - x2) / (sp * sqrt(1/n1 + 1/n2))
 * where sp is the pooled standard deviation.
 * 
 * @tparam T Numeric type of the matrix.
 * @param A CSC view of the matrix.
 * @param group_id Array of group IDs.
 * @param n_targets Number of target groups.
 * @param threads Number of threads (optional).
 * @param progress_ptr Pointer to progress counter (optional).
 * @return TTestResult.
 */
template<typename T>
TTestResult student_ttest(
    const view::CscView<T>& A,
    const int32_t* group_id,
    size_t n_targets,
    int threads = -1,
    size_t* progress_ptr = nullptr
);

/**
 * @brief Welch's t-test (does not assume equal variances).
 * 
 * Algorithm:
 * t = (x1 - x2) / sqrt(s1^2/n1 + s2^2/n2)
 * Degrees of freedom are calculated using the Welch-Satterthwaite equation.
 * 
 * @tparam T Numeric type of the matrix.
 * @param A CSC view of the matrix.
 * @param group_id Array of group IDs.
 * @param n_targets Number of target groups.
 * @param threads Number of threads (optional).
 * @param progress_ptr Pointer to progress counter (optional).
 * @return TTestResult.
 */
template<typename T>
TTestResult welch_ttest(
    const view::CscView<T>& A,
    const int32_t* group_id,
    size_t n_targets,
    int threads = -1,
    size_t* progress_ptr = nullptr
);

/**
 * @brief Log2 fold change computation.
 * 
 * Formula: log2((mean_target + epsilon) / (mean_ref + epsilon))
 * 
 * @tparam T Numeric type of the matrix.
 * @param A CSC view of the matrix.
 * @param group_id Array of group IDs.
 * @param n_targets Number of target groups.
 * @param pseudocount Epsilon to avoid division by zero.
 * @param threads Number of threads (optional).
 * @return TTestResult (with t_statistic and p_value being empty/NaN).
 */
template<typename T>
TTestResult log_fold_change(
    const view::CscView<T>& A,
    const int32_t* group_id,
    size_t n_targets,
    double pseudocount = 1e-9,
    int threads = -1
);

} // namespace kernel
} // namespace perturblab
