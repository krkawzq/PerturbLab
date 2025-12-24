// group_ops.hpp - Group-wise operations on sparse matrices
// Provides efficient group aggregation functions
#pragma once

#include "config.hpp"
#include "macro.hpp"
#include "sparse.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace perturblab {
namespace kernel {

// ================================================================
// Group Mean
// ================================================================

/**
 * @brief Calculate group-wise means for each column of a sparse CSC matrix.
 * 
 * Given CSC matrix X (R x C) and row-wise group labels group_id,
 * calculates the mean of each column within each group.
 * 
 * Output layout: [col * n_groups + group]
 * result[c * n_groups + g] = mean of column c in group g
 * 
 * Algorithm:
 * mean = sum(values_in_group) / count_in_group
 * 
 * @tparam T Numeric type.
 * @param A CSC matrix view.
 * @param group_id Row grouping labels (R,) in range [0, n_groups).
 * @param n_groups Number of groups.
 * @param include_zeros Whether to include implicit zeros in the denominator.
 * @param threads Number of threads (0 = auto).
 * @param use_kahan Whether to use Kahan summation for higher precision.
 * @return Vector of size C * n_groups containing group-wise means.
 */
template<class T>
std::vector<double> group_mean(
    const view::CscView<T>& A,
    const int32_t* group_id,
    const size_t& n_groups,
    bool include_zeros,
    int threads,
    bool use_kahan
);

// ================================================================
// Group Variance
// ================================================================

/**
 * @brief Calculate group-wise variances for each column of a sparse CSC matrix.
 * 
 * Uses Welford's online algorithm for single-pass, numerically stable computation.
 * 
 * Output layout: [col * n_groups + group]
 * 
 * @tparam T Numeric type.
 * @param A CSC matrix view.
 * @param group_id Row grouping labels (R,).
 * @param n_groups Number of groups.
 * @param include_zeros Whether to include implicit zeros in the variance.
 * @param threads Number of threads.
 * @param ddof Delta Degrees of Freedom (0 for population, 1 for sample).
 * @return Vector of size C * n_groups containing group-wise variances.
 */
template<class T>
std::vector<double> group_var(
    const view::CscView<T>& A,
    const int32_t* group_id,
    const size_t& n_groups,
    bool include_zeros,
    int threads,
    int ddof
);

// ================================================================
// Group Mean and Variance
// ================================================================

/**
 * @brief Calculate both mean and variance in a single pass.
 * 
 * @tparam T Numeric type.
 * @param A CSC matrix view.
 * @param group_id Row grouping labels.
 * @param n_groups Number of groups.
 * @param include_zeros Whether to include implicit zeros.
 * @param threads Number of threads.
 * @param ddof Delta Degrees of Freedom.
 * @return Pair of <means, vars> vectors, each of size C * n_groups.
 */
template<class T>
std::pair<std::vector<double>, std::vector<double>> group_mean_var(
    const view::CscView<T>& A,
    const int32_t* group_id,
    const size_t& n_groups,
    bool include_zeros,
    int threads,
    int ddof
);

// ================================================================
// Group Count
// ================================================================

/**
 * @brief Calculate non-zero counts for each column and group.
 * 
 * @tparam T Numeric type.
 * @param A CSC matrix view.
 * @param group_id Row grouping labels.
 * @param n_groups Number of groups.
 * @param threads Number of threads.
 * @return Vector of size C * n_groups containing non-zero counts.
 */
template<class T>
std::vector<size_t> group_count_nonzero(
    const view::CscView<T>& A,
    const int32_t* group_id,
    const size_t& n_groups,
    int threads
);

} // namespace kernel
} // namespace perturblab
