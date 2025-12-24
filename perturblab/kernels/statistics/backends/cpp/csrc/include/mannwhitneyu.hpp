// mannwhitneyu.hpp - Mann-Whitney U test
#pragma once
#include "macro.hpp"
#include "sparse.hpp"
#include <cstddef>
#include <vector>
#include <variant>

namespace perturblab {
namespace kernel {

/**
 * @brief Result structure for Mann-Whitney U test.
 */
struct MWUResult {
    std::vector<double> U1;
    std::vector<double> U2;
    std::vector<double> P;
};

/**
 * @brief Configuration options for the Mann-Whitney U test.
 */
struct MannWhitneyuOption {
    bool ref_sorted;
    bool tar_sorted;

    bool tie_correction;
    bool use_continuity;

    bool fast_norm;

    /**
     * @brief Strategy for handling zero values in sparse matrices.
     */
    enum ZeroHandling { none = 0, min = 1, max = 2, mix = 3 } zero_handling;

    /**
     * @brief Direction of the alternative hypothesis.
     */
    enum Alternative { less = 0, greater = 1, two_sided = 2 } alternative;

    /**
     * @brief Method for calculating the p-value.
     */
    enum Method { exact = 1, asymptotic = 2 } method;
};

/**
 * @brief Perform Mann-Whitney U test on a CSC sparse matrix.
 * 
 * Algorithm:
 * 1. Calculate ranks for all elements (including zeros).
 * 2. Sum ranks for each group.
 * 3. U1 = R1 - n1*(n1+1)/2, U2 = n1*n2 - U1.
 * 4. P-value calculation using asymptotic normal approximation (if requested).
 * 
 * @tparam T Numeric type of the matrix.
 * @param A CSC view of the matrix.
 * @param group_id Array of group IDs for each observation.
 * @param n_targets Number of unique target groups.
 * @param option Test options.
 * @param threads Number of threads (optional).
 * @param progress_ptr Pointer to progress counter (optional).
 * @return MWUResult containing U statistics and p-values.
 */
template<class T>
MWUResult mannwhitneyu(
    const view::CscView<T>& A,
    const int32_t* group_id,
    const size_t& n_targets,
    const MannWhitneyuOption& option,
    const int threads = -1,
    size_t* progress_ptr = nullptr
);

/**
 * @brief Compute means for multiple groups in a CSC sparse matrix.
 */
template<class T>
std::vector<double> group_mean(
    const view::CscView<T>&,
    const int32_t*,
    const size_t&,
    bool,
    int,
    bool = false
);

} // namespace kernel
} // namespace perturblab
