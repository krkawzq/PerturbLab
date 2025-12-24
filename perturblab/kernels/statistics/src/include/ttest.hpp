// ttest.hpp - T-test implementations for differential expression
#pragma once

#include "sparse.hpp"
#include <cmath>
#include <vector>
#include <cstddef>

namespace hpdex {

// Result structure for t-tests
struct TTestResult {
    std::vector<double> t_statistic;  // t statistics
    std::vector<double> p_value;      // two-sided p-values
    std::vector<double> mean_diff;    // mean difference (target - reference)
    std::vector<double> log2_fc;      // log2 fold change
};

// Student's t-test (assumes equal variances)
// Fast but less robust
template<typename T>
TTestResult student_ttest(
    const view::CscView<T>& A,
    const int32_t* group_id,
    size_t n_targets,
    int threads = -1,
    size_t* progress_ptr = nullptr
);

// Welch's t-test (does not assume equal variances)
// More robust, recommended for single-cell data
template<typename T>
TTestResult welch_ttest(
    const view::CscView<T>& A,
    const int32_t* group_id,
    size_t n_targets,
    int threads = -1,
    size_t* progress_ptr = nullptr
);

// Log fold change computation
// Simple but effective effect size measure
template<typename T>
TTestResult log_fold_change(
    const view::CscView<T>& A,
    const int32_t* group_id,
    size_t n_targets,
    double pseudocount = 1e-9,
    int threads = -1
);

} // namespace hpdex

