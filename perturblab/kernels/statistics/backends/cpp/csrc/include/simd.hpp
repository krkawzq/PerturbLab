// simd.hpp - Comprehensive SIMD Wrapper (Highway)
// Dependencies: config.hpp, macro.hpp
#pragma once

#include <cstddef>
#include <utility>
#include <limits>
#include <cstring>

#include "config.hpp"
#include "dependencies.hpp"
#include "macro.hpp"

// ================================================================
//      Conditional include Highway
// ================================================================

namespace hn = hwy::HWY_NAMESPACE;

namespace perturblab {
namespace kernel {
namespace simd {
using namespace hn;

/**
 * @brief Reduces an array to its maximum value using SIMD.
 * 
 * Algorithm: Parallel reduction using Highway's ReduceMax.
 * 
 * @tparam T Numeric type.
 * @param x Pointer to the array.
 * @param n Number of elements.
 * @return Maximum value in the array.
 */
template<class T>
force_inline_ T array_reduce_max(const T* x, size_t n) {
    using D = HWY_FULL(T);
    D d;

    const size_t step = Lanes(d);
    size_t i = 0;

    T max_val = std::numeric_limits<T>::lowest();

    // SIMD reduction
    if (n >= step) {
        auto vmax = Load(d, x);
        max_val = ReduceMax(d, vmax);
        i += step;

        for (; i + step <= n; i += step) {
            auto v = Load(d, x + i);
            auto local_max = ReduceMax(d, v);
            if (local_max > max_val) max_val = local_max;
        }
    }

    // Scalar tail processing
    for (; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }

    return max_val;
}

/**
 * @brief Reduces an array to its minimum value using SIMD.
 * 
 * Algorithm: Parallel reduction using Highway's ReduceMin.
 * 
 * @tparam T Numeric type.
 * @param x Pointer to the array.
 * @param n Number of elements.
 * @return Minimum value in the array.
 */
template<class T>
force_inline_ T array_reduce_min(const T* x, size_t n) {
    using D = HWY_FULL(T);
    D d;

    const size_t step = Lanes(d);
    size_t i = 0;

    T min_val = std::numeric_limits<T>::max();

    // SIMD reduction
    if (n >= step) {
        auto vmin = Load(d, x);
        min_val = ReduceMin(d, vmin);
        i += step;

        for (; i + step <= n; i += step) {
            auto v = Load(d, x + i);
            auto local_min = ReduceMin(d, v);
            if (local_min < min_val) min_val = local_min;
        }
    }

    // Scalar tail processing
    for (; i < n; ++i) {
        if (x[i] < min_val) min_val = x[i];
    }

    return min_val;
}

/**
 * @brief Reduces an array to its minimum and maximum values using SIMD.
 * 
 * @tparam T Numeric type.
 * @param x Pointer to the array.
 * @param n Number of elements.
 * @return Pair of {min, max} values.
 */
template<class T>
force_inline_ std::pair<T, T> array_reduce_minmax(const T* x, size_t n) {
    using D = HWY_FULL(T);
    D d;

    const size_t step = Lanes(d);
    size_t i = 0;

    T min_val = std::numeric_limits<T>::max();
    T max_val = std::numeric_limits<T>::lowest();

    if (n >= step) {
        auto v = Load(d, x);
        min_val = ReduceMin(d, v);
        max_val = ReduceMax(d, v);
        i += step;

        for (; i + step <= n; i += step) {
            v = Load(d, x + i);
            T local_min = ReduceMin(d, v);
            T local_max = ReduceMax(d, v);
            if (local_min < min_val) min_val = local_min;
            if (local_max > max_val) max_val = local_max;
        }
    }

    // Scalar tail processing
    for (; i < n; ++i) {
        if (x[i] < min_val) min_val = x[i];
        if (x[i] > max_val) max_val = x[i];
    }

    return {min_val, max_val};
}

} // namespace simd
} // namespace kernel
} // namespace perturblab