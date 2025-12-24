#pragma once
#include "config.hpp"

/**
 * @brief Macro for type dispatching across common numeric types.
 */
#define TYPE_DISPATCH(DO) \
  DO(int8_t) \
  DO(uint8_t) \
  DO(int16_t) \
  DO(int32_t) \
  DO(int64_t) \
  DO(uint16_t) \
  DO(uint32_t) \
  DO(uint64_t) \
  DO(float) \
  DO(double)

// ============================================
// Optimization and Platform Tools
// ============================================

#if defined(_MSC_VER)
  #define force_inline_ __forceinline
  #define restrict_     __restrict
#else
  #define force_inline_ inline __attribute__((always_inline))
  #define restrict_     __restrict__
#endif

#if defined(_MSC_VER)
  #define malloc_aligned_(ptr, size) \
    do { ptr = static_cast<void*>(_aligned_malloc((size), ALIGN_SIZE)); } while(0)
  #define free_aligned_(ptr) \
    do { _aligned_free((ptr)); } while(0)
#else
  #define malloc_aligned_(ptr, size) \
    do { if (posix_memalign(&(ptr), ALIGN_SIZE, (size)) != 0) ptr = nullptr; } while(0)
  #define free_aligned_(ptr) \
    do { free((ptr)); } while(0)
#endif

#if defined(_MSC_VER)
  #define malloc_aligned_custom_(ptr, size, align) \
    do { ptr = static_cast<void*>(_aligned_malloc((size), (align))); } while(0)
  #define free_aligned_custom_(ptr) \
    do { _aligned_free((ptr)); } while(0)
#else
  #define malloc_aligned_custom_(ptr, size, align) \
    do { if (posix_memalign(&(ptr), (align), (size)) != 0) ptr = nullptr; } while(0)
  #define free_aligned_custom_(ptr) \
    do { free((ptr)); } while(0)
#endif

// assume_aligned hint
#if defined(__clang__) || defined(__GNUC__)
  #define assume_aligned_(p, N) reinterpret_cast<decltype(p)>(__builtin_assume_aligned((p), (N)))
#else
  #define assume_aligned_(p, N) (p)
#endif

// Branch prediction hints
#if defined(__clang__) || defined(__GNUC__)
  #define likely_(x)   (__builtin_expect(!!(x), 1))
  #define unlikely_(x) (__builtin_expect(!!(x), 0))
#else
  #define likely_(x)   (x)
  #define unlikely_(x) (x)
#endif

// Prefetch hints
#if defined(__clang__) || defined(__GNUC__)
  #define prefetch_r_(p,loc) __builtin_prefetch((p), 0, (loc))
  #define prefetch_w_(p,loc) __builtin_prefetch((p), 1, (loc))
#else
  #define prefetch_r_(p,loc) ((void)0)
  #define prefetch_w_(p,loc) ((void)0)
#endif

// Symbol export
#if defined(_MSC_VER)
  #define export_ __declspec(dllexport)
#else
  #define export_ __attribute__((visibility("default")))
#endif

// Alignment specification
#if defined(_MSC_VER)
  #define alignas_(N) __declspec(align((N)))
#else
  #define alignas_(N) alignas((N))
#endif