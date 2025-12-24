// mannwhitneyu.cpp
// Mann-Whitney U test CPU implementation
//
// ⚠️  WARNING: MANUALLY OPTIMIZED CODE - DO NOT AUTO-REFACTOR ⚠️
//
// This file contains highly optimized C++ code with careful attention to:
// - Memory layout and cache efficiency
// - SIMD vectorization (Highway library)
// - Thread-local workspaces to avoid allocations
// - Branch prediction and loop unrolling
// - Numerical stability
//
// The code has been extensively benchmarked and profiled. Any modifications
// should be carefully tested for performance regression.
//
#include "mannwhitneyu.hpp"
#include "common.hpp"
#include "macro.hpp"
#include "simd.hpp"
#include "sparse.hpp"
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <string>
#include <tuple>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_max_threads() { return 1; }
static inline int omp_get_thread_num() { return 0; }
static inline int omp_get_num_threads() { return 1; }
#endif

#ifndef _MSC_VER
 #include <cstdlib>
#endif

namespace perturblab {
namespace kernel {

// -------- precise normal tail helpers (erfc-based) --------
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244008443621048490392848359376887
#endif

/**
 * @brief Precise normal survival function using std::erfc.
 * SF(z) = 0.5 * erfc(z / sqrt(2))
 */
force_inline_ double norm_sf_precise(double z) {
    return 0.5 * std::erfc(z * M_SQRT1_2);
}

/**
 * @brief Compute mean and standard deviation for Mann-Whitney U test with high precision.
 * 
 * Algorithm:
 * mu = (n1 * n2) / 2
 * sigma = sqrt((n1 * n2 / 12) * ((n1 + n2 + 1) - tie_sum / ((n1 + n2) * (n1 + n2 - 1))))
 */
force_inline_ void compute_mu_sigma_precise(
    double n1, double n2, double tie_sum, double& mu, double& sigma
) {
    const double N  = n1 + n2;
    const double nm = n1 * n2;
    mu = 0.5 * nm;

    double tie_term = 0.0;
    const double denom = N * (N - 1.0);
    if (denom > 0.0) tie_term = tie_sum / denom;

    const double var = nm * ((N + 1.0) - tie_term) / 12.0;
    sigma = (var > 0.0) ? std::sqrt(var) : 0.0;
}

static inline void array_p_asymptotic_greater_precise(
    const double* U1, const double* n1, const double* n2,
    const double* tie_sum, const double* cc,
    double* out, size_t len
){
    for (size_t i = 0; i < len; ++i) {
        double mu, sigma;
        compute_mu_sigma_precise(n1[i], n2[i], tie_sum[i], mu, sigma);
        double p = 1.0;
        if (sigma > 0.0) {
            const double z = (U1[i] - mu - cc[i]) / sigma;
            p = norm_sf_precise(z);
        }
        out[i] = (p < 0.0 ? 0.0 : (p > 1.0 ? 1.0 : p));
    }
}

static inline void array_p_asymptotic_less_precise(
    const double* U1, const double* n1, const double* n2,
    const double* tie_sum, const double* cc,
    double* out, size_t len
){
    for (size_t i = 0; i < len; ++i) {
        const double nm = n1[i] * n2[i];
        const double U2 = nm - U1[i];
        double mu, sigma;
        compute_mu_sigma_precise(n1[i], n2[i], tie_sum[i], mu, sigma);
        double p = 1.0;
        if (sigma > 0.0) {
            const double z = (U2 - mu - cc[i]) / sigma;
            p = norm_sf_precise(z);
        }
        out[i] = (p < 0.0 ? 0.0 : (p > 1.0 ? 1.0 : p));
    }
}

static inline void array_p_asymptotic_two_sided_precise(
    const double* U1, const double* n1, const double* n2,
    const double* tie_sum, const double* cc,
    double* out, size_t len
){
    for (size_t i = 0; i < len; ++i) {
        const double nm  = n1[i] * n2[i];
        const double U2  = nm - U1[i];
        const double U   = (U1[i] > U2 ? U1[i] : U2);
        double mu, sigma;
        compute_mu_sigma_precise(n1[i], n2[i], tie_sum[i], mu, sigma);

        double p = 1.0;
        if (sigma > 0.0) {
            const double z = (U - mu - cc[i]) / sigma;
            p = 2.0 * norm_sf_precise(z);
            if (p > 1.0) p = 1.0;
            if (p < 0.0) p = 0.0;
        }
        out[i] = p;
    }
}

/**
 * @brief Exact p-value calculation for Mann-Whitney U test using dynamic programming.
 * 
 * Algorithm:
 * Uses the recurrence relation for the distribution of the Mann-Whitney U statistic.
 * Time complexity: O(n1 * n2 * Umax), Space complexity: O(Umax).
 */
force_inline_ double p_exact(double U, size_t n1, size_t n2) {
    using U64 = unsigned long long;

    const size_t Umax = n1 * n2;
    if unlikely_(Umax == 0) return 1.0;

    const double U_clip = std::max(0.0, std::min(static_cast<double>(Umax), U));
    const size_t u_stat = static_cast<size_t>(std::floor(U_clip));

    const size_t SZ = Umax + 1;
    const size_t STACK_LIMIT = 1 << 12;

    U64* dp     = nullptr;
    U64* ndp    = nullptr;
    bool on_heap = false;

    alignas(64) U64 dp_stack[STACK_LIMIT];
    alignas(64) U64 ndp_stack[STACK_LIMIT];

    if (SZ <= STACK_LIMIT) {
        dp  = dp_stack;
        ndp = ndp_stack;
    } else {
        void* dp_raw  = nullptr;
        void* ndp_raw = nullptr;
        malloc_aligned_custom_(dp_raw,  SZ * sizeof(U64), 64);
        malloc_aligned_custom_(ndp_raw, SZ * sizeof(U64), 64);
        if unlikely_(!dp_raw || !ndp_raw) {
            if (dp_raw)  free_aligned_custom_(dp_raw);
            if (ndp_raw) free_aligned_custom_(ndp_raw);
            throw std::bad_alloc();
        }
        dp  = static_cast<U64*>(dp_raw);
        ndp = static_cast<U64*>(ndp_raw);
        on_heap = true;
    }

    // Initialization
    dp[0] = 1ULL;
    for (size_t i = 1; i < SZ; ++i) dp[i] = 0ULL;
    for (size_t i = 0; i < SZ; ++i) ndp[i] = 0ULL;

    U64* dp_cur = dp;
    U64* dp_nxt = ndp;

    for (size_t i = 1; i <= n1; ++i) {
        const size_t prev_up = (i - 1) * n2;
        const size_t up      = i * n2;

        U64 win = 0ULL;

        size_t u = 0;
        const size_t bound1 = prev_up < up ? prev_up : up;
        for (; u <= bound1; ++u) {
            win += dp_cur[u];
            if (u >= n2 + 1) win -= dp_cur[u - (n2 + 1)];
            dp_nxt[u] = win;
        }

        for (; u <= up; ++u) {
            if (u >= n2 + 1) win -= dp_cur[u - (n2 + 1)];
            dp_nxt[u] = win;
        }

        for (size_t j = 0; j <= up; ++j) dp_cur[j] = 0ULL;

        U64* tmp = dp_cur;
        dp_cur = dp_nxt;
        dp_nxt = tmp;
    }

    double total = 0.0;
    for (size_t u = 0; u <= Umax; ++u) {
        total += static_cast<double>(dp_cur[u]);
    }

    const size_t kc    = Umax - u_stat;
    const size_t small = (u_stat < kc ? u_stat : kc);

    double cdf_small = 0.0;
    for (size_t u = 0; u <= small; ++u) {
        cdf_small += static_cast<double>(dp_cur[u]);
    }
    const double pmf_small = static_cast<double>(dp_cur[small]);

    double sf_ge;
    if likely_(u_stat <= kc) {
        sf_ge = 1.0 - cdf_small / total + pmf_small / total;
    } else {
        sf_ge = cdf_small / total;
    }

    if (on_heap) {
        free_aligned_custom_(dp);
        free_aligned_custom_(ndp);
    }

    return sf_ge;
}

force_inline_ void p_asymptotic_parallel(
    const double* U1, const double* n1, const double* n2,
    const double* tie_sum, const double* cc,
    double* out, size_t N,
    MannWhitneyuOption::Alternative alt,
    bool fast_norm,
    int threads,
    size_t* progress_ptr
) {
    using AsympFn = void(*)(const double*, const double*, const double*, const double*, const double*, double*, size_t);

    AsympFn fn_fast = nullptr;
    switch (alt) {
    case MannWhitneyuOption::Alternative::two_sided: fn_fast = array_p_asymptotic_two_sided; break;
    case MannWhitneyuOption::Alternative::greater:   fn_fast = array_p_asymptotic_greater;   break;
    case MannWhitneyuOption::Alternative::less:      fn_fast = array_p_asymptotic_less;      break;
    }

    AsympFn fn_precise = nullptr;
    switch (alt) {
    case MannWhitneyuOption::Alternative::two_sided: fn_precise = array_p_asymptotic_two_sided_precise; break;
    case MannWhitneyuOption::Alternative::greater:   fn_precise = array_p_asymptotic_greater_precise;   break;
    case MannWhitneyuOption::Alternative::less:      fn_precise = array_p_asymptotic_less_precise;      break;
    }

    AsympFn fn = fast_norm ? fn_fast : fn_precise;

    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();
        constexpr size_t align = 8;  
        const size_t base = (N + nth - 1) / nth;  
        const size_t chunk = ((base + align - 1) / align) * align;
        const size_t begin = tid * chunk;
        const size_t end   = std::min(N, begin + chunk);
        
        if (begin < end) {
            fn(U1 + begin, n1 + begin, n2 + begin, tie_sum + begin, cc + begin,
               out + begin, end - begin);
        }
        if (progress_ptr) progress_ptr[tid]++;
    }
}

force_inline_ void p_exact_parallel(
    const double* U1,
    const double* n1,
    const double* n2,
    double*       out,
    size_t        N,
    MannWhitneyuOption::Alternative alt,
    int           threads,
    size_t*       progress_ptr
) {
    #pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)N; ++i) {
        const size_t a = static_cast<size_t>(n1[i]);
        const size_t b = static_cast<size_t>(n2[i]);
        const double U1d   = U1[i];
        const double Nprod = static_cast<double>(a) * static_cast<double>(b);

        double p = 1.0;
        switch (alt) {
            case MannWhitneyuOption::two_sided: {
                const double U2   = Nprod - U1d;
                const double Umax = (U1d > U2 ? U1d : U2);
                p = 2.0 * p_exact(Umax, a, b);
            } break;
            case MannWhitneyuOption::greater:
                p = p_exact(U1d, a, b);
                break;
            case MannWhitneyuOption::less: {
                const double U2 = Nprod - U1d;
                p = p_exact(U2, a, b);
            } break;
        }
        if (p < 0.0) p = 0.0;
        if (p > 1.0) p = 1.0;
        out[i] = p;
        if (progress_ptr) progress_ptr[omp_get_thread_num()]++;
    }
}

// ========================= Ranking Helpers =========================

static inline size_t merge_and_rank_segment(
    const double* a, size_t na,
    const double* b, size_t nb,
    size_t rank,
    double& R1, double& tie_sum, bool& has_tie
) {
    size_t i = 0, j = 0;
    while (i < na || j < nb) {
        double v; bool take_a = false, take_b = false;
        if (i < na && j < nb) {
            if (a[i] < b[j]) { v = a[i]; take_a = true; }
            else if (b[j] < a[i]) { v = b[j]; take_b = true; }
            else { v = a[i]; take_a = take_b = true; }
        } else if (i < na) {
            v = a[i]; take_a = true;
        } else {
            v = b[j]; take_b = true;
        }

        size_t eq_a = 0, eq_b = 0;
        if (take_a) { while (i + eq_a < na && !(a[i + eq_a] < v) && !(v < a[i + eq_a])) ++eq_a; }
        if (take_b) { while (j + eq_b < nb && !(b[j + eq_b] < v) && !(v < b[j + eq_b])) ++eq_b; }

        const size_t t = eq_a + eq_b;
        const double start = static_cast<double>(rank);
        const double end   = static_cast<double>(rank + t - 1);
        const double avg   = 0.5 * (start + end);

        R1 += static_cast<double>(eq_a) * avg;
        if (t > 1) {
            const double tt = static_cast<double>(t);
            tie_sum += (tt * tt * tt - tt);
            has_tie = true;
        }
        rank += t;
        i += eq_a; j += eq_b;
    }
    return rank;
}

static inline void merge_rank_sum_with_tie_include_zeros(
    const double* a, size_t n1_exp,
    const double* b, size_t n2_exp,
    size_t a_zero_implicit,
    size_t b_zero_implicit,
    double& R1, double& tie_sum, bool& has_tie
) {
    R1 = 0.0; tie_sum = 0.0; has_tie = false;
    size_t ai_neg_end = 0; while (ai_neg_end < n1_exp && a[ai_neg_end] < 0.0) ++ai_neg_end;
    size_t ai_zero_end = ai_neg_end;
    while (ai_zero_end < n1_exp && !(0.0 < a[ai_zero_end]) && !(a[ai_zero_end] < 0.0)) ++ai_zero_end;
    const size_t a_neg_n   = ai_neg_end;
    const size_t a_zero_exp= ai_zero_end - ai_neg_end;
    const size_t a_pos_n   = n1_exp - ai_zero_end;

    size_t bi_neg_end = 0; while (bi_neg_end < n2_exp && b[bi_neg_end] < 0.0) ++bi_neg_end;
    size_t bi_zero_end = bi_neg_end;
    while (bi_zero_end < n2_exp && !(0.0 < b[bi_zero_end]) && !(b[bi_zero_end] < 0.0)) ++bi_zero_end;
    const size_t b_neg_n   = bi_neg_end;
    const size_t b_zero_exp= bi_zero_end - bi_neg_end;
    const size_t b_pos_n   = n2_exp - bi_zero_end;

    const size_t A_zero_total = a_zero_exp + a_zero_implicit;
    const size_t B_zero_total = b_zero_exp + b_zero_implicit;
    size_t rank = 1;

    rank = merge_and_rank_segment(a, a_neg_n, b, b_neg_n, rank, R1, tie_sum, has_tie);
    const size_t t_zero = A_zero_total + B_zero_total;
    if (t_zero > 0) {
        const double start = static_cast<double>(rank);
        const double end   = static_cast<double>(rank + t_zero - 1);
        const double avg   = 0.5 * (start + end);
        R1 += static_cast<double>(A_zero_total) * avg;
        if (t_zero > 1) {
            const double tz = static_cast<double>(t_zero);
            tie_sum += (tz * tz * tz - tz);
            has_tie = true;
        }
        rank += t_zero;
    }
    rank = merge_and_rank_segment(a + ai_zero_end, a_pos_n, b + bi_zero_end, b_pos_n, rank, R1, tie_sum, has_tie);
}

static inline void merge_rank_sum_with_tie_zero_extreme(
    const double* a, size_t n1_exp,
    const double* b, size_t n2_exp,
    size_t a_zero_implicit,
    size_t b_zero_implicit,
    bool zero_at_head,
    double& R1, double& tie_sum, bool& has_tie
) {
    R1 = 0.0; tie_sum = 0.0; has_tie = false;
    size_t ai_neg_end = 0; while (ai_neg_end < n1_exp && a[ai_neg_end] < 0.0) ++ai_neg_end;
    size_t ai_zero_end = ai_neg_end;
    while (ai_zero_end < n1_exp && !(0.0 < a[ai_zero_end]) && !(a[ai_zero_end] < 0.0)) ++ai_zero_end;
    const size_t a_neg_n    = ai_neg_end;
    const size_t a_zero_exp = ai_zero_end - ai_neg_end;
    const size_t a_pos_n    = n1_exp - ai_zero_end;

    size_t bi_neg_end = 0; while (bi_neg_end < n2_exp && b[bi_neg_end] < 0.0) ++bi_neg_end;
    size_t bi_zero_end = bi_neg_end;
    while (bi_zero_end < n2_exp && !(0.0 < b[bi_zero_end]) && !(b[bi_zero_end] < 0.0)) ++bi_zero_end;
    const size_t b_neg_n    = bi_neg_end;
    const size_t b_zero_exp = bi_zero_end - bi_neg_end;
    const size_t b_pos_n    = n2_exp - bi_zero_end;

    const size_t A_zero_total = a_zero_exp + a_zero_implicit;
    const size_t B_zero_total = b_zero_exp + b_zero_implicit;
    const size_t t_zero = A_zero_total + B_zero_total;
    size_t rank = 1;

    auto emit_zero_block = [&](size_t& rk){
        if (t_zero > 0) {
            const double start = static_cast<double>(rk);
            const double end   = static_cast<double>(rk + t_zero - 1);
            const double avg   = 0.5 * (start + end);
            R1 += static_cast<double>(A_zero_total) * avg;
            if (t_zero > 1) {
                const double tz = static_cast<double>(t_zero);
                tie_sum += (tz * tz * tz - tz);
                has_tie = true;
            }
            rk += t_zero;
        }
    };

    if (zero_at_head) emit_zero_block(rank);
    rank = merge_and_rank_segment(a, a_neg_n, b, b_neg_n, rank, R1, tie_sum, has_tie);
    rank = merge_and_rank_segment(a + ai_zero_end, a_pos_n, b + bi_zero_end, b_pos_n, rank, R1, tie_sum, has_tie);
    if (!zero_at_head) emit_zero_block(rank);
}

static inline void merge_rank_sum_with_tie(
    const double* a, size_t n1,
    const double* b, size_t n2,
    double& R1, double& tie_sum, bool& has_tie
) {
    size_t i = 0, j = 0, rank = 1;
    R1 = 0.0; tie_sum = 0.0; has_tie = false;
    while (i < n1 || j < n2) {
        double v; bool take_a = false, take_b = false;
        if (i < n1 && j < n2) {
            if (a[i] < b[j]) { v = a[i]; take_a = true; }
            else if (b[j] < a[i]) { v = b[j]; take_b = true; }
            else { v = a[i]; take_a = take_b = true; }
        } else if (i < n1) { v = a[i]; take_a = true; }
        else { v = b[j]; take_b = true; }
        size_t eq_a = 0, eq_b = 0;
        if (take_a) { while (i + eq_a < n1 && !(a[i + eq_a] < v) && !(v < a[i + eq_a])) ++eq_a; }
        if (take_b) { while (j + eq_b < n2 && !(b[j + eq_b] < v) && !(v < b[j + eq_b])) ++eq_b; }
        const size_t t = eq_a + eq_b;
        const double start = static_cast<double>(rank);
        const double end   = static_cast<double>(rank + t - 1);
        const double avg   = 0.5 * (start + end);
        R1 += static_cast<double>(eq_a) * avg;
        if (t > 1) {
            const double tt = static_cast<double>(t);
            tie_sum += (tt * tt * tt - tt);
            has_tie = true;
        }
        rank += t;
        i += eq_a; j += eq_b;
    }
}

// =============== Main Kernel ===============

template<class T>
inline std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
mannWhitneyu_core(
    const T* data, const int64_t* indices, const int64_t* indptr,
    const size_t& R, const size_t& C, const size_t& /*nnz*/,
    const int32_t* group_id, size_t n_targets,
    const MannWhitneyuOption& opt, int threads,
    size_t* progress_ptr
){
    if (n_targets == 0) return std::make_tuple(std::vector<double>{}, std::vector<double>{}, std::vector<double>{});
    const size_t G = n_targets + 1;
    const size_t Npairs = C * n_targets;
    std::vector<double> U1_out(Npairs, 0.0);
    std::vector<double> P_out (Npairs, 1.0);
    std::vector<double> U2_out(Npairs, 0.0);
    std::vector<double> n1_arr(Npairs, 0.0);
    std::vector<double> n2_arr(Npairs, 0.0);
    std::vector<double> tie_arr(Npairs, 0.0);
    std::vector<double> cc_arr (Npairs, opt.use_continuity ? 0.5 : 0.0);

    std::vector<size_t> group_rows(G, 0);
    for (size_t r = 0; r < R; ++r) {
        int g = group_id[r];
        if (g >= 0 && (size_t)g < G) ++group_rows[(size_t)g];
    }
    std::vector<size_t> group_off(G + 1, 0);
    for (size_t g = 1; g <= G; ++g) group_off[g] = group_off[g - 1] + group_rows[g - 1];
    const size_t cap_total = group_off[G];

    std::atomic<bool> has_error{false};
    std::string error_message;

    using SortFn = void(*)(double*, size_t);
    auto sort_impl = +[](double* ptr, size_t n){
        if (n > 1) {
            hwy::HWY_NAMESPACE::VQSortStatic(ptr, n, hwy::SortAscending());
        }
    };
    auto sort_noop = +[](double*, size_t){};
    SortFn sort_ref = opt.ref_sorted ? sort_noop : sort_impl;
    SortFn sort_tar = opt.tar_sorted ? sort_noop : sort_impl;
    const double tie_mask = opt.tie_correction ? 1.0 : 0.0;
    const bool zero_at_head = (opt.zero_handling == MannWhitneyuOption::min);

    #pragma omp parallel num_threads(threads)
    {
        std::vector<double> buf(cap_total);
        std::vector<double*> base(G);
        for (size_t g = 0; g < G; ++g) base[g] = buf.data() + group_off[g];
        std::vector<size_t> cnt(G, 0);

        #pragma omp for schedule(static)
        for (std::ptrdiff_t cc = 0; cc < (std::ptrdiff_t)C; ++cc) {
            if (has_error.load()) {
                if (progress_ptr) progress_ptr[omp_get_thread_num()]++;
                continue;
            }
            std::fill(cnt.begin(), cnt.end(), 0);
            const int64_t p0 = indptr[cc], p1 = indptr[cc + 1];
            for (int64_t p = p0; p < p1; ++p) {
                const int64_t r = indices[p];
                if (r < 0 || (size_t)r >= R) continue;
                const int g = group_id[(size_t)r];
                if (g < 0 || (size_t)g >= G) continue;
                double v = (double)data[p];
                if (!is_valid_value(v)) continue;
                double* dst = base[(size_t)g] + cnt[(size_t)g];
                *dst = v;
                ++cnt[(size_t)g];
            }
            if (cnt[0] > 1) sort_ref(base[0], cnt[0]);
            for (size_t g = 1; g < G; ++g) { if (cnt[g] > 1) sort_tar(base[g], cnt[g]); }

            switch (opt.zero_handling) {
            case MannWhitneyuOption::none: {
                const size_t n1_total = cnt[0];
                if (n1_total < 2) {
                    if (!has_error.exchange(true)) error_message = "Sample too small for reference at column " + std::to_string(cc);
                    break;
                }
                const double n1d = (double)n1_total;
                for (size_t g = 1; g < G; ++g) {
                    const size_t n2_total = cnt[g];
                    if (n2_total < 2) {
                        if (!has_error.exchange(true)) error_message = "Sample too small for group " + std::to_string(g) + " at column " + std::to_string(cc);
                        continue;
                    }
                    double R1 = 0.0, tie_sum = 0.0; bool has_tie_local = false;
                    merge_rank_sum_with_tie(base[0], cnt[0], base[g], cnt[g], R1, tie_sum, has_tie_local);
                    const size_t idx = (size_t)cc * n_targets + (g - 1);
                    U1_out[idx]  = R1 - n1d * (n1d + 1.0) * 0.5;
                    n1_arr[idx]  = n1d;
                    n2_arr[idx]  = (double)n2_total;
                    tie_arr[idx] = tie_sum * tie_mask;
                }
            } break;
            case MannWhitneyuOption::min:
            case MannWhitneyuOption::max: {
                const size_t n1_total = group_rows[0];
                if (n1_total < 2) {
                    if (!has_error.exchange(true)) error_message = "Sample too small for reference at column " + std::to_string(cc);
                    break;
                }
                const size_t a_zero_imp = group_rows[0] - cnt[0];
                for (size_t g = 1; g < G; ++g) {
                    const size_t n2_total = group_rows[g];
                    if (n2_total < 2) {
                        if (!has_error.exchange(true)) error_message = "Sample too small for group " + std::to_string(g) + " at column " + std::to_string(cc);
                        continue;
                    }
                    const size_t b_zero_imp = group_rows[g] - cnt[g];
                    double R1 = 0.0, tie_sum = 0.0; bool has_tie_local = false;
                    merge_rank_sum_with_tie_zero_extreme(base[0], cnt[0], base[g], cnt[g], a_zero_imp, b_zero_imp, zero_at_head, R1, tie_sum, has_tie_local);
                    const double n1d = (double)n1_total;
                    const size_t idx = (size_t)cc * n_targets + (g - 1);
                    U1_out[idx]  = R1 - n1d * (n1d + 1.0) * 0.5;
                    n1_arr[idx]  = n1d;
                    n2_arr[idx]  = (double)n2_total;
                    tie_arr[idx] = tie_sum * tie_mask;
                }
            } break;
            case MannWhitneyuOption::mix: {
                const size_t n1_total = group_rows[0];
                if (n1_total < 2) {
                    if (!has_error.exchange(true)) error_message = "Sample too small for reference at column " + std::to_string(cc);
                    break;
                }
                const size_t a_zero_imp = group_rows[0] - cnt[0];
                for (size_t g = 1; g < G; ++g) {
                    const size_t n2_total = group_rows[g];
                    if (n2_total < 2) {
                        if (!has_error.exchange(true)) error_message = "Sample too small for group " + std::to_string(g) + " at column " + std::to_string(cc);
                        continue;
                    }
                    const size_t b_zero_imp = group_rows[g] - cnt[g];
                    double R1 = 0.0, tie_sum = 0.0; bool has_tie_local = false;
                    merge_rank_sum_with_tie_include_zeros(base[0], cnt[0], base[g], cnt[g], a_zero_imp, b_zero_imp, R1, tie_sum, has_tie_local);
                    const double n1d = (double)n1_total;
                    const size_t idx = (size_t)cc * n_targets + (g - 1);
                    U1_out[idx]  = R1 - n1d * (n1d + 1.0) * 0.5;
                    n1_arr[idx]  = n1d;
                    n2_arr[idx]  = (double)n2_total;
                    tie_arr[idx] = tie_sum * tie_mask;
                }
            } break;
            }
            if (progress_ptr) progress_ptr[omp_get_thread_num()]++;
        }
    }

    if (has_error.load()) throw std::runtime_error(error_message);
    if (opt.method == MannWhitneyuOption::asymptotic) {
        p_asymptotic_parallel(U1_out.data(), n1_arr.data(), n2_arr.data(), tie_arr.data(), cc_arr.data(), P_out.data(), Npairs, opt.alternative, opt.fast_norm, threads, progress_ptr);
    } else {
        p_exact_parallel(U1_out.data(), n1_arr.data(), n2_arr.data(), P_out.data(), Npairs, opt.alternative, threads, progress_ptr);
    }
    for (size_t i = 0; i < Npairs; ++i) { U2_out[i] = n1_arr[i] * n2_arr[i] - U1_out[i]; }
    return std::make_tuple(std::move(U1_out), std::move(U2_out), std::move(P_out));
}

template<class T>
MWUResult mannwhitneyu(
    const view::CscView<T>& A,
    const int32_t* group_id,
    const size_t&  n_targets,
    const MannWhitneyuOption& option,
    int            threads,
    size_t*        progress_ptr
) {
    threads = threads < 0 ? omp_get_max_threads(): threads > omp_get_max_threads() ? omp_get_max_threads() : threads;
    bool has_progress_ptr = progress_ptr != nullptr;
    if (!has_progress_ptr) progress_ptr = new size_t[threads];

    const T* data = A.data();
    const int64_t* indices = A.indices();
    const int64_t* indptr  = A.indptr();
    size_t C = A.cols();
    size_t R = A.rows();
    size_t nnz = A.nnz();

    if (n_targets < 1) { throw std::runtime_error("[mannwhitney] n_targets must be >= 1"); }

    auto ret = mannWhitneyu_core<T>(data, indices, indptr, R, C, nnz, group_id, n_targets, option, threads, progress_ptr);
    if (!has_progress_ptr) delete[] progress_ptr;
    
    MWUResult result;
    result.U1 = std::move(std::get<0>(ret));
    result.U2 = std::move(std::get<1>(ret));
    result.P  = std::move(std::get<2>(ret));
    return result;
}

// ========================= Group Mean =========================

template<class T>
static force_inline_ std::vector<double>
group_mean_core(
    const T*           data,
    const int64_t*     indices,
    const int64_t*     indptr,
    const size_t&      R,
    const size_t&      C,
    const size_t&      /*nnz*/,
    const int32_t*     group_id,
    const size_t&      n_groups,
    bool               include_zeros,
    int                threads
){
    const size_t G = n_groups;
    if (G == 0 || C == 0) return {};
    std::vector<size_t> group_rows(G, 0);
    for (size_t r = 0; r < R; ++r) {
        int g = group_id[r];
        if (g >= 0 && size_t(g) < G) ++group_rows[size_t(g)];
    }
    if (threads < 0) threads = omp_get_max_threads();
    std::vector<double> mean_out(C * G, 0.0);

    #pragma omp parallel num_threads(threads)
    {
        std::vector<double> sum(G);
        std::vector<size_t> valid_cnt(G);
        std::vector<size_t> invalid_cnt(G);
        #pragma omp for schedule(dynamic)
        for (std::ptrdiff_t cc = 0; cc < (std::ptrdiff_t)C; ++cc) {
            std::memset(sum.data(),        0, G * sizeof(double));
            std::memset(valid_cnt.data(),  0, G * sizeof(size_t));
            std::memset(invalid_cnt.data(),0, G * sizeof(size_t));
            const size_t c   = static_cast<size_t>(cc);
            const int64_t p0 = indptr[c];
            const int64_t p1 = indptr[c + 1];
            for (int64_t p = p0; p < p1; ++p) {
                const int64_t r = indices[p];
                if (r < 0 || size_t(r) >= R) continue;
                const int gi = group_id[size_t(r)];
                if (gi < 0 || size_t(gi) >= G) continue;
                const double v = static_cast<double>(data[p]);
                if (is_valid_value(v)) { sum[size_t(gi)] += v; ++valid_cnt[size_t(gi)]; }
                else { ++invalid_cnt[size_t(gi)]; }
            }
            double* out_col = mean_out.data() + c * G;
            for (size_t g = 0; g < G; ++g) {
                size_t denom = include_zeros ? (group_rows[g] - invalid_cnt[g]) : valid_cnt[g];
                out_col[g] = (denom > 0) ? (sum[g] / static_cast<double>(denom)) : 0.0;
            }
        }
    }
    return mean_out;
}

template<class T>
static force_inline_ std::vector<double>
group_mean_core_kahan(
    const T*           data,
    const int64_t*     indices,
    const int64_t*     indptr,
    const size_t&      R,
    const size_t&      C,
    const size_t&      /*nnz*/,
    const int32_t*     group_id,
    const size_t&      n_groups,
    bool               include_zeros,
    int                threads
){
    const size_t G = n_groups;
    if (G == 0 || C == 0) return {};
    std::vector<size_t> group_rows(G, 0);
    for (size_t r = 0; r < R; ++r) {
        int g = group_id[r];
        if (g >= 0 && size_t(g) < G) ++group_rows[size_t(g)];
    }
    if (threads < 0) threads = omp_get_max_threads();
    std::vector<double> mean_out(C * G, 0.0);

    #pragma omp parallel num_threads(threads)
    {
        std::vector<double> sum(G, 0.0);
        std::vector<double> c(G, 0.0);
        std::vector<size_t> valid_cnt(G);
        std::vector<size_t> invalid_cnt(G);
        #pragma omp for schedule(dynamic)
        for (std::ptrdiff_t cc = 0; cc < (std::ptrdiff_t)C; ++cc) {
            std::fill(sum.begin(), sum.end(), 0.0);
            std::fill(c.begin(),   c.end(),   0.0);
            std::fill(valid_cnt.begin(), valid_cnt.end(), 0);
            std::fill(invalid_cnt.begin(), invalid_cnt.end(), 0);
            const size_t cidx   = static_cast<size_t>(cc);
            const int64_t p0 = indptr[cidx];
            const int64_t p1 = indptr[cidx + 1];
            for (int64_t p = p0; p < p1; ++p) {
                const int64_t r = indices[p];
                if (r < 0 || size_t(r) >= R) continue;
                const int gi = group_id[size_t(r)];
                if (gi < 0 || size_t(gi) >= G) continue;
                const double v = static_cast<double>(data[p]);
                if (is_valid_value(v)) {
                    double y = v - c[gi];
                    double t = sum[gi] + y;
                    c[gi] = (t - sum[gi]) - y;
                    sum[gi] = t;
                    ++valid_cnt[gi];
                } else { ++invalid_cnt[gi]; } 
            }
            double* out_col = mean_out.data() + cidx * G;
            for (size_t g = 0; g < G; ++g) {
                size_t denom = include_zeros ? (group_rows[g] - invalid_cnt[g]) : valid_cnt[g];
                out_col[g] = (denom > 0) ? (sum[g] / static_cast<double>(denom)) : 0.0;
            }
        }
    }
    return mean_out;
}

template<class T>
std::vector<double> group_mean(
    const view::CscView<T>& A,
    const int32_t* group_id,
    const size_t&  n_groups,
    bool           include_zeros,
    int            threads,
    bool           use_kahan
){
    const T* data = A.data();
    const int64_t* indices = A.indices();
    const int64_t* indptr  = A.indptr();
    size_t C = A.cols();
    size_t R = A.rows();
    size_t nnz = A.nnz();
    if (use_kahan) {
        return group_mean_core_kahan<T>(data, indices, indptr, R, C, nnz, group_id, n_groups, include_zeros, threads);
    } else {
        return group_mean_core<T>(data, indices, indptr, R, C, nnz, group_id, n_groups, include_zeros, threads);
    }
}

#define MWU_INSTANTIATE(T) \
    template MWUResult mannwhitneyu<T>( \
        const view::CscView<T>&, \
        const int32_t*, \
        const size_t&, \
        const MannWhitneyuOption&, \
        const int, \
        size_t* \
    );
TYPE_DISPATCH(MWU_INSTANTIATE);
#undef MWU_INSTANTIATE

#define GROUP_MEAN_INSTANTIATE(T) \
    template std::vector<double> group_mean<T>( \
        const view::CscView<T>&, \
        const int32_t*, \
        const size_t&, \
        bool, \
        int, \
        bool \
    );
TYPE_DISPATCH(GROUP_MEAN_INSTANTIATE);
#undef GROUP_MEAN_INSTANTIATE

} // namespace kernel
} // namespace perturblab
