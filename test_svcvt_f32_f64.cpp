/**
 * test_svcvt_f32_f64.cpp
 *
 * Tests correctness of:
 *   svcvt_f64_f32_x  -- convert even-indexed f32 elements → f64
 *   svcvtlt_f64_f32_x -- convert odd-indexed f32 elements → f64  (SVE2 "long top")
 *
 * Compile (AArch64 with SVE2):
 *   g++ -O2 -std=c++17 -march=armv9-a+sve2 -o test_svcvt test_svcvt_f32_f64.cpp
 *   or with clang:
 *   clang++ -O2 -std=c++17 -march=armv9-a+sve2 -o test_svcvt test_svcvt_f32_f64.cpp
 *
 * Run:
 *   ./test_svcvt
 */

#include <arm_sve.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>

// ---- helper: print pass/fail ------------------------------------------------
static int g_tests = 0, g_fails = 0;

static void check(bool ok, const char* msg) {
    ++g_tests;
    if (!ok) {
        ++g_fails;
        printf("  FAIL: %s\n", msg);
    }
}

// ---- core test --------------------------------------------------------------
/**
 * Given an f32 vector  [a0, a1, a2, a3, a4, a5, a6, a7, ...]
 *
 *  svcvt_f64_f32_x(pg64, vec):
 *      reads the BOTTOM half of each 128-bit segment (SVE "bottom" = even indices
 *      within the f64 vector width), i.e. elements at positions 0, 2, 4, ... of
 *      the f32 vector and produces a f64 vector of the same byte-width.
 *      Result f64: [(double)a0, (double)a2, (double)a4, ...]
 *
 *  svcvtlt_f64_f32_x(pg64, vec):
 *      reads the TOP half, i.e. the odd positions: a1, a3, a5, ...
 *      Result f64: [(double)a1, (double)a3, (double)a5, ...]
 *
 * This test verifies both by comparing against scalar reference values.
 */
static void test_even_odd_conversion() {
    printf("=== test_even_odd_conversion ===\n");

    // Build a reference array large enough for one full SVE vector of f32.
    // svcntw() = number of f32 lanes (e.g. 4 for 128-bit SVE, 8 for 256-bit, ...).
    const uint64_t vl_f32 = svcntw();   // f32 lanes per vector
    const uint64_t vl_f64 = svcntd();   // f64 lanes per vector  (= vl_f32 / 2)

    printf("  SVE f32 lanes : %lu\n", vl_f32);
    printf("  SVE f64 lanes : %lu\n", vl_f64);

    std::vector<float>  src(vl_f32);
    std::vector<double> ref_even(vl_f64), ref_odd(vl_f64);
    std::vector<double> got_even(vl_f64), got_odd(vl_f64);

    // Fill with easily distinguishable values: even index i → i*0.5+1, odd → i*0.5+100
    for (uint64_t i = 0; i < vl_f32; ++i) {
        if (i % 2 == 0) {
            src[i] = static_cast<float>(i * 0.5f + 1.0f);
        } else {
            src[i] = static_cast<float>(i * 0.5f + 100.0f);
        }
    }

    // Scalar reference
    for (uint64_t k = 0; k < vl_f64; ++k) {
        ref_even[k] = static_cast<double>(src[2 * k]);     // element 0,2,4,...
        ref_odd [k] = static_cast<double>(src[2 * k + 1]); // element 1,3,5,...
    }

    // --- SVE computation ---
    svbool_t pg32 = svptrue_b32();
    svbool_t pg64 = svptrue_b64();

    // Load the f32 vector
    svfloat32_t v32 = svld1_f32(pg32, src.data());

    // Even lanes: svcvt_f64_f32_x
    svfloat64_t v_even = svcvt_f64_f32_x(pg64, v32);
    svst1_f64(pg64, got_even.data(), v_even);

    // Odd lanes: svcvtlt_f64_f32_x
    // The intrinsic reads the upper f32 within each 64-bit element pair,
    // i.e. the odd-indexed f32 elements, and widens them to f64.
    svfloat64_t v_odd  = svcvtlt_f64_f32_x(pg64, v32);
    svst1_f64(pg64, got_odd.data(), v_odd);

    // --- Verify ---
    bool all_even_ok = true, all_odd_ok = true;
    for (uint64_t k = 0; k < vl_f64; ++k) {
        if (got_even[k] != ref_even[k]) {
            printf("  even[%lu]: got %.10g, expected %.10g\n", k, got_even[k], ref_even[k]);
            all_even_ok = false;
        }
        if (got_odd[k] != ref_odd[k]) {
            printf("  odd[%lu]:  got %.10g, expected %.10g\n", k, got_odd[k], ref_odd[k]);
            all_odd_ok = false;
        }
    }
    check(all_even_ok, "svcvt_f64_f32_x  (even elements match scalar cast)");
    check(all_odd_ok,  "svcvtlt_f64_f32_x (odd elements match scalar cast)");
}

/**
 * Test with special floating-point values: 0, -0, ±inf, NaN, denormals,
 * FLT_MAX, -FLT_MAX.
 */
static void test_special_values() {
    printf("=== test_special_values ===\n");

    const uint64_t vl_f32 = svcntw();
    const uint64_t vl_f64 = svcntd();

    // We need at least 16 special values; pad with 1.0f if the vector is smaller.
    static const float specials[] = {
        0.0f, -0.0f,
        std::numeric_limits<float>::infinity(),
       -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::signaling_NaN(),
        std::numeric_limits<float>::min(),        // smallest normal
        std::numeric_limits<float>::denorm_min(), // smallest denormal
        std::numeric_limits<float>::max(),
       -std::numeric_limits<float>::max(),
        1.0f, -1.0f, 0.1f, -0.1f, 1e-30f, 1e30f
    };
    const size_t Ns = sizeof(specials) / sizeof(specials[0]);

    std::vector<float> src(vl_f32, 1.0f);
    for (uint64_t i = 0; i < vl_f32 && i < Ns; ++i)
        src[i] = specials[i];

    svbool_t pg32 = svptrue_b32();
    svbool_t pg64 = svptrue_b64();
    svfloat32_t v32 = svld1_f32(pg32, src.data());

    std::vector<double> got_even(vl_f64), got_odd(vl_f64);
    svst1_f64(pg64, got_even.data(), svcvt_f64_f32_x(pg64, v32));
    svst1_f64(pg64, got_odd.data(),  svcvtlt_f64_f32_x(pg64, v32));

    bool all_ok = true;
    for (uint64_t k = 0; k < vl_f64; ++k) {
        double ref_e = static_cast<double>(src[2 * k]);
        double ref_o = static_cast<double>(src[2 * k + 1]);

        // NaN: both should be NaN
        bool e_ok = (std::isnan(ref_e) && std::isnan(got_even[k])) || (got_even[k] == ref_e);
        bool o_ok = (std::isnan(ref_o) && std::isnan(got_odd[k]))  || (got_odd[k]  == ref_o);

        if (!e_ok) {
            printf("  special even[%lu]: got %.10g, expected %.10g  (src=%.10g)\n",
                   k, got_even[k], ref_e, (double)src[2*k]);
            all_ok = false;
        }
        if (!o_ok) {
            printf("  special odd[%lu]:  got %.10g, expected %.10g  (src=%.10g)\n",
                   k, got_odd[k], ref_o, (double)src[2*k+1]);
            all_ok = false;
        }
    }
    check(all_ok, "special values (0, -0, inf, NaN, denormal, max)");
}

/**
 * Test that interleaving even + odd results reproduces the original values
 * after round-trip f32→f64→f32 conversion. This mirrors the usage pattern
 * in Interpolation_quantizer.inl (svzip1 + svuzp1).
 */
static void test_roundtrip_interleave() {
    printf("=== test_roundtrip_interleave ===\n");

    const uint64_t vl_f32 = svcntw();
    const uint64_t vl_f64 = svcntd();

    std::vector<float> src(vl_f32), result(vl_f32, 0.0f);
    for (uint64_t i = 0; i < vl_f32; ++i)
        src[i] = static_cast<float>(i + 1) * 1.25f;

    svbool_t pg32 = svptrue_b32();
    svbool_t pg64 = svptrue_b64();

    svfloat32_t v32 = svld1_f32(pg32, src.data());

    svfloat64_t v_even64 = svcvt_f64_f32_x(pg64, v32);
    svfloat64_t v_odd64  = svcvtlt_f64_f32_x(pg64, v32);

    // Convert back to f32
    svfloat32_t even_f32 = svcvt_f32_f64_x(pg64, v_even64);
    svfloat32_t odd_f32  = svcvt_f32_f64_x(pg64, v_odd64);

    // Interleave: zip even low halves with odd low halves → original order
    // (mirrors the svzip1 + svuzp1 pattern in the production code)
    svfloat32_t reconstructed = svzip1_f32(
            svuzp1_f32(even_f32, even_f32),
            svuzp1_f32(odd_f32,  odd_f32));
    svst1_f32(pg32, result.data(), reconstructed);

    bool all_ok = true;
    for (uint64_t i = 0; i < vl_f32; ++i) {
        if (result[i] != src[i]) {
            printf("  roundtrip[%lu]: got %.10g, expected %.10g\n",
                   i, (double)result[i], (double)src[i]);
            all_ok = false;
        }
    }
    check(all_ok, "round-trip f32→f64→f32 + interleave matches original");
}

// ---- main -------------------------------------------------------------------
int main() {
    printf("SVE2 svcvt_f64_f32_x / svcvtlt_f64_f32_x correctness test\n");
    printf("----------------------------------------------------------\n");

    test_even_odd_conversion();
    test_special_values();
    test_roundtrip_interleave();

    printf("----------------------------------------------------------\n");
    printf("Results: %d / %d tests passed\n", g_tests - g_fails, g_tests);
    return (g_fails == 0) ? 0 : 1;
}
