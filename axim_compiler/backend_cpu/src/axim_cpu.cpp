/*
 * AXIM CPU Backend — Implementation
 * ==================================
 * SIMD kernels with runtime ISA dispatch. AVX-512/AVX2 on x86, NEON on ARM,
 * scalar fallback everywhere. No CUDA, no vendor lock-in.
 */
#include "axim_cpu.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <string>

/* ── SIMD headers, guarded by architecture ── */
#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#  define AXIM_X86 1
#elif defined(__aarch64__) || defined(__ARM_NEON)
#  include <arm_neon.h>
#  define AXIM_ARM 1
#endif

/* ── Runtime SIMD detection ── */
axim_simd_t axim_cpu_detect_simd(void) {
#if defined(AXIM_ARM)
    return AXIM_SIMD_NEON;
#elif defined(AXIM_X86)
    /* __builtin_cpu_supports is available on GCC/Clang. */
#  if defined(__GNUC__) || defined(__clang__)
    if (__builtin_cpu_supports("avx512f")) return AXIM_SIMD_AVX512;
    if (__builtin_cpu_supports("avx2"))    return AXIM_SIMD_AVX2;
#  endif
    return AXIM_SIMD_SCALAR;
#else
    return AXIM_SIMD_SCALAR;
#endif
}

const char* axim_cpu_backend_name(void) {
    switch (axim_cpu_detect_simd()) {
        case AXIM_SIMD_AVX512: return "AVX-512";
        case AXIM_SIMD_AVX2:   return "AVX2";
        case AXIM_SIMD_NEON:   return "NEON";
        default:               return "scalar";
    }
}

/* ── Elementwise: add ── */
void axim_cpu_add(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
#if defined(AXIM_X86)
    if (axim_cpu_detect_simd() >= AXIM_SIMD_AVX2) {
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
        }
    }
#elif defined(AXIM_ARM)
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(va, vb));
    }
#endif
    for (; i < n; ++i) out[i] = a[i] + b[i];
}

/* ── Elementwise: mul ── */
void axim_cpu_mul(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
#if defined(AXIM_X86)
    if (axim_cpu_detect_simd() >= AXIM_SIMD_AVX2) {
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
        }
    }
#elif defined(AXIM_ARM)
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
#endif
    for (; i < n; ++i) out[i] = a[i] * b[i];
}

/* ── Elementwise: sub ── */
void axim_cpu_sub(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
#if defined(AXIM_X86)
    if (axim_cpu_detect_simd() >= AXIM_SIMD_AVX2) {
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(out + i, _mm256_sub_ps(va, vb));
        }
    }
#elif defined(AXIM_ARM)
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
#endif
    for (; i < n; ++i) out[i] = a[i] - b[i];
}

/* ── SiLU: x * sigmoid(x) (scalar; SIMD sigmoid is a later optimization) ── */
void axim_cpu_silu(const float* x, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float v = x[i];
        float clamped = v < -60.f ? -60.f : (v > 60.f ? 60.f : v);
        float sig = 1.0f / (1.0f + std::exp(-clamped));
        out[i] = v * sig;
    }
}

/* ── RMSNorm ── */
void axim_cpu_rmsnorm(const float* x, const float* weight, float* out,
                      size_t n, float eps) {
    double var = 0.0;
    for (size_t i = 0; i < n; ++i) var += (double)x[i] * x[i];
    var /= (double)n;
    float inv_rms = 1.0f / std::sqrt((float)var + eps);
    for (size_t i = 0; i < n; ++i) out[i] = x[i] * inv_rms * weight[i];
}

/* ── Fused INT4 matvec (SYNAXIM Boolean-bit pipeline) ── */
void axim_cpu_int4_matvec(const float* x, const uint8_t* packed,
                          const float* scales, const float* zeros,
                          float* out, size_t out_dim, size_t in_dim,
                          size_t group_size) {
    size_t n_groups = in_dim / group_size;
    for (size_t row = 0; row < out_dim; ++row) {
        float acc = 0.0f;
        for (size_t g = 0; g < n_groups; ++g) {
            float s = scales[row * n_groups + g];
            float z = zeros[row * n_groups + g];
            size_t base_col = g * group_size;
            size_t base_byte = base_col / 2;
            for (size_t i = 0; i < group_size / 2; ++i) {
                uint8_t byte = packed[row * (in_dim / 2) + base_byte + i];
                float w_hi = (float)((byte >> 4) & 0x0F) * s + z;  /* Boolean */
                float w_lo = (float)(byte & 0x0F) * s + z;         /* Boolean */
                acc += w_hi * x[base_col + i * 2];
                acc += w_lo * x[base_col + i * 2 + 1];
            }
        }
        out[row] = acc;
    }
}

/* ── Low-rank retrieve: out = (q @ U) @ V^T ── */
void axim_cpu_lowrank_retrieve(const float* q, const float* U, const float* V,
                               float* out, size_t D, size_t r) {
    /* tmp = q @ U  (length r) */
    float tmp_stack[512];
    float* tmp = (r <= 512) ? tmp_stack : new float[r];
    for (size_t c = 0; c < r; ++c) {
        float acc = 0.0f;
        for (size_t d = 0; d < D; ++d) acc += q[d] * U[d * r + c];
        tmp[c] = acc;
    }
    /* out = tmp @ V^T  (length D) */
    for (size_t d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (size_t c = 0; c < r; ++c) acc += tmp[c] * V[d * r + c];
        out[d] = acc;
    }
    if (r > 512) delete[] tmp;
}

/* ── Scaffold sanity kernel ── */
int axim_cpu_hello(char* out, size_t out_cap) {
    int written = std::snprintf(
        out, out_cap,
        "AXIM CPU backend online — SIMD=%s — CUDA-free, vendor-neutral.",
        axim_cpu_backend_name());
    return written;
}
