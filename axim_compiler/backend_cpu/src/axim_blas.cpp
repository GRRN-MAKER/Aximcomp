/*
 * aximBLAS — Implementation (tuned, CUDA-free)
 * =============================================
 * Cache-blocked GEMM + SIMD GEMV/AXPY/DOT. AVX2 on x86, NEON on ARM,
 * scalar fallback. Blocking sizes chosen for L1/L2 residency.
 */
#include "axim_blas.h"

#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#  define AXIM_X86 1
#elif defined(__aarch64__) || defined(__ARM_NEON)
#  include <arm_neon.h>
#  define AXIM_ARM 1
#endif

/* ── SIMD dot helper ── */
static inline float dot_simd(const float* a, const float* b, int n) {
    int i = 0;
    float acc = 0.0f;
#if defined(AXIM_X86)
    __m256 vacc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vacc = _mm256_fmadd_ps(va, vb, vacc);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, vacc);
    for (int k = 0; k < 8; ++k) acc += tmp[k];
#elif defined(AXIM_ARM)
    float32x4_t vacc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        vacc = vmlaq_f32(vacc, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    acc += vaddvq_f32(vacc);
#endif
    for (; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

/* ── sdot ── */
float axim_sdot(int n, const float* x, const float* y) {
    return dot_simd(x, y, n);
}

/* ── saxpy: y = alpha*x + y ── */
void axim_saxpy(int n, float alpha, const float* x, float* y) {
    int i = 0;
#if defined(AXIM_X86)
    __m256 va = _mm256_set1_ps(alpha);
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, vx, vy));
    }
#elif defined(AXIM_ARM)
    float32x4_t va = vdupq_n_f32(alpha);
    for (; i + 4 <= n; i += 4) {
        float32x4_t vy = vld1q_f32(y + i);
        vst1q_f32(y + i, vmlaq_f32(vy, va, vld1q_f32(x + i)));
    }
#endif
    for (; i < n; ++i) y[i] += alpha * x[i];
}

/* ── sgemv: y = alpha*A*x + beta*y  (A row-major M x N) ── */
void axim_sgemv(int M, int N, float alpha,
                const float* A, const float* x,
                float beta, float* y) {
    for (int r = 0; r < M; ++r) {
        float acc = dot_simd(A + (size_t)r * N, x, N);
        y[r] = alpha * acc + beta * y[r];
    }
}

/* ── sgemm: C = alpha*A*B + beta*C, cache-blocked ──
 * A: MxK, B: KxN, C: MxN, all row-major. Blocking keeps tiles L1/L2 hot. */
void axim_sgemm(int M, int N, int K, float alpha,
                const float* A, const float* B,
                float beta, float* C) {
    const int BM = 64, BN = 64, BK = 64;   // tuned tile sizes

    // scale C by beta first
    if (beta != 1.0f) {
        for (size_t i = 0; i < (size_t)M * N; ++i) C[i] *= beta;
    }

    for (int i0 = 0; i0 < M; i0 += BM) {
        int iMax = (i0 + BM < M) ? i0 + BM : M;
        for (int k0 = 0; k0 < K; k0 += BK) {
            int kMax = (k0 + BK < K) ? k0 + BK : K;
            for (int j0 = 0; j0 < N; j0 += BN) {
                int jMax = (j0 + BN < N) ? j0 + BN : N;
                for (int i = i0; i < iMax; ++i) {
                    for (int k = k0; k < kMax; ++k) {
                        float a = alpha * A[(size_t)i * K + k];
                        const float* brow = B + (size_t)k * N;
                        float* crow = C + (size_t)i * N;
                        int j = j0;
#if defined(AXIM_X86)
                        __m256 va = _mm256_set1_ps(a);
                        for (; j + 8 <= jMax; j += 8) {
                            __m256 vb = _mm256_loadu_ps(brow + j);
                            __m256 vc = _mm256_loadu_ps(crow + j);
                            _mm256_storeu_ps(crow + j, _mm256_fmadd_ps(va, vb, vc));
                        }
#elif defined(AXIM_ARM)
                        float32x4_t va = vdupq_n_f32(a);
                        for (; j + 4 <= jMax; j += 4) {
                            float32x4_t vc = vld1q_f32(crow + j);
                            vst1q_f32(crow + j, vmlaq_f32(vc, va, vld1q_f32(brow + j)));
                        }
#endif
                        for (; j < jMax; ++j) crow[j] += a * brow[j];
                    }
                }
            }
        }
    }
}

/* ── BLAS Level-1 completions ──
 * Portable, -O3 auto-vectorized. Kept simple and correct; the compiler
 * generates AVX/NEON here just as for the hand-tuned kernels above. */
#include <cmath>

extern "C" void axim_sscal(int n, float alpha, float* x) {
    for (int i = 0; i < n; ++i) x[i] *= alpha;
}

extern "C" float axim_snrm2(int n, const float* x) {
    /* two-pass-free, overflow-aware would use scaling; this is the common
     * fast path used across BLAS for well-scaled inputs. */
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)x[i] * (double)x[i];
    return (float)std::sqrt(s);
}

extern "C" float axim_sasum(int n, const float* x) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::fabs((double)x[i]);
    return (float)s;
}

extern "C" int axim_isamax(int n, const float* x) {
    if (n <= 0) return -1;
    int best = 0;
    float bestv = std::fabs(x[0]);
    for (int i = 1; i < n; ++i) {
        float v = std::fabs(x[i]);
        if (v > bestv) { bestv = v; best = i; }
    }
    return best;
}
