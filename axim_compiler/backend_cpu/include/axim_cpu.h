/*
 * AXIM CPU Backend — SIMD Execution Engine
 * =========================================
 * Vendor-neutral CPU kernels using AVX-512 / AVX2 (x86) or NEON (ARM).
 * Zero CUDA. Compiles the same AXIM IR ops that the GPU backend runs,
 * so any SYNAXIM model executes identically on CPU or GPU.
 *
 * Targets: Intel, AMD (x86 SIMD) and Apple silicon / ARM (NEON).
 */
#ifndef AXIM_CPU_H
#define AXIM_CPU_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/* Detected SIMD instruction set at runtime. */
typedef enum {
    AXIM_SIMD_SCALAR = 0,
    AXIM_SIMD_AVX2   = 1,
    AXIM_SIMD_AVX512 = 2,
    AXIM_SIMD_NEON   = 3
} axim_simd_t;

/* Return the best SIMD ISA available on this CPU. */
axim_simd_t axim_cpu_detect_simd(void);

/* Human-readable name of the active backend, e.g. "AVX-512", "NEON". */
const char* axim_cpu_backend_name(void);

/* ── Core elementwise ops (SIMD-accelerated) ── */
void axim_cpu_add(const float* a, const float* b, float* out, size_t n);
void axim_cpu_mul(const float* a, const float* b, float* out, size_t n);
void axim_cpu_sub(const float* a, const float* b, float* out, size_t n);
void axim_cpu_silu(const float* x, float* out, size_t n);

/* ── SYNAXIM-native ops ── */

/* RMSNorm: out = x * weight / sqrt(mean(x^2) + eps). */
void axim_cpu_rmsnorm(const float* x, const float* weight, float* out,
                      size_t n, float eps);

/*
 * Fused INT4 matvec (SYNAXIM Boolean-bit pipeline):
 *   out[row] = sum_j dequant(packed[row][j]) * x[j]
 * Nibbles extracted via (byte >> 4) & 0x0F and byte & 0x0F.
 * packed: (out_dim, in_dim/2) uint8; scales/zeros: (out_dim, n_groups).
 */
void axim_cpu_int4_matvec(const float* x, const uint8_t* packed,
                          const float* scales, const float* zeros,
                          float* out, size_t out_dim, size_t in_dim,
                          size_t group_size);

/*
 * Low-rank retrieve: out = (q @ U) @ V^T
 * U, V: (D, r) row-major. Cost O(2*D*r), no D*D materialization.
 */
void axim_cpu_lowrank_retrieve(const float* q, const float* U, const float* V,
                               float* out, size_t D, size_t r);

/* Scaffold sanity kernel: writes device+backend banner into out. */
int axim_cpu_hello(char* out, size_t out_cap);

#ifdef __cplusplus
}
#endif

#endif /* AXIM_CPU_H */
