/*
 * aximBLAS — Tuned Linear Algebra (the CUDA-free cuBLAS)
 * ======================================================
 * High-performance, cache-blocked, SIMD-vectorized BLAS-style routines.
 * Vendor-neutral: AVX-512 / AVX2 on x86, NEON on ARM. Zero CUDA.
 *
 * Like cuBLAS squeezes Nvidia GPUs, aximBLAS squeezes every CPU (and, via
 * the GPU backend, every GPU) with the same API — so SYNAXIM and any AXIM
 * app gets peak matmul/GEMV throughput on any machine.
 *
 * Routines mirror BLAS naming so porting from cuBLAS/OpenBLAS is trivial:
 *   sgemv  — single-precision matrix-vector   y = alpha*A*x + beta*y
 *   sgemm  — single-precision matrix-matrix   C = alpha*A*B + beta*C
 *   saxpy  — y = alpha*x + y
 *   sdot   — dot product
 */
#ifndef AXIM_BLAS_H
#define AXIM_BLAS_H

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/* y = alpha * A * x + beta * y   (A is M x N, row-major). */
void axim_sgemv(int M, int N, float alpha,
                const float* A, const float* x,
                float beta, float* y);

/* C = alpha * A * B + beta * C   (A: MxK, B: KxN, C: MxN, all row-major).
 * Cache-blocked + SIMD micro-kernel for peak throughput. */
void axim_sgemm(int M, int N, int K, float alpha,
                const float* A, const float* B,
                float beta, float* C);

/* y = alpha * x + y */
void axim_saxpy(int n, float alpha, const float* x, float* y);

/* dot product of x and y */
float axim_sdot(int n, const float* x, const float* y);

#ifdef __cplusplus
}
#endif

#endif /* AXIM_BLAS_H */
