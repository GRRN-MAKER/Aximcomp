/*
 * aximDNN — Tuned Deep Neural Network Primitives (the CUDA-free cuDNN)
 * ====================================================================
 * High-performance, fused, SIMD-vectorized DNN kernels. Vendor-neutral.
 * Zero CUDA. Provides the ops LLMs and vision models need at peak speed:
 *
 *   softmax        — numerically-stable row softmax
 *   layernorm      — mean/variance normalization + affine
 *   gelu           — GELU activation (tanh approx)
 *   attention_step — single-token attention with a running K/V state
 *
 * Like cuDNN gives Nvidia its DNN speed, aximDNN gives every CPU/GPU the
 * same tuned primitives through one API.
 */
#ifndef AXIM_DNN_H
#define AXIM_DNN_H

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/* Numerically-stable softmax over a length-n vector (in place-safe). */
void axim_softmax(const float* x, float* out, int n);

/* LayerNorm: out = ((x - mean) / sqrt(var + eps)) * gamma + beta. */
void axim_layernorm(const float* x, const float* gamma, const float* beta,
                    float* out, int n, float eps);

/* GELU activation (tanh approximation), elementwise. */
void axim_gelu(const float* x, float* out, int n);

/*
 * Single-token scaled dot-product attention against a running K/V cache.
 *   q:     (d_head)
 *   K:     (seq_len, d_head)   accumulated keys
 *   V:     (seq_len, d_head)   accumulated values
 *   out:   (d_head)            attention output
 * Computes softmax(q·Kᵀ / sqrt(d_head)) · V. Tuned for inference.
 */
void axim_attention_step(const float* q, const float* K, const float* V,
                         float* out, int seq_len, int d_head);

#ifdef __cplusplus
}
#endif

#endif /* AXIM_DNN_H */
