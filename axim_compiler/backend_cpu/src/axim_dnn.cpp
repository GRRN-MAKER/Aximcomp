/*
 * aximDNN — Implementation (tuned, CUDA-free)
 * ============================================
 * Fused, numerically-stable DNN primitives with SIMD where it helps.
 */
#include "axim_dnn.h"
#include "axim_blas.h"   // reuse tuned sdot for attention scores

#include <cmath>

/* ── softmax (stable) ── */
void axim_softmax(const float* x, float* out, int n) {
    float m = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > m) m = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float e = std::exp(x[i] - m);
        out[i] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) out[i] *= inv;
}

/* ── layernorm ── */
void axim_layernorm(const float* x, const float* gamma, const float* beta,
                    float* out, int n, float eps) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    double var = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = x[i] - mean;
        var += d * d;
    }
    var /= n;
    float inv = 1.0f / std::sqrt((float)var + eps);
    for (int i = 0; i < n; ++i)
        out[i] = ((x[i] - (float)mean) * inv) * gamma[i] + beta[i];
}

/* ── GELU (tanh approx) ── */
void axim_gelu(const float* x, float* out, int n) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        float inner = c * (v + 0.044715f * v * v * v);
        out[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
}

/* ── single-token attention step ── */
void axim_attention_step(const float* q, const float* K, const float* V,
                         float* out, int seq_len, int d_head) {
    float scale = 1.0f / std::sqrt((float)d_head);

    // scores[t] = (q · K[t]) * scale
    // (stack alloc for common seq lengths; heap for large)
    float stackbuf[1024];
    float* scores = (seq_len <= 1024) ? stackbuf : new float[seq_len];
    for (int t = 0; t < seq_len; ++t) {
        float s = axim_sdot(d_head, q, K + (size_t)t * d_head);
        scores[t] = s * scale;
    }

    // softmax over scores
    axim_softmax(scores, scores, seq_len);

    // out = sum_t scores[t] * V[t]
    for (int d = 0; d < d_head; ++d) out[d] = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
        float w = scores[t];
        const float* vrow = V + (size_t)t * d_head;
        for (int d = 0; d < d_head; ++d) out[d] += w * vrow[d];
    }

    if (seq_len > 1024) delete[] scores;
}

/* ── aximDNN completions: SiLU, RMSNorm, SwiGLU ── */
#include <cmath>

extern "C" void axim_silu(const float* x, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        out[i] = v / (1.0f + std::exp(-v));   /* v * sigmoid(v) */
    }
}

extern "C" void axim_rmsnorm(const float* x, const float* weight,
                             float* out, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * (double)x[i];
    float inv_rms = 1.0f / std::sqrt((float)(ss / (double)n) + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv_rms * weight[i];
}

extern "C" void axim_swiglu(const float* gate, const float* up,
                            float* out, int n) {
    for (int i = 0; i < n; ++i) {
        float g = gate[i];
        float silu = g / (1.0f + std::exp(-g));
        out[i] = silu * up[i];
    }
}
