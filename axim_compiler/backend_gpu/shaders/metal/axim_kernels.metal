//
// AXIM Metal Compute Shaders
// ==========================
// Vendor-neutral compute kernels for Apple silicon GPUs (Metal).
// Mirror the Vulkan SPIR-V kernels exactly, so the same AXIM IR runs
// identically on Apple GPUs and Nvidia/AMD/Intel GPUs. Zero CUDA.
//
// Compile to a .metallib:
//   xcrun -sdk macosx metal   -c axim_kernels.metal -o axim_kernels.air
//   xcrun -sdk macosx metallib   axim_kernels.air   -o axim_kernels.metallib
//

#include <metal_stdlib>
using namespace metal;

// ── Elementwise add: out[i] = a[i] + b[i] ──
kernel void axim_add(device const float* a   [[buffer(0)]],
                     device const float* b   [[buffer(1)]],
                     device float*       out [[buffer(2)]],
                     constant uint&      n   [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < n) out[gid] = a[gid] + b[gid];
}

// ── Elementwise mul ──
kernel void axim_mul(device const float* a   [[buffer(0)]],
                     device const float* b   [[buffer(1)]],
                     device float*       out [[buffer(2)]],
                     constant uint&      n   [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < n) out[gid] = a[gid] * b[gid];
}

// ── SiLU: out = x * sigmoid(x) ──
kernel void axim_silu(device const float* x   [[buffer(0)]],
                      device float*       out [[buffer(1)]],
                      constant uint&      n   [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < n) {
        float v = x[gid];
        float c = clamp(v, -60.0f, 60.0f);
        out[gid] = v * (1.0f / (1.0f + exp(-c)));
    }
}

// ── Fused INT4 matvec (SYNAXIM Boolean-bit pipeline) ──
// One thread per output row.
//   packed: (out_dim, in_dim/2) uint8 flattened as uchar
//   scales/zeros: (out_dim, n_groups)
kernel void axim_int4_matvec(
        device const float* x          [[buffer(0)]],
        device const uchar* packed     [[buffer(1)]],
        device const float* scales     [[buffer(2)]],
        device const float* zeros      [[buffer(3)]],
        device float*       out        [[buffer(4)]],
        constant uint&      out_dim    [[buffer(5)]],
        constant uint&      in_dim     [[buffer(6)]],
        constant uint&      group_size [[buffer(7)]],
        uint row [[thread_position_in_grid]]) {
    if (row >= out_dim) return;
    uint n_groups = in_dim / group_size;
    uint row_half = in_dim / 2;
    float acc = 0.0f;
    for (uint g = 0; g < n_groups; ++g) {
        float s = scales[row * n_groups + g];
        float z = zeros[row * n_groups + g];
        uint base_col = g * group_size;
        uint base_byte = base_col / 2;
        for (uint i = 0; i < group_size / 2; ++i) {
            uchar byte = packed[row * row_half + base_byte + i];
            float w_hi = float((byte >> 4) & 0x0F) * s + z;  // Boolean extract
            float w_lo = float(byte & 0x0F) * s + z;         // Boolean extract
            acc += w_hi * x[base_col + i * 2];
            acc += w_lo * x[base_col + i * 2 + 1];
        }
    }
    out[row] = acc;
}

// ── RMSNorm (one threadgroup reduces, simple version: one thread) ──
// For large D use a reduction; this correctness-first version runs per-elem
// after a group sum computed on the host or via a two-pass kernel.
kernel void axim_rmsnorm_apply(
        device const float* x       [[buffer(0)]],
        device const float* weight  [[buffer(1)]],
        device float*       out     [[buffer(2)]],
        constant float&     inv_rms [[buffer(3)]],
        constant uint&      n       [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid < n) out[gid] = x[gid] * inv_rms * weight[gid];
}
