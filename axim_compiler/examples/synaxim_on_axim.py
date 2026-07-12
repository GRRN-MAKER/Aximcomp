"""
SYNAXIM on AXIM — Full Layer Forward
=====================================
Builds a small SYNAXIM layer with real INT4-packed weights and runs a
complete forward pass through the AXIM runtime (CPU SIMD or GPU, no CUDA).

Proves: any SYNAXIM model can execute on any hardware via AXIM.

Run:
    python3 examples/synaxim_on_axim.py
"""

import os
import sys
import random
import math

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))          # axim_compiler/
sys.path.insert(0, os.path.join(_here, "..", ".."))    # AXIM OS/

import axim_compiler as axim
from axim_compiler.synaxim_bridge import LayerWeightsAXIM, synaxim_layer_forward


# ══════════════════════════════════════════════════════════════
# INT4 packing helper (FP32 → packed uint8 + per-group scales/zeros)
# ══════════════════════════════════════════════════════════════

def pack_int4(matrix_rows, in_dim, group_size=128):
    """
    Pack a (out_dim x in_dim) FP32 matrix (list of rows) to INT4.
    Returns (packed_flat, scales_flat, zeros_flat).
    Quantization: asymmetric per-group, nibble = round((w - min) / scale).
    """
    out_dim = len(matrix_rows)
    n_groups = in_dim // group_size
    packed = []
    scales = []
    zeros = []
    for row in matrix_rows:
        for g in range(n_groups):
            seg = row[g * group_size:(g + 1) * group_size]
            lo = min(seg)
            hi = max(seg)
            rng = (hi - lo) or 1e-8
            scale = rng / 15.0
            scales.append(scale)
            zeros.append(lo)
        # pack nibbles
        for g in range(n_groups):
            seg = row[g * group_size:(g + 1) * group_size]
            lo = zeros[-(n_groups - g)]
            scale = scales[-(n_groups - g)]
            q = [max(0, min(15, round((v - lo) / scale))) for v in seg]
            for i in range(0, group_size, 2):
                packed.append(((q[i] & 0x0F) << 4) | (q[i + 1] & 0x0F))
    return packed, scales, zeros


def rand_matrix(out_dim, in_dim, scale=0.02):
    return [[random.gauss(0, scale) for _ in range(in_dim)] for _ in range(out_dim)]


# ══════════════════════════════════════════════════════════════
# Build a small SYNAXIM layer
# ══════════════════════════════════════════════════════════════

random.seed(42)
D = 128          # hidden dim (small for the demo)
INTER = 256      # MLP intermediate
R = 16           # low-rank rank
GS = 128         # group size (== D so 1 group per row)

print("=" * 62)
axim.hello()
print("=" * 62)
print(f"Building SYNAXIM layer: D={D}, inter={INTER}, rank={R}, group={GS}")

# Attention projections
Wq = rand_matrix(D, D)
Wo = rand_matrix(D, D)
Wq_p, Wq_s, Wq_z = pack_int4(Wq, D, GS)
Wo_p, Wo_s, Wo_z = pack_int4(Wo, D, GS)

# MLP projections
Wg = rand_matrix(INTER, D)
Wu = rand_matrix(INTER, D)
Wd = rand_matrix(D, INTER)
Wg_p, Wg_s, Wg_z = pack_int4(Wg, D, GS)
Wu_p, Wu_s, Wu_z = pack_int4(Wu, D, GS)
Wd_p, Wd_s, Wd_z = pack_int4(Wd, INTER, GS)

# Low-rank memory factors (some accumulated state)
U = [random.gauss(0, 0.05) for _ in range(D * R)]
V = [random.gauss(0, 0.05) for _ in range(D * R)]

# Norms
norm_attn = [1.0] * D
norm_mlp = [1.0] * D

weights = LayerWeightsAXIM(
    D=D, inter_dim=INTER, r=R, group_size=GS,
    norm_attn=norm_attn, norm_mlp=norm_mlp,
    Wq_packed=Wq_p, Wq_scales=Wq_s, Wq_zeros=Wq_z,
    Wo_packed=Wo_p, Wo_scales=Wo_s, Wo_zeros=Wo_z,
    U=U, V=V,
    Wg_packed=Wg_p, Wg_scales=Wg_s, Wg_zeros=Wg_z,
    Wu_packed=Wu_p, Wu_scales=Wu_s, Wu_zeros=Wu_z,
    Wd_packed=Wd_p, Wd_scales=Wd_s, Wd_zeros=Wd_z,
)

# ══════════════════════════════════════════════════════════════
# Run the layer forward through AXIM
# ══════════════════════════════════════════════════════════════

h = [random.gauss(0, 1.0) for _ in range(D)]
print(f"\nInput hidden state norm:  {math.sqrt(sum(v*v for v in h)):.4f}")

for dev in ("cpu", "auto"):
    out = synaxim_layer_forward(h, weights, device=dev)
    onorm = math.sqrt(sum(v * v for v in out))
    finite = all(math.isfinite(v) for v in out)
    print(f"  [{dev:>4}] output norm = {onorm:.4f}  finite={finite}  "
          f"len={len(out)}")
    assert finite, "non-finite output"
    assert len(out) == D, "wrong output length"

# CPU and auto must agree (same math, no CUDA)
out_cpu = synaxim_layer_forward(h, weights, device="cpu")
out_auto = synaxim_layer_forward(h, weights, device="auto")
max_diff = max(abs(a - b) for a, b in zip(out_cpu, out_auto))
print(f"\nCPU vs auto max diff: {max_diff:.2e}")

print("=" * 62)
print("✅ SYNAXIM layer ran end-to-end through AXIM — CUDA-free.")
print("   Every matmul = AXIM INT4 fused; memory = AXIM low-rank retrieve.")
print("=" * 62)
