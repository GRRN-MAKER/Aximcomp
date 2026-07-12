"""
SYNAXIM-on-AXIM Bridge
======================
Runs a full SYNAXIM transformer-replacement layer through the AXIM runtime.
Every matmul is an AXIM INT4 fused matvec; the memory read is an AXIM
low-rank retrieve; normalization and activation are AXIM ops.

Result: any SYNAXIM model executes on any hardware (Nvidia, AMD, Intel,
Apple silicon) with zero CUDA.

Layer forward (single token):
    h_norm  = rmsnorm(h, norm_attn)
    q       = int4_matvec(h_norm, W_q)          # AXIM
    attn    = lowrank_retrieve(q, U, V)         # AXIM (O(D×r))
    o       = int4_matvec(attn, W_o)            # AXIM
    h       = h + o                             # residual
    h_norm  = rmsnorm(h, norm_mlp)
    mlp     = swiglu(h_norm, W_gate, W_up) -> int4_matvec(_, W_down)  # AXIM
    h       = h + mlp
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, ".."))

import axim_compiler as axim


@dataclass
class LayerWeightsAXIM:
    """INT4-packed weights for one SYNAXIM layer, laid out for AXIM."""
    D: int
    inter_dim: int
    r: int
    group_size: int
    # norms (fp32)
    norm_attn: List[float]
    norm_mlp: List[float]
    # attention projections (INT4 packed + scales/zeros)
    Wq_packed: List[int]; Wq_scales: List[float]; Wq_zeros: List[float]
    Wo_packed: List[int]; Wo_scales: List[float]; Wo_zeros: List[float]
    # low-rank memory factors (fp32, D×r row-major)
    U: List[float]
    V: List[float]
    # MLP projections
    Wg_packed: List[int]; Wg_scales: List[float]; Wg_zeros: List[float]
    Wu_packed: List[int]; Wu_scales: List[float]; Wu_zeros: List[float]
    Wd_packed: List[int]; Wd_scales: List[float]; Wd_zeros: List[float]


def synaxim_layer_forward(h: List[float], w: LayerWeightsAXIM,
                          device: str = "auto") -> List[float]:
    """
    Execute one SYNAXIM layer for a single token through AXIM.
    Returns the updated hidden state (length D).
    """
    D, r, gs = w.D, w.r, w.group_size

    # ── Attention block ──
    h_norm = axim.rmsnorm(h, w.norm_attn, device=device)

    q = axim.int4_matvec(
        h_norm, w.Wq_packed, w.Wq_scales, w.Wq_zeros,
        out_dim=D, in_dim=D, group_size=gs, device=device,
    )

    # Low-rank associative memory read (O(D×r))
    attn = axim.lowrank_retrieve(q, w.U, w.V, D, r, device=device)

    o = axim.int4_matvec(
        attn, w.Wo_packed, w.Wo_scales, w.Wo_zeros,
        out_dim=D, in_dim=D, group_size=gs, device=device,
    )

    h = [h[i] + o[i] for i in range(D)]  # residual

    # ── MLP block ──
    h_norm = axim.rmsnorm(h, w.norm_mlp, device=device)

    swig = axim.swiglu(
        h_norm,
        w.Wg_packed, w.Wu_packed,
        w.Wg_scales, w.Wg_zeros, w.Wu_scales, w.Wu_zeros,
        inter_dim=w.inter_dim, in_dim=D, group_size=gs, device=device,
    )

    mlp = axim.int4_matvec(
        swig, w.Wd_packed, w.Wd_scales, w.Wd_zeros,
        out_dim=D, in_dim=w.inter_dim, group_size=gs, device=device,
    )

    h = [h[i] + mlp[i] for i in range(D)]  # residual
    return h
