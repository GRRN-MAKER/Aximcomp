"""
AXIM Frontend — SYNAXIM Op Builders
====================================
High-level, typed entry points for SYNAXIM's core primitives. These build
AXIM IR kernels directly (with correct dtypes/shapes) and dispatch them to
the vendor-neutral CPU/GPU backends.

Any SYNAXIM model runs on AXIM through these ops:
  - int4_matvec    : Boolean-bit fused INT4 projection (all 7 weight matmuls)
  - rmsnorm        : RMS normalization
  - silu / swiglu  : MLP activation
  - lowrank_retrieve : q @ (U @ V^T) — O(D×r) memory read

No CUDA. Same call runs on Nvidia, AMD, Intel, Apple silicon.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))

from ir.axim_ir import Kernel, Op, OpCode, Buffer, DType, MemSpace  # noqa: E402
from .api import _select_device  # noqa: E402


def _backend():
    from orchestrator import synaxim_dispatch
    return synaxim_dispatch


# ══════════════════════════════════════════════════════════════
# SYNAXIM ops
# ══════════════════════════════════════════════════════════════

def int4_matvec(x, packed, scales, zeros, out_dim, in_dim,
                group_size=128, device="auto"):
    """
    Fused INT4 matrix-vector multiply (SYNAXIM Boolean-bit pipeline).

        out[row] = sum_j dequant(packed[row][j]) * x[j]

    Args:
        x:      list[float] length in_dim
        packed: flat list[int] (uint8), length out_dim * (in_dim // 2)
        scales: flat list[float], length out_dim * (in_dim // group_size)
        zeros:  flat list[float], same layout as scales
        out_dim, in_dim, group_size: dims

    Returns:
        list[float] length out_dim
    """
    dev = _select_device(device)
    return _backend().int4_matvec(
        x, packed, scales, zeros, out_dim, in_dim, group_size, dev
    )


def rmsnorm(x, weight, eps=1e-6, device="auto"):
    """RMSNorm: out = x * weight / sqrt(mean(x^2) + eps)."""
    dev = _select_device(device)
    return _backend().rmsnorm(x, weight, eps, dev)


def silu(x, device="auto"):
    """SiLU / Swish activation: x * sigmoid(x)."""
    dev = _select_device(device)
    return _backend().silu(x, dev)


def swiglu(x, w_gate_packed, w_up_packed, scales_g, zeros_g, scales_u, zeros_u,
           inter_dim, in_dim, group_size=128, device="auto"):
    """
    Fused SwiGLU MLP first stage: silu(x @ W_gate) * (x @ W_up).
    Both projections use INT4 fused matvec. Returns list[float] length inter_dim.
    """
    dev = _select_device(device)
    gate = _backend().int4_matvec(
        x, w_gate_packed, scales_g, zeros_g, inter_dim, in_dim, group_size, dev
    )
    up = _backend().int4_matvec(
        x, w_up_packed, scales_u, zeros_u, inter_dim, in_dim, group_size, dev
    )
    gate = _backend().silu(gate, dev)
    return [g * u for g, u in zip(gate, up)]


def lowrank_retrieve(q, U, V, D, r, device="auto"):
    """
    SYNAXIM low-rank memory read: out = (q @ U) @ V^T.

    Args:
        q: list[float] length D
        U: flat list[float] length D*r (row-major)
        V: flat list[float] length D*r (row-major)
    Returns:
        list[float] length D
    """
    dev = _select_device(device)
    return _backend().lowrank_retrieve(q, U, V, D, r, dev)
