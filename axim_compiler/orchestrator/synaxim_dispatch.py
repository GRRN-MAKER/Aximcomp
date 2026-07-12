"""
AXIM Orchestrator — SYNAXIM Op Dispatch
========================================
Executes SYNAXIM's core kernels (INT4 matvec, RMSNorm, SiLU, low-rank
retrieve) on the vendor-neutral CPU/GPU backends via ctypes, with a
pure-Python reference fallback so the path always runs.

This is the layer that lets ANY SYNAXIM model run on ANY hardware
through AXIM — Nvidia, AMD, Intel, Apple silicon — with zero CUDA.
"""

from __future__ import annotations

import os
import sys
import ctypes
import math
from typing import List

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))

from .dispatch import _load_native, load_metal  # CPU + Metal GPU loaders


# ══════════════════════════════════════════════════════════════
# ctypes helpers
# ══════════════════════════════════════════════════════════════

def _f32(data) -> "ctypes.Array":
    return (ctypes.c_float * len(data))(*[float(x) for x in data])


def _u8(data) -> "ctypes.Array":
    return (ctypes.c_u8 if hasattr(ctypes, "c_u8") else ctypes.c_uint8 * len(data))(
        *[int(x) & 0xFF for x in data]
    )


def _fp(arr):
    return ctypes.cast(arr, ctypes.POINTER(ctypes.c_float))


def _up(arr):
    return ctypes.cast(arr, ctypes.POINTER(ctypes.c_uint8))


def _uses_native(dev) -> bool:
    # GPU backend shaders for these ops land later; CPU native path is live.
    return bool(_load_native()) and dev.kind == "cpu"


# ══════════════════════════════════════════════════════════════
# INT4 fused matvec
# ══════════════════════════════════════════════════════════════

def int4_matvec(x, packed, scales, zeros, out_dim, in_dim, group_size, dev):
    # ── GPU path: run on real Metal device (Apple silicon) ──
    if dev.kind == "gpu":
        mlib = load_metal()
        if mlib:
            cx = _f32(x)
            cpacked = (ctypes.c_uint8 * len(packed))(*[int(v) & 0xFF for v in packed])
            cscales = _f32(scales)
            czeros = _f32(zeros)
            cout = (ctypes.c_float * out_dim)()
            rc = mlib.axim_metal_int4_matvec(
                _fp(cx), _up(cpacked), _fp(cscales), _fp(czeros), _fp(cout),
                out_dim, in_dim, group_size,
            )
            if rc == 0:
                return list(cout)
            # fall through to CPU on GPU failure

    # ── CPU path: native SIMD ──
    lib = _load_native()
    if lib:
        cx = _f32(x)
        cpacked = (ctypes.c_uint8 * len(packed))(*[int(v) & 0xFF for v in packed])
        cscales = _f32(scales)
        czeros = _f32(zeros)
        cout = (ctypes.c_float * out_dim)()
        lib.axim_cpu_int4_matvec(
            _fp(cx), _up(cpacked), _fp(cscales), _fp(czeros), _fp(cout),
            out_dim, in_dim, group_size,
        )
        return list(cout)
    return _int4_matvec_py(x, packed, scales, zeros, out_dim, in_dim, group_size)


def _int4_matvec_py(x, packed, scales, zeros, out_dim, in_dim, group_size):
    n_groups = in_dim // group_size
    out = [0.0] * out_dim
    half = in_dim // 2
    for row in range(out_dim):
        acc = 0.0
        for g in range(n_groups):
            s = scales[row * n_groups + g]
            z = zeros[row * n_groups + g]
            base_col = g * group_size
            base_byte = base_col // 2
            for i in range(group_size // 2):
                byte = packed[row * half + base_byte + i]
                w_hi = ((byte >> 4) & 0x0F) * s + z   # Boolean bit extract
                w_lo = (byte & 0x0F) * s + z           # Boolean bit extract
                acc += w_hi * x[base_col + i * 2]
                acc += w_lo * x[base_col + i * 2 + 1]
        out[row] = acc
    return out


# ══════════════════════════════════════════════════════════════
# RMSNorm
# ══════════════════════════════════════════════════════════════

def rmsnorm(x, weight, eps, dev):
    lib = _load_native()
    n = len(x)
    if _uses_native(dev):
        cx = _f32(x)
        cw = _f32(weight)
        cout = (ctypes.c_float * n)()
        lib.axim_cpu_rmsnorm(_fp(cx), _fp(cw), _fp(cout), n, ctypes.c_float(eps))
        return list(cout)
    var = sum(v * v for v in x) / n
    inv = 1.0 / math.sqrt(var + eps)
    return [x[i] * inv * weight[i] for i in range(n)]


# ══════════════════════════════════════════════════════════════
# SiLU
# ══════════════════════════════════════════════════════════════

def silu(x, dev):
    lib = _load_native()
    n = len(x)
    if _uses_native(dev):
        cx = _f32(x)
        cout = (ctypes.c_float * n)()
        lib.axim_cpu_silu(_fp(cx), _fp(cout), n)
        return list(cout)
    out = []
    for v in x:
        c = max(-60.0, min(60.0, v))
        out.append(v * (1.0 / (1.0 + math.exp(-c))))
    return out


# ══════════════════════════════════════════════════════════════
# Low-rank retrieve
# ══════════════════════════════════════════════════════════════

def lowrank_retrieve(q, U, V, D, r, dev):
    lib = _load_native()
    if _uses_native(dev):
        cq = _f32(q)
        cU = _f32(U)
        cV = _f32(V)
        cout = (ctypes.c_float * D)()
        lib.axim_cpu_lowrank_retrieve(_fp(cq), _fp(cU), _fp(cV), _fp(cout), D, r)
        return list(cout)
    # Python reference: tmp = q @ U ; out = tmp @ V^T
    tmp = [0.0] * r
    for c in range(r):
        acc = 0.0
        for d in range(D):
            acc += q[d] * U[d * r + c]
        tmp[c] = acc
    out = [0.0] * D
    for d in range(D):
        acc = 0.0
        for c in range(r):
            acc += tmp[c] * V[d * r + c]
        out[d] = acc
    return out
