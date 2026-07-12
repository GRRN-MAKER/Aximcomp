"""
AXIM ↔ SYNAXIM Op Tests
=======================
Verifies AXIM's SYNAXIM ops (INT4 matvec, RMSNorm, SiLU, low-rank retrieve)
match a pure reference implementation — i.e. any SYNAXIM model computed
through AXIM produces the correct result on CPU (native SIMD) with no CUDA.

Run:
    python3 tests/test_synaxim.py
"""

import os
import sys
import math
import random

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))          # axim_compiler/
sys.path.insert(0, os.path.join(_here, "..", ".."))    # AXIM OS/

import axim_compiler as axim

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}  {detail}")


def approx(a, b, tol=1e-4):
    return abs(a - b) <= tol * (1 + abs(b))


# ── reference implementations ──

def ref_rmsnorm(x, w, eps=1e-6):
    var = sum(v * v for v in x) / len(x)
    inv = 1.0 / math.sqrt(var + eps)
    return [x[i] * inv * w[i] for i in range(len(x))]


def ref_silu(x):
    return [v * (1.0 / (1.0 + math.exp(-max(-60, min(60, v))))) for v in x]


def ref_lowrank(q, U, V, D, r):
    tmp = [sum(q[d] * U[d * r + c] for d in range(D)) for c in range(r)]
    return [sum(tmp[c] * V[d * r + c] for c in range(r)) for d in range(D)]


def pack_int4_row(row, in_dim, group_size):
    """Pack one row; returns (packed_bytes, scales, zeros)."""
    n_groups = in_dim // group_size
    scales, zeros, packed = [], [], []
    for g in range(n_groups):
        seg = row[g * group_size:(g + 1) * group_size]
        lo, hi = min(seg), max(seg)
        rng = (hi - lo) or 1e-8
        scale = rng / 15.0
        scales.append(scale)
        zeros.append(lo)
        q = [max(0, min(15, round((v - lo) / scale))) for v in seg]
        for i in range(0, group_size, 2):
            packed.append(((q[i] & 0x0F) << 4) | (q[i + 1] & 0x0F))
    return packed, scales, zeros


def ref_int4_matvec(x, rows, in_dim, group_size):
    """Reference: dequantize each row then dot with x."""
    out = []
    for row in rows:
        packed, scales, zeros = pack_int4_row(row, in_dim, group_size)
        n_groups = in_dim // group_size
        acc = 0.0
        for g in range(n_groups):
            s, z = scales[g], zeros[g]
            base = g * group_size
            bbyte = base // 2
            for i in range(group_size // 2):
                byte = packed[bbyte + i]
                w_hi = ((byte >> 4) & 0x0F) * s + z
                w_lo = (byte & 0x0F) * s + z
                acc += w_hi * x[base + i * 2]
                acc += w_lo * x[base + i * 2 + 1]
        out.append(acc)
    return out


# ── tests ──

def test_rmsnorm():
    x = [random.gauss(0, 1) for _ in range(64)]
    w = [random.uniform(0.5, 1.5) for _ in range(64)]
    got = axim.rmsnorm(x, w, device="cpu")
    exp = ref_rmsnorm(x, w)
    ok = all(approx(g, e) for g, e in zip(got, exp))
    check("rmsnorm matches reference", ok)


def test_silu():
    x = [random.gauss(0, 2) for _ in range(64)]
    got = axim.silu(x, device="cpu")
    exp = ref_silu(x)
    ok = all(approx(g, e) for g, e in zip(got, exp))
    check("silu matches reference", ok)


def test_lowrank():
    D, r = 96, 12
    q = [random.gauss(0, 1) for _ in range(D)]
    U = [random.gauss(0, 0.1) for _ in range(D * r)]
    V = [random.gauss(0, 0.1) for _ in range(D * r)]
    got = axim.lowrank_retrieve(q, U, V, D, r, device="cpu")
    exp = ref_lowrank(q, U, V, D, r)
    ok = all(approx(g, e) for g, e in zip(got, exp))
    check("lowrank_retrieve matches reference", ok)


def test_int4_matvec():
    in_dim, out_dim, gs = 128, 32, 128
    x = [random.gauss(0, 1) for _ in range(in_dim)]
    rows = [[random.gauss(0, 0.02) for _ in range(in_dim)] for _ in range(out_dim)]

    # build packed inputs
    packed, scales, zeros = [], [], []
    for row in rows:
        p, s, z = pack_int4_row(row, in_dim, gs)
        packed += p
        scales += s
        zeros += z

    got = axim.int4_matvec(x, packed, scales, zeros, out_dim, in_dim, gs, device="cpu")
    exp = ref_int4_matvec(x, rows, in_dim, gs)
    max_err = max(abs(g - e) for g, e in zip(got, exp))
    check("int4_matvec matches reference (INT4 tol)", max_err < 1e-3,
          f"max_err={max_err:.2e}")


def test_no_cuda_in_devices():
    devs = axim.devices()
    ok = all("cuda" not in d.backend.lower() and "cuda" not in d.vendor.lower()
             for d in devs)
    check("no CUDA vendor/backend exposed", ok)


if __name__ == "__main__":
    random.seed(7)
    print("=" * 60)
    print("AXIM ↔ SYNAXIM Op Test Suite")
    print("=" * 60)
    test_rmsnorm()
    test_silu()
    test_lowrank()
    test_int4_matvec()
    test_no_cuda_in_devices()
    print("=" * 60)
    print(f"Results: {_passed} passed, {_failed} failed")
    print("=" * 60)
    sys.exit(1 if _failed else 0)
