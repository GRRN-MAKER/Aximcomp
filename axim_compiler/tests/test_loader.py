"""
AXIM .symb Loader Test
======================
Writes a synthetic .symb INT4 file in SYNAXIM's exact binary format,
loads it back through the AXIM loader, and runs it through AXIM's INT4
matvec — proving real SYNAXIM weights execute on any hardware via AXIM.

Run:
    python3 tests/test_loader.py
"""

import os
import sys
import struct
import random
import tempfile

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))          # axim_compiler/
sys.path.insert(0, os.path.join(_here, "..", ".."))    # AXIM OS/

import axim_compiler as axim
from axim_compiler.loader import load_symb_int4

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


def write_symb_int4(path, matrix_rows, in_dim, group_size=128):
    """Write a matrix to SYNAXIM's exact .symb INT4 format (symmetric)."""
    out_dim = len(matrix_rows)
    numel = out_dim * in_dim
    n_groups_per_row = in_dim // group_size
    num_groups = out_dim * n_groups_per_row

    scales = []
    packed = bytearray()
    for row in matrix_rows:
        for g in range(n_groups_per_row):
            seg = row[g * group_size:(g + 1) * group_size]
            max_abs = max(abs(v) for v in seg) or 1e-8
            scale = max_abs / 7.0
            scales.append(scale)
            # symmetric signed [-8,7] → unsigned [0,15]
            q = [max(-8, min(7, round(v / scale))) + 8 for v in seg]
            for i in range(0, group_size, 2):
                packed.append(((q[i] & 0x0F) << 4) | (q[i + 1] & 0x0F))

    with open(path, "wb") as f:
        f.write(struct.pack("III", num_groups, group_size, numel))
        f.write(struct.pack("II", out_dim, in_dim))
        f.write(struct.pack(f"{num_groups}f", *scales))
        f.write(bytes(packed))


def test_roundtrip():
    random.seed(3)
    in_dim, out_dim, gs = 128, 16, 128
    rows = [[random.gauss(0, 0.05) for _ in range(in_dim)] for _ in range(out_dim)]

    tmp = tempfile.NamedTemporaryFile(suffix=".symb", delete=False)
    tmp.close()
    try:
        write_symb_int4(tmp.name, rows, in_dim, gs)
        loaded = load_symb_int4(tmp.name)

        check("loader: shape correct", loaded["shape"] == (out_dim, in_dim))
        check("loader: group_size correct", loaded["group_size"] == gs)
        check("loader: packed length correct",
              len(loaded["packed"]) == out_dim * (in_dim // 2))
        check("loader: scales length correct",
              len(loaded["scales"]) == out_dim * (in_dim // gs))

        # Run through AXIM on CPU and (if available) GPU
        x = [random.gauss(0, 1) for _ in range(in_dim)]
        r_cpu = axim.int4_matvec(
            x, loaded["packed"], loaded["scales"], loaded["zeros"],
            out_dim, in_dim, gs, device="cpu",
        )
        check("loader→AXIM: CPU matvec finite",
              all(abs(v) < 1e6 for v in r_cpu) and len(r_cpu) == out_dim)

        # reference dequant matvec
        ref = []
        for row in rows:
            acc = 0.0
            for j in range(in_dim):
                # replicate symmetric quant/dequant
                seg_g = j // gs
                seg = row[seg_g * gs:(seg_g + 1) * gs]
                max_abs = max(abs(v) for v in seg) or 1e-8
                scale = max_abs / 7.0
                q = max(-8, min(7, round(row[j] / scale)))
                acc += (q * scale) * x[j]
            ref.append(acc)
        max_err = max(abs(a - b) for a, b in zip(r_cpu, ref))
        check("loader→AXIM: matches dequant reference", max_err < 1e-3,
              f"max_err={max_err:.2e}")

        r_gpu = axim.int4_matvec(
            x, loaded["packed"], loaded["scales"], loaded["zeros"],
            out_dim, in_dim, gs, device="gpu",
        )
        gpu_diff = max(abs(a - b) for a, b in zip(r_cpu, r_gpu))
        check("loader→AXIM: GPU == CPU", gpu_diff < 1e-4,
              f"diff={gpu_diff:.2e}")
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    print("=" * 60)
    print("AXIM .symb Loader Test")
    print("=" * 60)
    test_roundtrip()
    print("=" * 60)
    print(f"Results: {_passed} passed, {_failed} failed")
    print("=" * 60)
    sys.exit(1 if _failed else 0)
