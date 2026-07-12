"""
AXIM Pipeline Tests
===================
Verifies the compiler scaffold end-to-end: IR construction, tracing,
device dispatch, and native/Python execution parity.

Run:
    python3 tests/test_pipeline.py
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))          # axim_compiler/
sys.path.insert(0, os.path.join(_here, "..", ".."))    # AXIM OS/

import axim_compiler as axim
from axim_compiler.ir.axim_ir import (
    Kernel, Buffer, Op, OpCode, DType, MemSpace, Module
)

_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


# ── IR construction ──
def test_ir():
    k = Kernel("matvec_demo", grid=(4096,))
    k.add_buffer(Buffer("x", DType.F32, (4096,), is_input=True))
    k.add_buffer(Buffer("W", DType.I4, (4096, 4096), MemSpace.DEVICE, is_input=True))
    k.add_buffer(Buffer("y", DType.F32, (4096,), is_output=True))
    k.emit(Op(OpCode.INT4_MATVEC, ["x", "W"], ["y"], {"group_size": 128}))

    check("IR: buffer count", len(k.buffers) == 3)
    check("IR: op count", len(k.ops) == 1)
    check("IR: INT4 buffer nbytes (0.5 B/elem)",
          k.buffers["W"].nbytes == 4096 * 4096 // 2)
    check("IR: text dump non-empty", len(k.to_text()) > 0)

    m = Module("synaxim_layer")
    m.add_kernel(k)
    check("Module: kernel registered", "matvec_demo" in m.kernels)


# ── Device discovery ──
def test_devices():
    devs = axim.devices()
    check("Devices: at least CPU present",
          any(d.kind == "cpu" for d in devs))
    check("Devices: no CUDA vendor leaked",
          all("cuda" not in d.backend.lower() for d in devs))


# ── Execution parity ──
def test_execution():
    @axim.kernel
    def add(a, b):
        return a + b

    @axim.kernel
    def sub(a, b):
        return a - b

    @axim.kernel
    def fma(a, b, c):
        return a * b + c

    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    y = [10.0] * 9

    r_add = axim.run(add, x, y, device="cpu")
    check("Exec: add correct", r_add == [v + 10.0 for v in x])

    r_sub = axim.run(sub, x, y, device="cpu")
    check("Exec: sub correct", r_sub == [v - 10.0 for v in x])

    z = [100.0] * 9
    r_fma = axim.run(fma, x, y, z, device="cpu")
    check("Exec: fma correct", r_fma == [v * 10.0 + 100.0 for v in x])

    # auto device must match cpu
    r_auto = axim.run(add, x, y, device="auto")
    check("Exec: auto == cpu", r_auto == r_add)


# ── SIMD size stress (non-multiple-of-lane-width) ──
def test_tail_handling():
    @axim.kernel
    def add(a, b):
        return a + b

    # 13 elements — forces scalar-tail path after SIMD lanes
    x = [float(i) for i in range(13)]
    y = [float(i * 2) for i in range(13)]
    r = axim.run(add, x, y, device="cpu")
    check("SIMD tail: 13-elem add correct",
          r == [x[i] + y[i] for i in range(13)])


if __name__ == "__main__":
    print("=" * 60)
    print("AXIM Pipeline Test Suite")
    print("=" * 60)
    test_ir()
    test_devices()
    test_execution()
    test_tail_handling()
    print("=" * 60)
    print(f"Results: {_passed} passed, {_failed} failed")
    print("=" * 60)
    sys.exit(1 if _failed else 0)
