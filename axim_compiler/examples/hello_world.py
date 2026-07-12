"""
AXIM Hello World
================
Proves the full pipeline runs CUDA-free on CPU (SIMD if the native
backend is built, pure-Python otherwise), on any machine.

Run:
    python3 examples/hello_world.py
"""

import os
import sys

# Make `import axim_compiler as axim` work when run from the repo root
# or from the examples/ folder.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))          # axim_compiler/
sys.path.insert(0, os.path.join(_here, "..", ".."))    # AXIM OS/

import axim_compiler as axim  # noqa: E402


# ── 1. Show the active devices / backends ──
print("=" * 60)
axim.hello()
print("=" * 60)
print("Devices AXIM can target:")
for d in axim.devices():
    print(f"  {d}")
print("=" * 60)


# ── 2. Define a kernel — the SAME code runs on CPU or GPU ──
@axim.kernel
def add(a, b):
    return a + b


@axim.kernel
def fma(a, b, c):
    # (a * b) + c  — fused multiply-add, traced into AXIM IR
    return a * b + c


# ── 3. Run on CPU (explicit) ──
x = [1.0, 2.0, 3.0, 4.0]
y = [10.0, 20.0, 30.0, 40.0]

r_add = axim.run(add, x, y, device="cpu")
print(f"add(x, y) on CPU     = {r_add}")
assert r_add == [11.0, 22.0, 33.0, 44.0], "add failed"

# ── 4. Run on auto device (GPU if present, else CPU) ──
r_auto = axim.run(add, x, y, device="auto")
print(f"add(x, y) on auto    = {r_auto}")

# ── 5. A fused kernel ──
z = [100.0, 100.0, 100.0, 100.0]
r_fma = axim.run(fma, x, y, z, device="cpu")
print(f"fma(x, y, z) on CPU  = {r_fma}")
assert r_fma == [110.0, 140.0, 190.0, 260.0], "fma failed"

print("=" * 60)
print("✅ AXIM hello-world passed — CUDA-free execution verified.")
print("=" * 60)
