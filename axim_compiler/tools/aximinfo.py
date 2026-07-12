#!/usr/bin/env python3
"""
aximinfo — AXIM Device Information Tool
=======================================
Like `rocminfo` / `nvidia-smi`, but for AXIM. Enumerates every device AXIM
can target (CPU SIMD + GPU Vulkan/Metal), the active backends, and which
native libraries are loaded. Zero CUDA.

Run:
    python3 tools/aximinfo.py
"""

import os
import sys
import platform

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))          # axim_compiler/
sys.path.insert(0, os.path.join(_here, "..", ".."))    # AXIM OS/

import axim_compiler as axim
from axim_compiler.orchestrator import dispatch as _d


def main():
    line = "=" * 62
    print(line)
    print("  aximinfo — AXIM Runtime & Device Report")
    print(line)

    # ── Host ──
    print(f"  Host OS       : {platform.system()} {platform.release()}")
    print(f"  Architecture  : {platform.machine()}")
    print(f"  AXIM version  : {axim.__version__}")
    print(line)

    # ── Devices ──
    print("  Devices AXIM can target:")
    for i, d in enumerate(axim.devices()):
        marker = "GPU" if d.kind == "gpu" else "CPU"
        print(f"    [{i}] {marker:>3}  backend={d.backend:<8} "
              f"vendor={d.vendor:<8} name={d.name}")
    print(line)

    # ── Native backends ──
    cpu_lib = _d._load_native()
    print("  Native backends:")
    if cpu_lib:
        print(f"    CPU  : libaxim_cpu  loaded  → SIMD={cpu_lib.axim_cpu_backend_name().decode()}")
    else:
        print("    CPU  : python-fallback (build backend_cpu for native SIMD)")

    metal = _d.load_metal()
    if metal:
        print(f"    GPU  : libaxim_metal loaded → device={_d.metal_device_name()}")
    else:
        if platform.system() == "Darwin":
            print("    GPU  : Metal available (build backend_gpu/libaxim_metal.dylib)")
        else:
            print("    GPU  : Vulkan path (build SPIR-V shaders + Vulkan loader)")
    print(line)

    # ── CUDA status ──
    print("  CUDA          : NOT USED (AXIM is CUDA-free by design)")
    print("  CUDA compat   : AXIM-HIP shim available (axim_compiler/hip)")
    print(line)


if __name__ == "__main__":
    main()
