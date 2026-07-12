"""
AXIM — Universal CUDA-Free Compute Runtime
==========================================
AXIM lets any SYNAXIM AI model run on any hardware — Nvidia, AMD, Intel,
or Apple silicon — with zero CUDA.

  - GPU: Vulkan Compute (SPIR-V) / Metal (MSL)   [vendor-neutral]
  - CPU: AVX-512 / AVX2 / NEON SIMD               [vendor-neutral]

Quick start:

    import axim

    @axim.kernel
    def add(a, b):
        return a + b

    print(axim.run(add, [1, 2, 3], [4, 5, 6], device="auto"))
    # → [5.0, 7.0, 9.0]  — same result on any device, no CUDA

    axim.hello()   # prints the active CPU/GPU backends
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from frontend import kernel, run, devices, Device, __version__  # noqa: E402

# SYNAXIM-native ops — run any SYNAXIM model on any hardware, no CUDA.
from frontend.synaxim_ops import (  # noqa: E402
    int4_matvec,
    rmsnorm,
    silu,
    swiglu,
    lowrank_retrieve,
)


def hello() -> str:
    """Print and return the AXIM device banner (CPU + GPU backends)."""
    from orchestrator.dispatch import hello as _hello
    banner = _hello()
    print(banner)
    return banner


__all__ = [
    "kernel", "run", "devices", "Device", "hello", "__version__",
    # SYNAXIM ops
    "int4_matvec", "rmsnorm", "silu", "swiglu", "lowrank_retrieve",
]
