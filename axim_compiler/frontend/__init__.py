"""
AXIM Frontend — the developer-facing API.

    import axim

    @axim.kernel
    def add(a, b):
        return a + b

    axim.run(add, x, y, device="auto")   # runs on CPU or GPU, no CUDA

The frontend traces a Python function into AXIM IR, hands it to the
orchestrator, which dispatches to the CPU (SIMD) or GPU (Vulkan/Metal)
backend. The same kernel runs on Nvidia, AMD, Intel, and Apple silicon.
"""

from .api import kernel, run, devices, Device, __version__

__all__ = ["kernel", "run", "devices", "Device", "__version__"]
