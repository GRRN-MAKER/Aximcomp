"""
AXIM Frontend API
=================
Public developer interface: the @axim.kernel decorator, device discovery,
and the run() dispatcher.

This module traces decorated Python functions into AXIM IR and routes
execution to the appropriate backend (CPU SIMD or GPU Vulkan/Metal).
No CUDA anywhere in the path.
"""

from __future__ import annotations

import os
import platform
import functools
from dataclasses import dataclass
from typing import Callable, Any, List, Optional

import sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))

from ir.axim_ir import Kernel, Op, OpCode, Buffer, DType, MemSpace  # noqa: E402

__version__ = "0.1.0"


# ══════════════════════════════════════════════════════════════
# Device Discovery — vendor-neutral
# ══════════════════════════════════════════════════════════════

@dataclass
class Device:
    """A compute device AXIM can target."""
    kind: str          # "cpu" | "gpu"
    backend: str       # "simd" | "vulkan" | "metal"
    name: str
    vendor: str        # "apple" | "nvidia" | "amd" | "intel" | "generic"

    def __repr__(self) -> str:
        return f"<Device {self.kind}:{self.backend} {self.name} ({self.vendor})>"


def _detect_cpu_backend() -> str:
    """Pick the best SIMD ISA for this CPU — never CUDA."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "neon"        # Apple silicon, ARM servers
    # x86_64 — prefer AVX-512, fall back to AVX2
    # (runtime CPUID check happens in the C++ backend; default to avx2)
    return "avx2"


def _detect_gpu_backend() -> Optional[tuple]:
    """
    Detect an available vendor-neutral GPU backend.
    Returns (backend, vendor, name) or None.
    Metal on Apple, Vulkan everywhere else. No CUDA.
    """
    system = platform.system()
    if system == "Darwin":
        return ("metal", "apple", "Apple GPU (Metal)")
    # On Linux/Windows, Vulkan is the universal path for Nvidia/AMD/Intel
    # (actual enumeration done by the C++/Rust layer at runtime)
    return ("vulkan", "generic", "Vulkan GPU")


def devices() -> List[Device]:
    """Enumerate all AXIM-targetable devices on this machine."""
    devs: List[Device] = [
        Device("cpu", _detect_cpu_backend(), platform.processor() or platform.machine(),
               "apple" if platform.machine().lower() in ("arm64", "aarch64") else "generic")
    ]
    gpu = _detect_gpu_backend()
    if gpu:
        backend, vendor, name = gpu
        devs.append(Device("gpu", backend, name, vendor))
    return devs


def _select_device(pref: str) -> Device:
    """Resolve a device preference string to a concrete Device."""
    devs = devices()
    if pref == "cpu":
        return next(d for d in devs if d.kind == "cpu")
    if pref == "gpu":
        gpu = next((d for d in devs if d.kind == "gpu"), None)
        if gpu is None:
            raise RuntimeError("No AXIM GPU backend available; use device='cpu'")
        return gpu
    # "auto" — prefer GPU if present, else CPU
    return next((d for d in devs if d.kind == "gpu"), devs[0])


# ══════════════════════════════════════════════════════════════
# @axim.kernel — trace a Python function into AXIM IR
# ══════════════════════════════════════════════════════════════

# A minimal tracer: the decorated function is executed with symbolic
# operands that record ops into a Kernel. This keeps the frontend pure
# Python while producing real IR the backends can lower.

class _Sym:
    """Symbolic operand that records IR ops instead of computing values."""
    _counter = 0

    def __init__(self, kernel: Kernel, name: str):
        self.kernel = kernel
        self.name = name

    @classmethod
    def _fresh(cls, kernel: Kernel, prefix: str = "t") -> "_Sym":
        cls._counter += 1
        name = f"{prefix}{cls._counter}"
        kernel.add_buffer(Buffer(name, DType.F32, (0,)))  # shape filled at run
        return cls(kernel, name)

    def _binop(self, other: "_Sym", opcode: OpCode) -> "_Sym":
        out = _Sym._fresh(self.kernel)
        rhs = other.name if isinstance(other, _Sym) else str(other)
        self.kernel.emit(Op(opcode, [self.name, rhs], [out.name]))
        return out

    def __add__(self, other):
        return self._binop(other, OpCode.ADD)

    def __mul__(self, other):
        return self._binop(other, OpCode.MUL)

    def __sub__(self, other):
        return self._binop(other, OpCode.SUB)


def kernel(fn: Callable) -> Callable:
    """
    Decorator: mark a Python function as an AXIM kernel.

    The function is traced once into AXIM IR on first use. The resulting
    Kernel is cached on the wrapper as `.axim_kernel`.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Direct-call convenience: trace + run on auto device.
        return run(wrapper, *args, **kwargs)

    def _trace(n_args: int) -> Kernel:
        k = Kernel(name=fn.__name__)
        syms = []
        for i in range(n_args):
            bname = f"arg{i}"
            k.add_buffer(Buffer(bname, DType.F32, (0,), is_input=True))
            syms.append(_Sym(k, bname))
        result = fn(*syms)
        if isinstance(result, _Sym):
            # mark the traced result buffer as output
            if result.name in k.buffers:
                k.buffers[result.name].is_output = True
            else:
                k.add_buffer(Buffer(result.name, DType.F32, (0,), is_output=True))
        return k

    wrapper._axim_trace = _trace          # type: ignore[attr-defined]
    wrapper._axim_fn = fn                 # type: ignore[attr-defined]
    return wrapper


# ══════════════════════════════════════════════════════════════
# run() — dispatch a kernel to a device
# ══════════════════════════════════════════════════════════════

def run(kfn: Callable, *args, device: str = "auto", **kwargs):
    """
    Execute an @axim.kernel on the chosen device.

    Args:
        kfn: an @axim.kernel-decorated function
        *args: input arrays (Python lists or numpy arrays)
        device: "auto" | "cpu" | "gpu"

    Returns:
        The kernel result as a Python list (host memory).
    """
    if not hasattr(kfn, "_axim_trace"):
        raise TypeError("run() requires an @axim.kernel function")

    dev = _select_device(device)
    k = kfn._axim_trace(len(args))  # type: ignore[attr-defined]

    # Import the orchestrator lazily (it bridges to Rust/C++).
    from orchestrator.dispatch import execute
    return execute(k, list(args), dev)
