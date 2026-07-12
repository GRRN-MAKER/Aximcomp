"""
AXIM Orchestrator — Python Dispatch Bridge
===========================================
Executes AXIM IR by calling the compiled vendor-neutral C++ backends via
ctypes. Chooses CPU (SIMD) or GPU (Vulkan/Metal) and falls back to a pure
Python interpreter if the native library hasn't been built yet — so the
scaffold always runs, on any machine, with zero CUDA.

Backend library search order:
  1. libaxim_cpu.dylib / .so / .dll   (built by backend_cpu/CMakeLists.txt)
  2. pure-Python IR interpreter fallback
"""

from __future__ import annotations

import os
import sys
import ctypes
import platform
from typing import List, Any

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))

from ir.axim_ir import Kernel, Op, OpCode  # noqa: E402


# ══════════════════════════════════════════════════════════════
# Native backend loading (ctypes)
# ══════════════════════════════════════════════════════════════

def _lib_names() -> List[str]:
    system = platform.system()
    if system == "Darwin":
        return ["libaxim_cpu.dylib"]
    if system == "Windows":
        return ["axim_cpu.dll", "libaxim_cpu.dll"]
    return ["libaxim_cpu.so"]


def _search_dirs() -> List[str]:
    return [
        os.path.join(_here, "..", "backend_cpu", "build"),
        os.path.join(_here, "..", "backend_cpu"),
        os.path.join(_here, "..", "build"),
        _here,
    ]


_NATIVE = None
_METAL = None


def _load_native():
    global _NATIVE
    if _NATIVE is not None:
        return _NATIVE
    for d in _search_dirs():
        for name in _lib_names():
            path = os.path.abspath(os.path.join(d, name))
            if os.path.exists(path):
                try:
                    lib = ctypes.CDLL(path)
                    _bind(lib)
                    _NATIVE = lib
                    return lib
                except OSError:
                    continue
    _NATIVE = False  # mark as "tried, not found"
    return False


def _metal_lib_dirs():
    return [
        os.path.join(_here, "..", "backend_gpu", "build"),
        os.path.join(_here, "..", "backend_gpu"),
    ]


def load_metal():
    """Load the live Metal GPU backend (Apple silicon). Returns lib or False."""
    global _METAL
    if _METAL is not None:
        return _METAL
    if platform.system() != "Darwin":
        _METAL = False
        return False
    for d in _metal_lib_dirs():
        path = os.path.abspath(os.path.join(d, "libaxim_metal.dylib"))
        if os.path.exists(path):
            try:
                lib = ctypes.CDLL(path)
                lib.axim_metal_init.restype = ctypes.c_int
                lib.axim_metal_device_name.restype = ctypes.c_char_p
                f32 = ctypes.POINTER(ctypes.c_float)
                u8 = ctypes.POINTER(ctypes.c_uint8)
                lib.axim_metal_add.argtypes = [f32, f32, f32, ctypes.c_size_t]
                lib.axim_metal_mul.argtypes = [f32, f32, f32, ctypes.c_size_t]
                lib.axim_metal_int4_matvec.argtypes = [
                    f32, u8, f32, f32, f32,
                    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                ]
                if lib.axim_metal_init() == 0:
                    _METAL = lib
                    return lib
            except OSError:
                continue
    _METAL = False
    return False


def metal_device_name():
    lib = load_metal()
    if lib:
        return lib.axim_metal_device_name().decode()
    return None


def _bind(lib):
    """Declare C signatures for the CPU backend functions we use."""
    f32p = ctypes.POINTER(ctypes.c_float)
    u8p = ctypes.POINTER(ctypes.c_uint8)

    lib.axim_cpu_backend_name.restype = ctypes.c_char_p

    lib.axim_cpu_add.argtypes = [f32p, f32p, f32p, ctypes.c_size_t]
    lib.axim_cpu_mul.argtypes = [f32p, f32p, f32p, ctypes.c_size_t]
    lib.axim_cpu_sub.argtypes = [f32p, f32p, f32p, ctypes.c_size_t]
    lib.axim_cpu_silu.argtypes = [f32p, f32p, ctypes.c_size_t]

    lib.axim_cpu_rmsnorm.argtypes = [
        f32p, f32p, f32p, ctypes.c_size_t, ctypes.c_float
    ]
    lib.axim_cpu_int4_matvec.argtypes = [
        f32p, u8p, f32p, f32p, f32p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t
    ]
    lib.axim_cpu_lowrank_retrieve.argtypes = [
        f32p, f32p, f32p, f32p, ctypes.c_size_t, ctypes.c_size_t
    ]
    lib.axim_cpu_hello.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.axim_cpu_hello.restype = ctypes.c_int


def backend_name() -> str:
    lib = _load_native()
    if lib:
        return lib.axim_cpu_backend_name().decode()
    return "python-fallback"


# ══════════════════════════════════════════════════════════════
# ctypes helpers
# ══════════════════════════════════════════════════════════════

def _to_c_float_array(data):
    arr = (ctypes.c_float * len(data))(*[float(x) for x in data])
    return arr


# ══════════════════════════════════════════════════════════════
# Execution — native path + pure-Python fallback
# ══════════════════════════════════════════════════════════════

def _run_native(op: OpCode, a, b):
    lib = _load_native()
    n = len(a)
    ca = _to_c_float_array(a)
    cb = _to_c_float_array(b)
    out = (ctypes.c_float * n)()
    cast = lambda x: ctypes.cast(x, ctypes.POINTER(ctypes.c_float))
    if op == OpCode.ADD:
        lib.axim_cpu_add(cast(ca), cast(cb), cast(out), n)
    elif op == OpCode.MUL:
        lib.axim_cpu_mul(cast(ca), cast(cb), cast(out), n)
    elif op == OpCode.SUB:
        lib.axim_cpu_sub(cast(ca), cast(cb), cast(out), n)
    else:
        raise NotImplementedError(op)
    return list(out)


def _run_python(op: OpCode, a, b):
    if op == OpCode.ADD:
        return [x + y for x, y in zip(a, b)]
    if op == OpCode.MUL:
        return [x * y for x, y in zip(a, b)]
    if op == OpCode.SUB:
        return [x - y for x, y in zip(a, b)]
    raise NotImplementedError(op)


def execute(kernel: Kernel, args: List[Any], device) -> Any:
    """
    Execute a traced kernel's IR against the provided input arrays.

    A minimal register machine: buffer name → concrete list. Native
    backend used when available, pure Python otherwise.
    """
    env = {}
    # bind inputs
    input_names = [b.name for b in kernel.buffers.values() if b.is_input]
    for name, val in zip(input_names, args):
        env[name] = list(val)

    lib = _load_native()
    use_native = bool(lib) and device.kind == "cpu"

    last_out = None
    for op in kernel.ops:
        a = env[op.inputs[0]]
        b = env.get(op.inputs[1], None) if len(op.inputs) > 1 else None
        if use_native and op.opcode in (OpCode.ADD, OpCode.MUL, OpCode.SUB):
            res = _run_native(op.opcode, a, b)
        else:
            res = _run_python(op.opcode, a, b)
        env[op.outputs[0]] = res
        last_out = res

    # Prefer the buffer explicitly marked as output.
    for bname, buf in kernel.buffers.items():
        if buf.is_output and bname in env:
            return env[bname]
    return last_out


def hello() -> str:
    """Return the full AXIM device banner."""
    lib = _load_native()
    lines = ["AXIM Orchestrator ready."]
    if lib:
        buf = ctypes.create_string_buffer(256)
        lib.axim_cpu_hello(buf, 256)
        lines.append("  " + buf.value.decode())
    else:
        lines.append("  CPU backend: python-fallback (native lib not built)")
    lines.append(f"  Backend: {backend_name()}")
    return "\n".join(lines)
