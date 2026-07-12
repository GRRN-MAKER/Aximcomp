"""
AXIM IR — Heterogeneous Intermediate Representation
====================================================
The device-agnostic instruction set that AXIM compiles kernels into.

A single AXIM IR program is lowered to:
  - CPU: AVX-512 / AVX2 / NEON SIMD (via C++ backend)
  - GPU: Vulkan Compute (SPIR-V) / Metal (via C++ backend)

The IR is intentionally aligned with SYNAXIM's core operations so that
any SYNAXIM model runs on any hardware through AXIM — Nvidia, AMD, Intel,
or Apple silicon — with zero CUDA.

Design goals:
  - Vendor-neutral: no CUDA, no ROCm, no proprietary intrinsics in the IR
  - Portable: the same IR emits SPIR-V for Vulkan and MSL for Metal
  - SYNAXIM-native: first-class ops for INT4 matvec, low-rank M, RMSNorm, SwiGLU
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple


# ══════════════════════════════════════════════════════════════
# Data Types
# ══════════════════════════════════════════════════════════════

class DType(Enum):
    """Element data types supported across all AXIM backends."""
    F32 = "f32"
    F16 = "f16"
    I32 = "i32"
    I8 = "i8"
    U8 = "u8"        # packed INT4 pairs live in U8 buffers
    I4 = "i4"        # logical INT4 (physically packed 2-per-byte)

    @property
    def byte_size(self) -> float:
        return {
            DType.F32: 4.0, DType.F16: 2.0, DType.I32: 4.0,
            DType.I8: 1.0, DType.U8: 1.0, DType.I4: 0.5,
        }[self]


class MemSpace(Enum):
    """Where a buffer physically lives."""
    HOST = "host"        # CPU RAM
    DEVICE = "device"    # GPU VRAM (Vulkan/Metal buffer)
    SHARED = "shared"    # unified memory (Apple silicon, integrated GPU)


# ══════════════════════════════════════════════════════════════
# Tensor / Buffer Descriptors
# ══════════════════════════════════════════════════════════════

@dataclass
class Buffer:
    """A device-agnostic memory buffer descriptor."""
    name: str
    dtype: DType
    shape: Tuple[int, ...]
    space: MemSpace = MemSpace.HOST
    is_input: bool = False
    is_output: bool = False

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return int(self.numel * self.dtype.byte_size)


# ══════════════════════════════════════════════════════════════
# IR Operations
# ══════════════════════════════════════════════════════════════

class OpCode(Enum):
    """
    The AXIM instruction set. Each op has a CPU and GPU lowering.

    Generic ops (map to any elementwise/reduction kernel):
    """
    # --- Elementwise ---
    ADD = "add"
    MUL = "mul"
    SUB = "sub"
    SILU = "silu"
    SIGMOID = "sigmoid"

    # --- Linear algebra ---
    MATVEC = "matvec"              # y = W @ x
    INT4_MATVEC = "int4_matvec"    # SYNAXIM Boolean-bit fused INT4 matvec

    # --- Normalization ---
    RMSNORM = "rmsnorm"

    # --- SYNAXIM low-rank memory ---
    LOWRANK_UPDATE = "lowrank_update"     # circular-buffer rank-1 write into U,V
    LOWRANK_RETRIEVE = "lowrank_retrieve" # out = (q @ U) @ V^T

    # --- MLP fusion ---
    SWIGLU = "swiglu"              # silu(x@Wg) * (x@Wu)

    # --- Data movement ---
    COPY = "copy"
    HELLO = "hello"                # scaffold sanity kernel


@dataclass
class Op:
    """A single IR instruction."""
    opcode: OpCode
    inputs: List[str]                    # buffer names read
    outputs: List[str]                   # buffer names written
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        ins = ", ".join(self.inputs)
        outs = ", ".join(self.outputs)
        a = f" {self.attrs}" if self.attrs else ""
        return f"{outs} = {self.opcode.value}({ins}){a}"


# ══════════════════════════════════════════════════════════════
# Kernel — a compilable unit
# ══════════════════════════════════════════════════════════════

@dataclass
class Kernel:
    """
    A named, compilable AXIM kernel: buffers + ordered ops.
    The compiler lowers this to CPU SIMD and/or GPU compute shaders.
    """
    name: str
    buffers: Dict[str, Buffer] = field(default_factory=dict)
    ops: List[Op] = field(default_factory=list)
    # Launch grid — how many parallel work items (rows/elements)
    grid: Tuple[int, ...] = (1,)

    def add_buffer(self, buf: Buffer) -> str:
        self.buffers[buf.name] = buf
        return buf.name

    def emit(self, op: Op) -> None:
        self.ops.append(op)

    def inputs(self) -> List[Buffer]:
        return [b for b in self.buffers.values() if b.is_input]

    def outputs(self) -> List[Buffer]:
        return [b for b in self.buffers.values() if b.is_output]

    def to_text(self) -> str:
        """Human-readable IR dump."""
        lines = [f"kernel @{self.name} grid={self.grid} {{"]
        for b in self.buffers.values():
            tags = []
            if b.is_input:
                tags.append("in")
            if b.is_output:
                tags.append("out")
            tag = f" [{','.join(tags)}]" if tags else ""
            lines.append(
                f"  buffer {b.name}: {b.dtype.value}{list(b.shape)} "
                f"@{b.space.value}{tag}"
            )
        lines.append("  ---")
        for op in self.ops:
            lines.append(f"  {op}")
        lines.append("}")
        return "\n".join(lines)


@dataclass
class Module:
    """A collection of kernels compiled together (e.g. a full SYNAXIM layer)."""
    name: str
    kernels: Dict[str, Kernel] = field(default_factory=dict)

    def add_kernel(self, k: Kernel) -> Kernel:
        self.kernels[k.name] = k
        return k

    def to_text(self) -> str:
        parts = [f"module @{self.name}", ""]
        for k in self.kernels.values():
            parts.append(k.to_text())
            parts.append("")
        return "\n".join(parts)
