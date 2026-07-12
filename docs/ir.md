# AXIM IR

The AXIM Intermediate Representation is a device-agnostic instruction set.
The same IR lowers to CPU SIMD and GPU compute shaders, and is intentionally
aligned with SYNAXIM's core operations.

## Data Types

| DType | Bytes | Use |
|-------|-------|-----|
| `F32` | 4 | activations, scales |
| `F16` | 2 | embeddings, norms |
| `I8` / `U8` | 1 | packed INT4 lives in U8 |
| `I4` | 0.5 | logical INT4 (2-per-byte) |

## Ops

| OpCode | Meaning |
|--------|---------|
| `ADD`, `MUL`, `SUB` | elementwise |
| `SILU`, `SIGMOID` | activations |
| `MATVEC` | `y = W @ x` |
| `INT4_MATVEC` | Boolean-bit fused INT4 projection |
| `RMSNORM` | RMS normalization |
| `LOWRANK_UPDATE` | circular-buffer rank-1 write into U, V |
| `LOWRANK_RETRIEVE` | `out = (q @ U) @ V^T` |
| `SWIGLU` | `silu(x@Wg) * (x@Wu)` |

## Example

```python
from axim_compiler.ir.axim_ir import Kernel, Buffer, Op, OpCode, DType, MemSpace

k = Kernel("matvec", grid=(4096,))
k.add_buffer(Buffer("x", DType.F32, (4096,), is_input=True))
k.add_buffer(Buffer("W", DType.I4, (4096, 4096), MemSpace.DEVICE, is_input=True))
k.add_buffer(Buffer("y", DType.F32, (4096,), is_output=True))
k.emit(Op(OpCode.INT4_MATVEC, ["x", "W"], ["y"], {"group_size": 128}))
print(k.to_text())
```

The IR carries buffer metadata (dtype, shape, memory space, in/out) so both
backends know how to allocate and dispatch.
