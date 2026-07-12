# Architecture

AXIM is a layered compiler/runtime. Each layer is vendor-neutral and
CUDA-free.

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (Python)          @axim.kernel + SYNAXIM ops   │
│  axim_compiler/frontend/    traces Python → AXIM IR      │
├─────────────────────────────────────────────────────────┤
│  IR (Python)                heterogeneous instruction set │
│  axim_compiler/ir/          INT4 matvec, low-rank M, ...  │
├─────────────────────────────────────────────────────────┤
│  Orchestrator (Rust)        device dispatch + fallback    │
│  axim_compiler/orchestrator/  CPU ↔ GPU routing           │
├──────────────────────────┬──────────────────────────────┤
│  CPU Backend (C++)        │  GPU Backend (C++/ObjC++)     │
│  backend_cpu/             │  backend_gpu/                 │
│  AVX-512 / AVX2 / NEON    │  Vulkan (SPIR-V) / Metal      │
│  + aximBLAS + aximDNN     │  + shaders/                   │
├──────────────────────────┴──────────────────────────────┤
│  Graphics (C++/ObjC++)     game render pipeline           │
│  AXIM-HIP (C++)            CUDA-compat porting shim       │
│  loader/                   .symb model reader             │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

A kernel call travels top to bottom:

1. **Frontend** — `@axim.kernel` traces Python into AXIM IR, or a SYNAXIM op
   (`int4_matvec`, `rmsnorm`, ...) builds IR directly with typed buffers.
2. **IR** — device-agnostic ops with SYNAXIM-native primitives.
3. **Orchestrator** — resolves `device="auto"` to CPU or GPU and dispatches.
4. **Backend** — executes on native SIMD (CPU) or compute shaders (GPU).

If a native library is missing, the pure-Python interpreter runs the IR so
the pipeline always works.

## Why Python + Rust + C++

| Layer | Language | Reason |
|-------|----------|--------|
| Frontend | Python | Ergonomic kernel authoring, ML ecosystem |
| Orchestrator | Rust | Memory-safe device dispatch, PyO3 module |
| Backends | C++ / ObjC++ | Direct SIMD intrinsics, Metal/Vulkan APIs |

## Design Principles

- **Vendor-neutral first** — never a CUDA/ROCm-specific intrinsic in the IR.
- **SYNAXIM-native** — first-class ops for INT4 matvec and low-rank memory.
- **Graceful fallback** — GPU → CPU → pure-Python, always runnable.
- **Shared device for games** — compute and graphics use one GPU queue.

Next: [AXIM IR →](ir.md)
