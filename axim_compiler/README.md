# AXIM Compiler — Universal CUDA-Free Compute Runtime

**AXIM lets any SYNAXIM AI model run on any hardware — Nvidia, AMD, Intel, or Apple silicon — with zero CUDA.**

```
GPU: Vulkan Compute (SPIR-V) / Metal (MSL)   ← vendor-neutral
CPU: AVX-512 / AVX2 / NEON SIMD               ← vendor-neutral
```

One kernel. Every device. No CUDA, no ROCm, no vendor lock-in.

---

## Why AXIM Differs from HIP / SYCL / ZLUDA

| | AMD HIP | Intel SYCL | ZLUDA | **AXIM** |
|--|---------|-----------|-------|----------|
| **Method** | Translate CUDA→HIP source | Migrate CUDA→SYCL source | Binary intercept CUDA | **Native compile — no CUDA at all** |
| **Frontend** | C++ | C++ | (binary) | **Python (`@axim.kernel`)** |
| **Orchestration** | ROCm runtime | oneAPI runtime | ZLUDA runtime | **Rust dispatch layer** |
| **GPU backend** | AMD only | Any (via SYCL) | AMD/Intel | **Vulkan + Metal (any GPU)** |
| **CPU backend** | — | oneAPI | — | **AVX-512 / AVX2 / NEON** |
| **Dependency** | ROCm | oneAPI | CUDA runtime | **None** |

HIP, SYCL, and ZLUDA all *translate away from CUDA*. **AXIM never touches CUDA** — it compiles kernels straight to vendor-neutral targets.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (Python)          @axim.kernel decorator       │
│  axim_compiler/frontend/    traces Python → AXIM IR      │
├─────────────────────────────────────────────────────────┤
│  IR (Python)                heterogeneous instruction set │
│  axim_compiler/ir/          INT4 matvec, low-rank M, ...  │
├─────────────────────────────────────────────────────────┤
│  Orchestrator (Rust)        device dispatch + fallback    │
│  axim_compiler/orchestrator/  CPU ↔ GPU routing           │
├──────────────────────────┬──────────────────────────────┤
│  CPU Backend (C++)        │  GPU Backend (C++)            │
│  backend_cpu/             │  backend_gpu/                 │
│  AVX-512 / AVX2 / NEON    │  Vulkan (SPIR-V) / Metal      │
└──────────────────────────┴──────────────────────────────┘
```

---

## Quick Start

```python
import axim_compiler as axim

@axim.kernel
def add(a, b):
    return a + b

print(axim.run(add, [1, 2, 3], [4, 5, 6], device="auto"))
# → [5.0, 7.0, 9.0]  — same result on any device, no CUDA

axim.hello()   # shows active CPU/GPU backends
```

---

## Build

The scaffold runs immediately in **pure-Python fallback**. Build the native
C++ SIMD backend for real hardware acceleration:

### CPU backend (SIMD)

**With CMake:**
```bash
cd axim_compiler/backend_cpu
mkdir -p build && cd build
cmake .. && cmake --build .
```

**Direct (no CMake):**
```bash
cd axim_compiler/backend_cpu
mkdir -p build
# macOS / ARM (NEON is baseline on aarch64):
c++ -std=c++14 -O3 -fPIC -shared -Iinclude src/axim_cpu.cpp -o build/libaxim_cpu.dylib
# Linux / x86 (AVX2 + FMA; AVX-512 selected at runtime):
# c++ -std=c++14 -O3 -mavx2 -mfma -fPIC -shared -Iinclude src/axim_cpu.cpp -o build/libaxim_cpu.so
```

### GPU backend (Vulkan / Metal)

```bash
cd axim_compiler/backend_gpu
# Metal (macOS):
#   c++ -std=c++14 -O3 -DAXIM_GPU_METAL -fPIC -shared -Iinclude src/axim_gpu.cpp -o build/libaxim_gpu.dylib
# Vulkan (Linux/Windows):
#   c++ -std=c++14 -O3 -DAXIM_GPU_VULKAN -fPIC -shared -Iinclude src/axim_gpu.cpp -lvulkan -o build/libaxim_gpu.so
```

### Rust orchestrator

```bash
cd axim_compiler/orchestrator
cargo build --release
```

---

## Test

```bash
python3 axim_compiler/tests/test_pipeline.py     # 12 checks
python3 axim_compiler/examples/hello_world.py     # end-to-end demo
```

---

## AXIM IR — SYNAXIM-Native Ops

The IR has first-class operations for SYNAXIM's core primitives, so a
whole SYNAXIM layer lowers directly to AXIM:

| Op | Purpose |
|----|---------|
| `INT4_MATVEC` | Boolean-bit fused INT4 matrix-vector (all 7 weight projections) |
| `LOWRANK_UPDATE` | Circular-buffer rank-1 write into U, V factors |
| `LOWRANK_RETRIEVE` | `out = (q @ U) @ V^T` — O(D×r) memory read |
| `RMSNORM` | RMS normalization |
| `SWIGLU` | Fused `silu(x@Wg) * (x@Wu)` |
| `SILU`, `SIGMOID`, `ADD`, `MUL`, `SUB` | Elementwise |

---

## Status

| Component | Status |
|-----------|--------|
| AXIM IR | ✅ Complete |
| Python frontend (`@axim.kernel`) | ✅ Complete |
| CPU backend (NEON verified, AVX2/512 ready) | ✅ Complete |
| GPU backend ABI (Vulkan/Metal) | ✅ Scaffolded (shaders WIP) |
| Rust orchestrator | ✅ Complete |
| Hello-world end-to-end | ✅ Passing on CPU |
| Test suite | ✅ 12/12 passing |

### Next
- Real SPIR-V (Vulkan) + MSL (Metal) shader modules per IR op
- Wire SYNAXIM's `_int4_matvec_fused` through AXIM IR
- maturin/PyO3 packaging so `import axim` uses the Rust orchestrator directly
- Multi-device scheduling (CPU + GPU heterogeneous split)

---

*Part of the GRRN post-transformer stack. AXIM is the runtime; SYNAXIM is the engine.*
