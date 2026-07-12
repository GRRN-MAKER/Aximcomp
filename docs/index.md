# AXIM — Universal CUDA-Free Compute + Graphics Runtime

**AXIM runs any SYNAXIM AI model, and any GPU workload, on any hardware — Nvidia, AMD, Intel, or Apple silicon — with zero CUDA.**

```
GPU compute + graphics : Vulkan (SPIR-V) / Metal (MSL)
CPU compute            : AVX-512 / AVX2 / NEON SIMD
Tuned math             : aximBLAS (cuBLAS-class) + aximDNN (cuDNN-class)
CUDA compatibility     : AXIM-HIP shim (port CUDA in 1 #include)
```

One kernel. Every device. For AI inference, HPC, and **games**.

---

## Why AXIM Exists

Nvidia's dominance rests on CUDA and its tuned libraries: **cuBLAS** (linear
algebra) and **cuDNN** (deep neural networks), which squeeze maximum
performance from Nvidia GPUs. That lock-in means AMD, Intel, and Apple GPUs
are second-class citizens for compute.

AXIM breaks the lock-in by providing:

1. **A vendor-neutral runtime** — Vulkan/Metal on GPU, AVX-512/NEON on CPU.
2. **Tuned libraries** — `aximBLAS` and `aximDNN` mirror cuBLAS/cuDNN so the
   same code hits peak throughput on *any* device.
3. **A CUDA-compat shim** — existing CUDA source runs on AXIM with one
   `#include`, no rewrite.
4. **A built-in graphics path** — compute and rendering share the GPU, so
   games get fast AI/physics *and* rendering without CUDA-graphics interop.

## Comparison

| | ROCm/HIP | SYCL | ZLUDA | **AXIM** |
|--|----------|------|-------|----------|
| Method | Translate CUDA→HIP | Migrate CUDA→SYCL | Binary intercept | **Native, no CUDA** |
| GPU targets | AMD only | Any | AMD/Intel | **Nvidia + AMD + Intel + Apple** |
| CPU compute | — | oneAPI | — | **AVX-512 / AVX2 / NEON** |
| Tuned math | rocBLAS/MIOpen | oneMKL/oneDNN | — | **aximBLAS / aximDNN** |
| Graphics/gaming | extra libs | — | — | **Built-in** |
| CUDA porting | HIPIFY | SYCLomatic | (none) | **AXIM-HIP `#include`** |
| Language | C++ | C++ | binary | **Python + Rust + C++** |

## Verified (Apple M3)

- ✅ CPU: NEON SIMD (native)
- ✅ GPU: Metal compute **and** graphics (Apple M3)
- ✅ INT4 matvec CPU == GPU, exact match
- ✅ CUDA source runs via AXIM-HIP shim
- ✅ Game frame rendered on GPU
- ✅ aximBLAS + aximDNN tuned libs

Continue to [Getting Started →](getting-started.md)
