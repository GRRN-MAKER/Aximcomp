# FAQ

### Does AXIM use CUDA?

**No.** AXIM never touches CUDA. It uses Vulkan/Metal on GPU and
AVX-512/AVX2/NEON on CPU. There is a CUDA-*compat* shim (AXIM-HIP) that lets
you *port* CUDA code onto AXIM, but no CUDA runtime is required.

### How is AXIM different from ROCm/HIP?

ROCm/HIP translates CUDA → HIP and runs on AMD GPUs only. AXIM is native
(no translation), runs on **Nvidia, AMD, Intel, and Apple**, includes a
built-in graphics path for games, and provides a one-`#include` CUDA porting
shim.

### Can I run games on AXIM?

Yes — AXIM has a built-in render pipeline (Metal live, Vulkan in progress).
Compute and rendering share the GPU, so AI/physics and rendering run on one
device with zero interop overhead.

### Do I need to build on my own PC?

No. GitHub Actions builds every backend on cloud runners (macOS for Metal,
Ubuntu for Vulkan, Rust everywhere). See
[Building in the Cloud](Building-in-the-Cloud).

### What are aximBLAS and aximDNN?

They are AXIM's tuned math libraries — the CUDA-free equivalents of cuBLAS
(linear algebra) and cuDNN (deep neural networks). Cache-blocked, SIMD
vectorized, vendor-neutral.

### What is a `.symb` file?

SYNAXIM's proprietary INT4 weight format. AXIM's loader reads it directly so
real models (e.g. Magnus/Mistral 7B) run on any hardware.

### Which GPUs work today?

Apple silicon (Metal) is live and verified on M3. Nvidia/AMD/Intel via
Vulkan: shaders are written and SPIR-V builds in CI; the live Vulkan runtime
loader is in progress.

### Is it fast?

The CPU path uses native SIMD and cache-blocked GEMM (OpenBLAS-class). The
GPU path is direct Metal/Vulkan with no translation layer. Performance
benchmarks vs cuBLAS/cuDNN are on the roadmap.
