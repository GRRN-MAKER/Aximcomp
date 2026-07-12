# AXIM Wiki

**AXIM — Universal CUDA-Free Compute + Graphics Runtime**

Run any SYNAXIM AI model, and any GPU workload, on any hardware — Nvidia,
AMD, Intel, or Apple silicon — with **zero CUDA**.

## Wiki Pages

- **[Quick Start](Quick-Start)** — install and run in minutes
- **[Architecture](Architecture)** — how the layers fit together
- **[Tuned Libraries](Tuned-Libraries)** — aximBLAS & aximDNN (cuBLAS/cuDNN class)
- **[CUDA Porting](CUDA-Porting)** — run CUDA code on AXIM with one `#include`
- **[Graphics & Gaming](Graphics-and-Gaming)** — fast AI + rendering, no interop
- **[Building in the Cloud](Building-in-the-Cloud)** — GitHub Actions does the builds
- **[FAQ](FAQ)** — common questions
- **[Roadmap](Roadmap)** — what's next

## At a Glance

```
GPU compute + graphics : Vulkan (SPIR-V) / Metal (MSL)
CPU compute            : AVX-512 / AVX2 / NEON SIMD
Tuned math             : aximBLAS + aximDNN
CUDA compatibility     : AXIM-HIP shim
```

## Verified on Apple M3

- ✅ CPU: NEON SIMD
- ✅ GPU: Metal compute **and** graphics
- ✅ INT4 matvec CPU == GPU exact match
- ✅ CUDA source runs via AXIM-HIP
- ✅ Game frame rendered on GPU
- ✅ aximBLAS + aximDNN tuned libraries

Repository: [github.com/GRRN-MAKER/Aximcomp](https://github.com/GRRN-MAKER/Aximcomp)
