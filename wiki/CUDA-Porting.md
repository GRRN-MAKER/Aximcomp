# CUDA Porting — AXIM-HIP

AXIM-HIP is a drop-in header that lets existing CUDA source compile and run
on AXIM across any GPU/CPU — with **zero CUDA runtime**.

Like AMD's HIP mirrors the CUDA API, AXIM-HIP mirrors it too — but unlike
HIP (AMD-only), AXIM lowers every call to its vendor-neutral runtime
(Vulkan/Metal on GPU, AVX-512/NEON on CPU).

## One-include porting

```cpp
#define AXIM_HIP_CUDA_ALIASES
#include "axim_hip.h"

int main() {
    float* d;
    cudaMalloc((void**)&d, N * sizeof(float));       // → aximMalloc
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice); // → aximMemcpy
    // ... launch kernels ...
    cudaFree(d);                                     // → aximFree
    cudaDeviceSynchronize();                         // → aximDeviceSynchronize
}
```

Compile:

```bash
c++ -std=c++14 -Iinclude your_cuda_code.cpp src/axim_hip.cpp -o app
./app
```

## API Mapping

| CUDA | AXIM-HIP |
|------|----------|
| `cudaGetDeviceCount` | `aximGetDeviceCount` |
| `cudaSetDevice` | `aximSetDevice` |
| `cudaMalloc` | `aximMalloc` |
| `cudaFree` | `aximFree` |
| `cudaMemcpy` | `aximMemcpy` |
| `cudaDeviceSynchronize` | `aximDeviceSynchronize` |
| `cudaGetErrorString` | `aximGetErrorString` |
| `cudaMemcpyHostToDevice` | `aximMemcpyHostToDevice` |
| `cudaSuccess` | `aximSuccess` |

## Two ways to use

1. **Explicit** — call `aximMalloc`, `aximMemcpy`, etc. directly.
2. **Aliased** — define `AXIM_HIP_CUDA_ALIASES` before the include and your
   unmodified CUDA host code builds against AXIM.

## Verified

```
AXIM-HIP: 2 device(s) visible
AXIM-HIP: device 0 = AXIM CPU (SIMD)
AXIM-HIP: roundtrip OK — CUDA source ran on AXIM, zero CUDA.
```

## Roadmap: `aximify`

A HIPIFY-style search-and-replace tool that mechanically rewrites `cuda*`
calls to `axim*` across a codebase is planned.

Next: [Graphics & Gaming →](graphics.md)
