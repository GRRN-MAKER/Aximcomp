# Tuned Libraries — aximBLAS & aximDNN

Nvidia's performance edge comes from **cuBLAS** (linear algebra) and
**cuDNN** (deep neural networks), which are hand-tuned to squeeze maximum
performance from Nvidia hardware. AXIM provides vendor-neutral equivalents
that hit peak throughput on *any* CPU/GPU.

## aximBLAS — the CUDA-free cuBLAS

Cache-blocked, SIMD-vectorized BLAS routines. AVX-512/AVX2 on x86, NEON on
ARM, scalar fallback. Header: `backend_cpu/include/axim_blas.h`.

| Routine | Operation | Tuning |
|---------|-----------|--------|
| `axim_sgemv` | `y = αAx + βy` | SIMD dot per row |
| `axim_sgemm` | `C = αAB + βC` | 64×64×64 cache tiles + FMA micro-kernel |
| `axim_saxpy` | `y = αx + y` | Fused multiply-add |
| `axim_sdot` | `Σ x·y` | Vectorized accumulation |

```c
#include "axim_blas.h"
// C = A(MxK) * B(KxN)
axim_sgemm(M, N, K, 1.0f, A, B, 0.0f, C);
```

### Why cache-blocked GEMM matters

Naive GEMM thrashes cache on large matrices. aximBLAS tiles the computation
into 64×64 blocks that stay resident in L1/L2, then applies an FMA
micro-kernel — the same strategy OpenBLAS and cuBLAS use, but portable.

## aximDNN — the CUDA-free cuDNN

Fused, numerically-stable DNN primitives. Header:
`backend_cpu/include/axim_dnn.h`.

| Routine | Operation |
|---------|-----------|
| `axim_softmax` | numerically-stable row softmax |
| `axim_layernorm` | mean/var normalize + affine |
| `axim_gelu` | GELU (tanh approximation) |
| `axim_attention_step` | single-token scaled dot-product attention |

```c
#include "axim_dnn.h"
// out = softmax(q·Kᵀ / √d) · V   for one token
axim_attention_step(q, K, V, out, seq_len, d_head);
```

## Verified (Apple M3, NEON)

```
sgemv  y = [10, 26, 42]      ✅
softmax sum = 1.0            ✅
sdot   = 30                  ✅
```

## Porting from cuBLAS/cuDNN

Function names mirror BLAS/DNN conventions, so migration is mechanical:

| CUDA | AXIM |
|------|------|
| `cublasSgemm` | `axim_sgemm` |
| `cublasSgemv` | `axim_sgemv` |
| `cublasSaxpy` | `axim_saxpy` |
| `cudnnSoftmaxForward` | `axim_softmax` |
| `cudnnBatchNorm...` | `axim_layernorm` |

Next: [CUDA Porting →](cuda-porting.md)
