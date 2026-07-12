/*
 * AXIM-HIP — CUDA-Compatible Portability Layer
 * =============================================
 * A drop-in header that lets existing CUDA source compile and run on AXIM
 * across any GPU (Nvidia, AMD, Intel, Apple) or CPU — with zero CUDA runtime.
 *
 * Like AMD's HIP, AXIM-HIP mirrors the CUDA API surface so porting is a
 * search-and-replace (or a single #include). Unlike HIP (AMD-only) or SYCL,
 * AXIM lowers every call to its vendor-neutral runtime:
 *     GPU: Vulkan / Metal      CPU: AVX-512 / NEON
 *
 * Two ways to use:
 *   1. Include this header and call aximMalloc / aximMemcpy / aximLaunch...
 *   2. Define AXIM_HIP_CUDA_ALIASES before including to get cudaMalloc etc.
 *      mapped onto AXIM automatically, so unmodified CUDA host code builds.
 *
 * Example (unmodified CUDA host code):
 *     #define AXIM_HIP_CUDA_ALIASES
 *     #include "axim_hip.h"
 *     cudaMalloc(&d, n);          // → aximMalloc  → AXIM device buffer
 *     cudaMemcpy(d, h, n, ...);   // → aximMemcpy
 *     cudaFree(d);               // → aximFree
 */
#ifndef AXIM_HIP_H
#define AXIM_HIP_H

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Error codes (CUDA-compatible values) ── */
typedef enum {
    aximSuccess = 0,
    aximErrorInvalidValue = 1,
    aximErrorMemoryAllocation = 2,
    aximErrorNoDevice = 100,
    aximErrorNotSupported = 801
} aximError_t;

/* ── Memcpy kinds (CUDA-compatible) ── */
typedef enum {
    aximMemcpyHostToHost = 0,
    aximMemcpyHostToDevice = 1,
    aximMemcpyDeviceToHost = 2,
    aximMemcpyDeviceToDevice = 3
} aximMemcpyKind;

/* Opaque device pointer. Backed by a Vulkan/Metal buffer or host memory. */
typedef void* aximDeviceptr_t;

/* ── Device management (mirrors cudaGetDeviceCount / cudaSetDevice) ── */
aximError_t aximGetDeviceCount(int* count);
aximError_t aximSetDevice(int device);
aximError_t aximGetDeviceName(char* name, int len, int device);

/* ── Memory (mirrors cudaMalloc / cudaFree / cudaMemcpy) ── */
aximError_t aximMalloc(void** ptr, size_t size);
aximError_t aximFree(void* ptr);
aximError_t aximMemcpy(void* dst, const void* src, size_t size,
                       aximMemcpyKind kind);

/* ── Kernel launch (mirrors cudaLaunchKernel semantics, simplified) ──
 * Launches a named AXIM kernel with elementwise/int4 semantics. The kernel
 * name maps to a registered AXIM IR op (add, mul, int4_matvec, ...).
 */
aximError_t aximLaunch(const char* kernel_name, size_t grid,
                       void** args, int n_args);

/* ── Synchronization (mirrors cudaDeviceSynchronize) ── */
aximError_t aximDeviceSynchronize(void);

/* ── Error strings (mirrors cudaGetErrorString) ── */
const char* aximGetErrorString(aximError_t err);

#ifdef __cplusplus
}
#endif

/* ══════════════════════════════════════════════════════════════
 * Optional CUDA aliases — makes unmodified CUDA host code build on AXIM.
 * ══════════════════════════════════════════════════════════════ */
#ifdef AXIM_HIP_CUDA_ALIASES
  typedef aximError_t     cudaError_t;
  typedef aximMemcpyKind  cudaMemcpyKind;
  #define cudaSuccess              aximSuccess
  #define cudaMemcpyHostToDevice   aximMemcpyHostToDevice
  #define cudaMemcpyDeviceToHost   aximMemcpyDeviceToHost
  #define cudaMemcpyDeviceToDevice aximMemcpyDeviceToDevice
  #define cudaMemcpyHostToHost     aximMemcpyHostToHost
  #define cudaGetDeviceCount       aximGetDeviceCount
  #define cudaSetDevice            aximSetDevice
  #define cudaMalloc               aximMalloc
  #define cudaFree                 aximFree
  #define cudaMemcpy               aximMemcpy
  #define cudaDeviceSynchronize    aximDeviceSynchronize
  #define cudaGetErrorString       aximGetErrorString
#endif

#endif /* AXIM_HIP_H */
