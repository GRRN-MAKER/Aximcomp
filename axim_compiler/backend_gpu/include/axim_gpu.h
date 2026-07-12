/*
 * AXIM GPU Backend — Vendor-Neutral Compute
 * ==========================================
 * One abstraction, two implementations:
 *   - Vulkan Compute (SPIR-V)  → Nvidia, AMD, Intel on Linux/Windows
 *   - Metal (MSL)              → Apple silicon on macOS
 *
 * Zero CUDA. The same AXIM IR that runs on the CPU SIMD backend is
 * lowered here to compute shaders, so any SYNAXIM model runs on any GPU.
 *
 * This header defines the stable C ABI the Rust orchestrator calls.
 * The .cpp selects Metal or Vulkan at compile time via AXIM_BUILD_METAL /
 * AXIM_BUILD_VULKAN, falling back to a CPU-emulated path if neither is
 * present (so the scaffold is always runnable).
 * (Build macros are AXIM_BUILD_* so they never collide with the
 *  axim_gpu_api_t enum values AXIM_GPU_METAL / AXIM_GPU_VULKAN.)
 */
#ifndef AXIM_GPU_H
#define AXIM_GPU_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    AXIM_GPU_NONE   = 0,   /* no GPU backend compiled/available */
    AXIM_GPU_VULKAN = 1,   /* Nvidia / AMD / Intel via Vulkan */
    AXIM_GPU_METAL  = 2    /* Apple silicon via Metal */
} axim_gpu_api_t;

/* Which GPU API this build targets (compile-time). */
axim_gpu_api_t axim_gpu_api(void);

/* Human-readable GPU backend name, e.g. "Vulkan", "Metal", "none". */
const char* axim_gpu_backend_name(void);

/* Initialize the GPU device. Returns 0 on success, non-zero on failure. */
int axim_gpu_init(void);

/* Release GPU resources. */
void axim_gpu_shutdown(void);

/*
 * Dispatch a named AXIM op on the GPU.
 * For the scaffold this routes elementwise/int4 ops; real SPIR-V/MSL
 * shader modules are registered per-op as the backend matures.
 *
 * Returns 0 on success. If the GPU backend is AXIM_GPU_NONE, callers
 * should fall back to the CPU backend (the orchestrator handles this).
 */
int axim_gpu_add(const float* a, const float* b, float* out, size_t n);
int axim_gpu_mul(const float* a, const float* b, float* out, size_t n);

/* Scaffold sanity kernel: writes the GPU banner into out. */
int axim_gpu_hello(char* out, size_t out_cap);

#ifdef __cplusplus
}
#endif

#endif /* AXIM_GPU_H */
