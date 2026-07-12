/*
 * AXIM GPU Backend — Implementation (Vulkan / Metal / emulated fallback)
 * ======================================================================
 * Compile-time selection:
 *   -DAXIM_GPU_METAL   → Metal path (macOS, Apple silicon)
 *   -DAXIM_GPU_VULKAN  → Vulkan path (Nvidia / AMD / Intel)
 *   (neither)          → CPU-emulated GPU path so the scaffold always runs
 *
 * No CUDA is referenced or linked anywhere in this file.
 *
 * NOTE: The scaffold ships the emulated path plus the stable ABI. The
 * real SPIR-V (Vulkan) and MSL (Metal) shader modules are added per-op
 * as the backend matures — the ABI below does not change.
 */
#include "axim_gpu.h"

#include <cstdio>
#include <cstring>

/* ── API selection ──
 * Build selector macros are AXIM_BUILD_* to avoid colliding with the
 * axim_gpu_api_t enum values (AXIM_GPU_METAL / AXIM_GPU_VULKAN). */
axim_gpu_api_t axim_gpu_api(void) {
#if defined(AXIM_BUILD_METAL)
    return AXIM_GPU_METAL;
#elif defined(AXIM_BUILD_VULKAN)
    return AXIM_GPU_VULKAN;
#else
    return AXIM_GPU_NONE;
#endif
}

const char* axim_gpu_backend_name(void) {
    switch (axim_gpu_api()) {
        case AXIM_GPU_METAL:  return "Metal";
        case AXIM_GPU_VULKAN: return "Vulkan";
        default:              return "none (CPU-emulated)";
    }
}

int axim_gpu_init(void) {
#if defined(AXIM_BUILD_METAL)
    /* TODO: create MTLDevice + command queue */
    return 0;
#elif defined(AXIM_BUILD_VULKAN)
    /* TODO: create VkInstance, pick physical device, create VkDevice */
    return 0;
#else
    return 0; /* emulated path always "initializes" */
#endif
}

void axim_gpu_shutdown(void) {
    /* release device handles when real backends are wired */
}

/*
 * Elementwise ops. When a real GPU backend is present these enqueue a
 * compute shader (SPIR-V / MSL). In the scaffold they compute on the
 * host so the whole pipeline is exercisable end-to-end without a GPU.
 */
int axim_gpu_add(const float* a, const float* b, float* out, size_t n) {
    if (axim_gpu_api() == AXIM_GPU_NONE) {
        for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
        return 0;
    }
    /* TODO: dispatch SPIR-V/MSL "add" compute shader */
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
    return 0;
}

int axim_gpu_mul(const float* a, const float* b, float* out, size_t n) {
    if (axim_gpu_api() == AXIM_GPU_NONE) {
        for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
        return 0;
    }
    /* TODO: dispatch SPIR-V/MSL "mul" compute shader */
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
    return 0;
}

int axim_gpu_hello(char* out, size_t out_cap) {
    return std::snprintf(
        out, out_cap,
        "AXIM GPU backend online — API=%s — CUDA-free, vendor-neutral.",
        axim_gpu_backend_name());
}
