/*
 * AXIM-HIP — Implementation
 * =========================
 * Backs the CUDA-compatible API with AXIM's vendor-neutral runtime.
 * Device buffers are host-shared allocations (unified memory model) so the
 * same pointer works on CPU and integrated/discrete GPUs. Zero CUDA.
 *
 * This is deliberately simple and dependency-free for the scaffold. It
 * routes to the CPU backend by default and can be linked with the Metal /
 * Vulkan backends for real device execution.
 */
#include "axim_hip.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>

/* Simple device registry (query-only for the scaffold). */
static int g_current_device = 0;

aximError_t aximGetDeviceCount(int* count) {
    if (!count) return aximErrorInvalidValue;
    /* AXIM always exposes at least the CPU device; GPU adds one more. */
    *count = 1; /* CPU */
#if defined(__APPLE__)
    *count += 1; /* Metal GPU */
#endif
    return aximSuccess;
}

aximError_t aximSetDevice(int device) {
    int n = 0;
    aximGetDeviceCount(&n);
    if (device < 0 || device >= n) return aximErrorNoDevice;
    g_current_device = device;
    return aximSuccess;
}

aximError_t aximGetDeviceName(char* name, int len, int device) {
    if (!name || len <= 0) return aximErrorInvalidValue;
    const char* n = (device == 0) ? "AXIM CPU (SIMD)" : "AXIM GPU (Vulkan/Metal)";
    std::snprintf(name, len, "%s", n);
    return aximSuccess;
}

/* ── Memory (unified/host-shared model) ── */
aximError_t aximMalloc(void** ptr, size_t size) {
    if (!ptr) return aximErrorInvalidValue;
    void* p = std::malloc(size);
    if (!p) return aximErrorMemoryAllocation;
    *ptr = p;
    return aximSuccess;
}

aximError_t aximFree(void* ptr) {
    std::free(ptr);
    return aximSuccess;
}

aximError_t aximMemcpy(void* dst, const void* src, size_t size,
                       aximMemcpyKind /*kind*/) {
    if (!dst || !src) return aximErrorInvalidValue;
    /* Unified memory: all kinds are a host memcpy in the scaffold. */
    std::memcpy(dst, src, size);
    return aximSuccess;
}

aximError_t aximLaunch(const char* kernel_name, size_t /*grid*/,
                       void** /*args*/, int /*n_args*/) {
    if (!kernel_name) return aximErrorInvalidValue;
    /* Kernel dispatch is routed through the AXIM orchestrator in the full
     * build; the scaffold validates the name is a known AXIM op. */
    static const char* known[] = {
        "add", "mul", "sub", "silu", "int4_matvec",
        "rmsnorm", "lowrank_retrieve", "swiglu",
    };
    for (auto k : known) {
        if (std::strcmp(k, kernel_name) == 0) return aximSuccess;
    }
    return aximErrorNotSupported;
}

aximError_t aximDeviceSynchronize(void) {
    return aximSuccess; /* synchronous scaffold */
}

const char* aximGetErrorString(aximError_t err) {
    switch (err) {
        case aximSuccess: return "aximSuccess";
        case aximErrorInvalidValue: return "aximErrorInvalidValue";
        case aximErrorMemoryAllocation: return "aximErrorMemoryAllocation";
        case aximErrorNoDevice: return "aximErrorNoDevice";
        case aximErrorNotSupported: return "aximErrorNotSupported";
        default: return "aximErrorUnknown";
    }
}
