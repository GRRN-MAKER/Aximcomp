/*
 * AXIM Vulkan Executor — runtime self-test
 * ========================================
 * Actually RUNS a SPIR-V compute kernel through the AXIM Vulkan executor:
 * uploads two input buffers, dispatches axim_add.spv, downloads the result,
 * and checks correctness. Exercises the full path
 *   VkInstance -> VkDevice -> buffers -> pipeline -> dispatch -> readback.
 *
 * Works on any Vulkan implementation, including software Vulkan
 * (Lavapipe / SwiftShader) on CI runners that have no physical GPU — so the
 * Vulkan path is *executed*, not merely compiled. On a machine with a real
 * NVIDIA/AMD/Intel GPU the same binary measures that device.
 *
 * Build (needs Vulkan SDK):
 *   c++ -std=c++17 -DAXIM_BUILD_VULKAN -Iinclude \
 *       src/axim_vulkan_executor.cpp test_vulkan_executor.cpp \
 *       -lvulkan -o axim_vk_selftest
 */
#include "axim_vulkan_executor.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main() {
    if (axim_vk_init() != 0) {
        std::fprintf(stderr, "axim_vk_init failed (no Vulkan loader/ICD?)\n");
        return 2;
    }
    std::printf("[axim-vk] device: %s\n", axim_vk_device_name());

    const int N = 1024;
    std::vector<float> a(N), b(N), out(N, 0.0f);
    for (int i = 0; i < N; ++i) { a[i] = (float)i; b[i] = (float)(2 * i); }

    // axim_add.comp binds: 0 = A (in), 1 = B (in), 2 = Out (out)
    // and a push constant `n` (uint element count).
    void*  bufs[3]  = { a.data(), b.data(), out.data() };
    size_t sizes[3] = { N * sizeof(float), N * sizeof(float), N * sizeof(float) };
    int    isout[3] = { 0, 0, 1 };

    // local_size_x = 256 in the shader → one workgroup per 256 elements
    const uint32_t groups = (N + 255) / 256;
    const uint32_t push_n = (uint32_t)N;

    if (axim_vk_run_spirv_pc("shaders/spirv/axim_add.spv",
                             bufs, sizes, isout, 3, groups,
                             &push_n, sizeof(push_n)) != 0) {
        std::fprintf(stderr, "axim_vk_run_spirv failed\n");
        axim_vk_shutdown();
        return 3;
    }

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expect = a[i] + b[i];
        if (std::fabs(out[i] - expect) > 1e-5f) {
            if (errors < 5)
                std::fprintf(stderr, "mismatch @%d: got %f want %f\n", i, out[i], expect);
            ++errors;
        }
    }
    axim_vk_shutdown();

    if (errors) {
        std::fprintf(stderr, "[axim-vk] FAIL: %d mismatches\n", errors);
        return 1;
    }
    std::printf("[axim-vk] PASS: add of %d elements correct via Vulkan dispatch.\n", N);
    return 0;
}
