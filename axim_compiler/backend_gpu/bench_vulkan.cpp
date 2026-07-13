/*
 * AXIM Vulkan Benchmark — real-GPU latency/throughput
 * ===================================================
 * Runs AXIM's SPIR-V kernels through the Vulkan executor on whatever GPU the
 * Vulkan loader selects (a real NVIDIA/AMD/Intel card, or software Lavapipe),
 * verifies correctness, then times repeated dispatches to report throughput.
 *
 * Zero CUDA. Build (needs Vulkan SDK):
 *   c++ -std=c++17 -O3 -DAXIM_BUILD_VULKAN -Iinclude \
 *       src/axim_vulkan_executor.cpp bench_vulkan.cpp -lvulkan -o axim_vk_bench
 * Run from backend_gpu/ (so shaders/spirv/*.spv resolve):
 *   ./axim_vk_bench
 */
#include "axim_vulkan_executor.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>

using clk = std::chrono::high_resolution_clock;

static double time_add(int N, int iters) {
    std::vector<float> a(N), b(N), out(N, 0.0f);
    for (int i = 0; i < N; ++i) { a[i] = (float)i; b[i] = (float)(2 * i); }
    void*  bufs[3]  = { a.data(), b.data(), out.data() };
    size_t sizes[3] = { N * sizeof(float), N * sizeof(float), N * sizeof(float) };
    int    isout[3] = { 0, 0, 1 };
    const uint32_t groups = (N + 255) / 256;
    const uint32_t push_n = (uint32_t)N;

    // warmup + correctness
    axim_vk_run_spirv_pc("shaders/spirv/axim_add.spv", bufs, sizes, isout, 3, groups,
                         &push_n, sizeof(push_n));
    for (int i = 0; i < N; ++i) {
        if (std::fabs(out[i] - (a[i] + b[i])) > 1e-5f) {
            std::fprintf(stderr, "add correctness FAIL @%d\n", i);
            return -1.0;
        }
    }

    auto t0 = clk::now();
    for (int it = 0; it < iters; ++it) {
        axim_vk_run_spirv_pc("shaders/spirv/axim_add.spv", bufs, sizes, isout, 3, groups,
                             &push_n, sizeof(push_n));
    }
    auto t1 = clk::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return ms / iters;   // avg ms per dispatch (incl. upload+download)
}

int main() {
    if (axim_vk_init() != 0) {
        std::fprintf(stderr, "axim_vk_init failed (no Vulkan loader/ICD?)\n");
        return 2;
    }
    std::printf("=======================================================\n");
    std::printf(" AXIM Vulkan Benchmark\n");
    std::printf(" Device: %s\n", axim_vk_device_name());
    std::printf("=======================================================\n");

    const int sizes[]  = { 4096, 65536, 1048576, 4194304 };
    const int iters[]  = { 200,  200,   100,     50 };
    std::printf("%12s  %14s  %14s\n", "elements", "avg ms/dispatch", "M elem/s");
    for (int k = 0; k < 4; ++k) {
        double ms = time_add(sizes[k], iters[k]);
        if (ms < 0) { axim_vk_shutdown(); return 1; }
        double meps = (sizes[k] / (ms / 1000.0)) / 1e6;
        std::printf("%12d  %14.4f  %14.1f\n", sizes[k], ms, meps);
    }

    axim_vk_shutdown();
    std::printf("-------------------------------------------------------\n");
    std::printf("Note: timing includes host<->device copy each dispatch\n"
                "(worst case). Persistent-buffer timing will be higher.\n");
    return 0;
}
