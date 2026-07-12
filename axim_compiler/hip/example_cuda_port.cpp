/*
 * Example: unmodified CUDA-style host code compiling on AXIM.
 * ===========================================================
 * Define AXIM_HIP_CUDA_ALIASES so cudaMalloc/cudaMemcpy/cudaFree map to
 * AXIM automatically. This exact source pattern is what a CUDA developer
 * already has — it now runs on any GPU/CPU with zero CUDA.
 *
 * Build:
 *   c++ -std=c++14 -Iinclude example_cuda_port.cpp src/axim_hip.cpp -o /tmp/axim_hip_demo
 *   /tmp/axim_hip_demo
 */
#define AXIM_HIP_CUDA_ALIASES
#include "axim_hip.h"

#include <cstdio>

int main() {
    int count = 0;
    cudaGetDeviceCount(&count);              // → aximGetDeviceCount
    printf("AXIM-HIP: %d device(s) visible\n", count);

    char name[128];
    aximGetDeviceName(name, sizeof(name), 0);
    printf("AXIM-HIP: device 0 = %s\n", name);

    const int N = 8;
    size_t bytes = N * sizeof(float);

    float h[N];
    for (int i = 0; i < N; ++i) h[i] = (float)i;

    float* d = nullptr;
    cudaMalloc((void**)&d, bytes);           // → aximMalloc
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);  // → aximMemcpy

    float back[N] = {0};
    cudaMemcpy(back, d, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d);                              // → aximFree
    cudaDeviceSynchronize();

    bool ok = true;
    for (int i = 0; i < N; ++i) ok = ok && (back[i] == (float)i);
    printf("AXIM-HIP: roundtrip %s — CUDA source ran on AXIM, zero CUDA.\n",
           ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
