//
// AXIM Metal Backend — Live GPU Execution (Objective-C++)
// ========================================================
// Runs AXIM compute kernels on a real Apple GPU via the Metal framework.
// Shaders are compiled at runtime from embedded MSL source, so no offline
// metal toolchain (full Xcode) is required — only the Metal.framework that
// ships with macOS. Zero CUDA.
//
// Build:
//   clang++ -std=c++17 -ObjC++ -O3 -fPIC -shared \
//     -framework Metal -framework Foundation \
//     -Iinclude src/axim_metal.mm -o build/libaxim_metal.dylib
//
// Exposes a C ABI matching backend_gpu's axim_gpu.h so the orchestrator
// can call the same functions whether the backend is Metal, Vulkan, or CPU.
//
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstring>
#include <cstdio>

// Embedded MSL source (mirrors shaders/metal/axim_kernels.metal).
static const char* AXIM_MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void axim_add(device const float* a [[buffer(0)]],
                     device const float* b [[buffer(1)]],
                     device float* out [[buffer(2)]],
                     constant uint& n [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < n) out[gid] = a[gid] + b[gid];
}

kernel void axim_mul(device const float* a [[buffer(0)]],
                     device const float* b [[buffer(1)]],
                     device float* out [[buffer(2)]],
                     constant uint& n [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < n) out[gid] = a[gid] * b[gid];
}

kernel void axim_int4_matvec(device const float* x [[buffer(0)]],
                             device const uchar* packed [[buffer(1)]],
                             device const float* scales [[buffer(2)]],
                             device const float* zeros [[buffer(3)]],
                             device float* out [[buffer(4)]],
                             constant uint& out_dim [[buffer(5)]],
                             constant uint& in_dim [[buffer(6)]],
                             constant uint& group_size [[buffer(7)]],
                             uint row [[thread_position_in_grid]]) {
    if (row >= out_dim) return;
    uint n_groups = in_dim / group_size;
    uint row_half = in_dim / 2;
    float acc = 0.0f;
    for (uint g = 0; g < n_groups; ++g) {
        float s = scales[row * n_groups + g];
        float z = zeros[row * n_groups + g];
        uint base_col = g * group_size;
        uint base_byte = base_col / 2;
        for (uint i = 0; i < group_size / 2; ++i) {
            uchar byte = packed[row * row_half + base_byte + i];
            float w_hi = float((byte >> 4) & 0x0F) * s + z;
            float w_lo = float(byte & 0x0F) * s + z;
            acc += w_hi * x[base_col + i * 2];
            acc += w_lo * x[base_col + i * 2 + 1];
        }
    }
    out[row] = acc;
}
)MSL";

// ── Global Metal state (lazy-initialized) ──
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_lib = nil;

extern "C" int axim_metal_init(void) {
    @autoreleasepool {
        if (g_device) return 0;
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) return 1;
        g_queue = [g_device newCommandQueue];
        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:AXIM_MSL];
        g_lib = [g_device newLibraryWithSource:src options:nil error:&err];
        if (!g_lib) {
            if (err) NSLog(@"AXIM Metal compile error: %@", err);
            return 2;
        }
        return 0;
    }
}

extern "C" const char* axim_metal_device_name(void) {
    static char buf[256] = {0};
    @autoreleasepool {
        if (!g_device) axim_metal_init();
        if (g_device) {
            std::snprintf(buf, sizeof(buf), "%s",
                          [[g_device name] UTF8String]);
        } else {
            std::snprintf(buf, sizeof(buf), "no Metal device");
        }
    }
    return buf;
}

// Generic 1D-grid dispatch helper for elementwise kernels (add/mul).
static int dispatch_elemwise(const char* fn, const float* a, const float* b,
                             float* out, size_t n) {
    @autoreleasepool {
        if (axim_metal_init() != 0) return 1;
        NSError* err = nil;
        id<MTLFunction> f = [g_lib newFunctionWithName:
                             [NSString stringWithUTF8String:fn]];
        id<MTLComputePipelineState> pso =
            [g_device newComputePipelineStateWithFunction:f error:&err];
        if (!pso) return 2;

        size_t bytes = n * sizeof(float);
        id<MTLBuffer> ba = [g_device newBufferWithBytes:a length:bytes
                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> bb = [g_device newBufferWithBytes:b length:bytes
                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> bo = [g_device newBufferWithLength:bytes
                            options:MTLResourceStorageModeShared];
        uint32_t n32 = (uint32_t)n;

        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:ba offset:0 atIndex:0];
        [enc setBuffer:bb offset:0 atIndex:1];
        [enc setBuffer:bo offset:0 atIndex:2];
        [enc setBytes:&n32 length:sizeof(uint32_t) atIndex:3];

        MTLSize grid = MTLSizeMake(n, 1, 1);
        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        if (tg > n) tg = n;
        MTLSize tgs = MTLSizeMake(tg, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        std::memcpy(out, [bo contents], bytes);
        return 0;
    }
}

extern "C" int axim_metal_add(const float* a, const float* b,
                              float* out, size_t n) {
    return dispatch_elemwise("axim_add", a, b, out, n);
}

extern "C" int axim_metal_mul(const float* a, const float* b,
                              float* out, size_t n) {
    return dispatch_elemwise("axim_mul", a, b, out, n);
}

// INT4 fused matvec on the GPU — one thread per output row.
extern "C" int axim_metal_int4_matvec(
        const float* x, const uint8_t* packed,
        const float* scales, const float* zeros, float* out,
        size_t out_dim, size_t in_dim, size_t group_size) {
    @autoreleasepool {
        if (axim_metal_init() != 0) return 1;
        NSError* err = nil;
        id<MTLFunction> f = [g_lib newFunctionWithName:@"axim_int4_matvec"];
        id<MTLComputePipelineState> pso =
            [g_device newComputePipelineStateWithFunction:f error:&err];
        if (!pso) return 2;

        size_t n_groups = in_dim / group_size;
        size_t packed_len = out_dim * (in_dim / 2);
        size_t sz_len = out_dim * n_groups;

        id<MTLBuffer> bx = [g_device newBufferWithBytes:x
                            length:in_dim * sizeof(float)
                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> bp = [g_device newBufferWithBytes:packed
                            length:packed_len
                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> bs = [g_device newBufferWithBytes:scales
                            length:sz_len * sizeof(float)
                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> bz = [g_device newBufferWithBytes:zeros
                            length:sz_len * sizeof(float)
                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> bo = [g_device newBufferWithLength:out_dim * sizeof(float)
                            options:MTLResourceStorageModeShared];
        uint32_t od = (uint32_t)out_dim, id_ = (uint32_t)in_dim,
                 gs = (uint32_t)group_size;

        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bx offset:0 atIndex:0];
        [enc setBuffer:bp offset:0 atIndex:1];
        [enc setBuffer:bs offset:0 atIndex:2];
        [enc setBuffer:bz offset:0 atIndex:3];
        [enc setBuffer:bo offset:0 atIndex:4];
        [enc setBytes:&od length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&id_ length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&gs length:sizeof(uint32_t) atIndex:7];

        MTLSize grid = MTLSizeMake(out_dim, 1, 1);
        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        if (tg > out_dim) tg = out_dim;
        MTLSize tgs = MTLSizeMake(tg, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        std::memcpy(out, [bo contents], out_dim * sizeof(float));
        return 0;
    }
}
