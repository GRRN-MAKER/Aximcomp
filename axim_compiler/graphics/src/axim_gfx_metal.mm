//
// AXIM Graphics — Metal Render Backend (Objective-C++)
// =====================================================
// Vendor-neutral game render path on Apple silicon. Shares the GPU with
// AXIM compute so games run AI/physics kernels and rendering on one device,
// zero CUDA. Renders offscreen to an RGBA8 texture (a full game frame),
// which a windowing layer (SDL/GLFW/CAMetalLayer) can present.
//
// Build:
//   clang++ -std=c++17 -ObjC++ -O3 -fPIC -shared \
//     -framework Metal -framework Foundation \
//     -Iinclude src/axim_gfx_metal.mm -o build/libaxim_gfx.dylib
//
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "axim_gfx.h"
#include <cstring>
#include <cstdio>

static const char* AXIM_GFX_MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct VIn  { float2 pos; float3 color; };
struct VOut { float4 pos [[position]]; float3 color; };

vertex VOut axim_vs(uint vid [[vertex_id]],
                    device const float* verts [[buffer(0)]]) {
    // interleaved: x, y, r, g, b  (5 floats per vertex)
    VOut o;
    float x = verts[vid * 5 + 0];
    float y = verts[vid * 5 + 1];
    o.pos = float4(x, y, 0.0, 1.0);
    o.color = float3(verts[vid*5+2], verts[vid*5+3], verts[vid*5+4]);
    return o;
}

fragment float4 axim_fs(VOut in [[stage_in]]) {
    return float4(in.color, 1.0);
}
)MSL";

static id<MTLDevice> g_dev = nil;
static id<MTLCommandQueue> g_q = nil;
static id<MTLRenderPipelineState> g_pso = nil;
static id<MTLTexture> g_tex = nil;
static int g_w = 0, g_h = 0;

axim_gfx_api_t axim_gfx_api(void) { return AXIM_GFX_METAL; }
const char* axim_gfx_backend_name(void) { return "Metal"; }

const char* axim_gfx_device_name(void) {
    static char buf[256] = {0};
    @autoreleasepool {
        if (!g_dev) g_dev = MTLCreateSystemDefaultDevice();
        std::snprintf(buf, sizeof(buf), "%s",
                      g_dev ? [[g_dev name] UTF8String] : "no Metal device");
    }
    return buf;
}

int axim_gfx_init(int width, int height) {
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) return 1;
        g_q = [g_dev newCommandQueue];
        g_w = width; g_h = height;

        NSError* err = nil;
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:
            [NSString stringWithUTF8String:AXIM_GFX_MSL] options:nil error:&err];
        if (!lib) { if (err) NSLog(@"AXIM gfx compile: %@", err); return 2; }

        MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
        pd.vertexFunction   = [lib newFunctionWithName:@"axim_vs"];
        pd.fragmentFunction = [lib newFunctionWithName:@"axim_fs"];
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        g_pso = [g_dev newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!g_pso) { if (err) NSLog(@"AXIM gfx pso: %@", err); return 3; }

        MTLTextureDescriptor* td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
            width:width height:height mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        g_tex = [g_dev newTextureWithDescriptor:td];
        return 0;
    }
}

void axim_gfx_shutdown(void) {
    g_pso = nil; g_tex = nil; g_q = nil; g_dev = nil;
}

int axim_gfx_render_frame(const float* vertices, int vertex_count,
                          float r, float g, float b, float a,
                          uint8_t* out_pixels) {
    @autoreleasepool {
        if (!g_pso || !g_tex) return 1;

        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = g_tex;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(r, g, b, a);

        id<MTLBuffer> vbuf = nil;
        if (vertices && vertex_count > 0) {
            vbuf = [g_dev newBufferWithBytes:vertices
                    length:vertex_count * 5 * sizeof(float)
                    options:MTLResourceStorageModeShared];
        }

        id<MTLCommandBuffer> cb = [g_q commandBuffer];
        id<MTLRenderCommandEncoder> enc =
            [cb renderCommandEncoderWithDescriptor:rp];
        [enc setRenderPipelineState:g_pso];
        if (vbuf) {
            [enc setVertexBuffer:vbuf offset:0 atIndex:0];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0 vertexCount:vertex_count];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (out_pixels) {
            [g_tex getBytes:out_pixels
                bytesPerRow:g_w * 4
                 fromRegion:MTLRegionMake2D(0, 0, g_w, g_h)
                mipmapLevel:0];
        }
        return 0;
    }
}
