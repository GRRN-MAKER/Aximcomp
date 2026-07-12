/*
 * AXIM Graphics — Vendor-Neutral Render Pipeline (for games)
 * ==========================================================
 * A thin graphics layer over Metal (Apple) and Vulkan (Nvidia/AMD/Intel)
 * so games and interactive apps get a single, CUDA-free graphics + compute
 * path. Compute (AXIM kernels) and render (rasterization) share the same
 * device and command queue — enabling fast physics/AI on the GPU alongside
 * rendering, without CUDA/CUDA-graphics interop headaches.
 *
 * Design goals for gaming:
 *   - Low overhead: direct Metal/Vulkan, no translation layer at runtime
 *   - Shared device: AXIM compute + graphics use one queue → zero copies
 *   - Portable shaders: MSL on Apple, SPIR-V on Vulkan, same pipeline model
 *
 * This header defines the stable C ABI; the Metal implementation
 * (axim_gfx_metal.mm) renders on Apple silicon today.
 */
#ifndef AXIM_GFX_H
#define AXIM_GFX_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    AXIM_GFX_NONE   = 0,
    AXIM_GFX_METAL  = 1,
    AXIM_GFX_VULKAN = 2
} axim_gfx_api_t;

/* Which graphics API this build targets. */
axim_gfx_api_t axim_gfx_api(void);
const char* axim_gfx_backend_name(void);

/* Initialize an offscreen render target of (width x height). 0 = success. */
int axim_gfx_init(int width, int height);
void axim_gfx_shutdown(void);

/*
 * Render a single frame: clears to (r,g,b,a) and draws `vertex_count`
 * vertices from `vertices` (interleaved x,y,r,g,b floats) as triangles.
 * Writes the rendered RGBA8 pixels into `out_pixels` (width*height*4 bytes).
 * Returns 0 on success. This is enough to drive a game frame loop.
 */
int axim_gfx_render_frame(const float* vertices, int vertex_count,
                          float r, float g, float b, float a,
                          uint8_t* out_pixels);

/* Report the GPU device backing graphics (shared with AXIM compute). */
const char* axim_gfx_device_name(void);

#ifdef __cplusplus
}
#endif

#endif /* AXIM_GFX_H */
