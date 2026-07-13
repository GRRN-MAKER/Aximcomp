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

/*
 * ── Zero-copy shared buffers (AI-in-games) ────────────────────────────
 * A shared buffer lives in GPU memory and is usable by BOTH an AXIM
 * compute kernel and the render pipeline WITHOUT a host round-trip. This
 * is the mechanism that lets a compute pass (e.g. SYNAXIM inference or
 * GPU physics) write results that the very next draw call consumes as
 * vertex/instance data on the same queue — no CPU copy, no CUDA interop.
 *
 * Typical loop:
 *   h = axim_gfx_shared_alloc(n_bytes);
 *   axim_gfx_shared_upload(h, initial, n_bytes);      // optional seed
 *   axim_gfx_compute_into(h, "spirv/axim_swiglu.spv", groups); // GPU writes
 *   axim_gfx_render_shared(h, vertex_count, r,g,b,a, out);      // GPU reads
 *   ...                                                // repeat, zero copies
 *   axim_gfx_shared_free(h);
 */
typedef uint64_t axim_gfx_buffer_t;   /* opaque handle; 0 = invalid */

/* Allocate a GPU-resident buffer shared between compute and render. */
axim_gfx_buffer_t axim_gfx_shared_alloc(size_t n_bytes);

/* Optional: seed a shared buffer from host memory (one-time). */
int  axim_gfx_shared_upload(axim_gfx_buffer_t h, const void* src, size_t n_bytes);

/* Optional: read a shared buffer back to host (debug / checkpoint). */
int  axim_gfx_shared_download(axim_gfx_buffer_t h, void* dst, size_t n_bytes);

/*
 * Run a compute shader that writes directly into shared buffer `h`, on the
 * same device/queue as rendering. `shader_path` is a SPIR-V (.spv) or, on
 * Metal, a named kernel in the loaded metallib. No host copy occurs.
 */
int  axim_gfx_compute_into(axim_gfx_buffer_t h, const char* shader_path, uint32_t groups);

/*
 * Render a frame whose vertex data IS the shared buffer `h` — the buffer a
 * compute pass just wrote — with no intervening CPU copy. Same semantics as
 * axim_gfx_render_frame otherwise.
 */
int  axim_gfx_render_shared(axim_gfx_buffer_t h, int vertex_count,
                            float r, float g, float b, float a,
                            uint8_t* out_pixels);

/* Free a shared buffer. */
void axim_gfx_shared_free(axim_gfx_buffer_t h);

#ifdef __cplusplus
}
#endif

#endif /* AXIM_GFX_H */
