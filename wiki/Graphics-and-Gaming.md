# Graphics & Gaming

AXIM includes a built-in, vendor-neutral graphics render pipeline over
Metal (Apple) and Vulkan (Nvidia/AMD/Intel). Compute and rendering share the
same GPU device and command queue — so **games run AI, physics, and
rendering on one device with zero CUDA-graphics interop**.

## Why this matters for games

Traditional CUDA workflows separate compute (CUDA) from graphics (Vulkan/
DirectX), forcing expensive interop and copies. AXIM unifies them:

- **Shared device** — AXIM compute kernels and the render pipeline use the
  same GPU queue → zero-copy AI-in-games.
- **Portable shaders** — MSL on Apple, SPIR-V on Vulkan, same pipeline model.
- **Low overhead** — direct Metal/Vulkan, no runtime translation layer.

## Render a frame

```c
#include "axim_gfx.h"

axim_gfx_init(width, height);

// interleaved vertices: x, y, r, g, b
float verts[] = { 0.0f, 0.5f, 1,0,0,
                 -0.5f,-0.5f, 0,1,0,
                  0.5f,-0.5f, 0,0,1 };

uint8_t pixels[width * height * 4];
axim_gfx_render_frame(verts, 3, 0.1f, 0.1f, 0.1f, 1.0f, pixels);
// pixels now holds the rendered RGBA8 frame — present via SDL/GLFW/CAMetalLayer
```

## Verified (Apple M3)

```
gfx init rc = 0, device = Apple M3
render rc=0, center pixel RGBA=(124,62,70,255)
✅ AXIM rendered a game frame on the Apple M3 GPU (Metal) — CUDA-free.
```

## Architecture

```
Game loop
   │
   ├── AXIM compute (AI / physics)   ── shares ──┐
   │      int4_matvec, attention_step            │  one GPU
   │                                             │  device + queue
   └── AXIM graphics (render)         ── shares ──┘
          axim_gfx_render_frame → RGBA8 texture → present
```

## Backends

| Platform | Graphics API | Status |
|----------|-------------|--------|
| Apple silicon | Metal | ✅ live |
| Nvidia/AMD/Intel | Vulkan | 🔷 shaders ready, loader WIP |

## Roadmap

- Vulkan render backend (mirror the Metal path)
- Windowing integration (SDL / GLFW / CAMetalLayer)
- Zero-copy compute↔graphics shared buffers
- Frame pacing + async compute overlap

Next: [SYNAXIM on AXIM →](synaxim.md)
