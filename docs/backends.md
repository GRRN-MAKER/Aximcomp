# Backends — CPU & GPU

AXIM has two execution backends behind one API. The orchestrator picks the
best available device and falls back gracefully.

## CPU Backend — SIMD

`backend_cpu/` compiles to native machine code with runtime ISA dispatch:

| ISA | Platform | Lanes |
|-----|----------|-------|
| AVX-512 | Intel Xeon | 16 float32 |
| AVX2 | AMD EPYC / consumer x86 | 8 float32 |
| NEON | Apple silicon / ARM | 4 float32 |
| scalar | fallback | 1 |

Detection is automatic (`axim_cpu_detect_simd`). Kernels: `add`, `mul`,
`sub`, `silu`, `rmsnorm`, `int4_matvec`, `lowrank_retrieve`, plus the tuned
[aximBLAS/aximDNN](libraries.md) routines.

## GPU Backend — Vulkan / Metal

`backend_gpu/` targets vendor-neutral compute:

| API | Platform | Status |
|-----|----------|--------|
| Metal (MSL) | Apple silicon | ✅ live |
| Vulkan (SPIR-V) | Nvidia / AMD / Intel | 🔷 shaders ready |

Shaders live in `backend_gpu/shaders/`:

- `metal/axim_kernels.metal` — MSL compute kernels
- `vulkan/*.comp` — GLSL compute → SPIR-V via `glslangValidator`

The Metal backend (`axim_metal.mm`) compiles MSL at runtime from the system
Metal framework, so no offline toolchain is needed to run on Apple GPUs.

## Dispatch

```python
axim.int4_matvec(..., device="cpu")   # force CPU SIMD
axim.int4_matvec(..., device="gpu")   # force GPU (Metal/Vulkan)
axim.int4_matvec(..., device="auto")  # GPU if available, else CPU
```

## Verified

| Op | CPU (NEON) | GPU (Metal M3) | Match |
|----|-----------|----------------|-------|
| int4_matvec | ✅ | ✅ | 0.00e+00 |
| add / mul | ✅ | ✅ | exact |
