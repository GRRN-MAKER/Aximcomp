# API Reference

## Python — `axim_compiler`

### Kernel authoring

| Function | Description |
|----------|-------------|
| `@axim.kernel` | Decorate a Python function as an AXIM kernel |
| `axim.run(fn, *args, device="auto")` | Execute a kernel on a device |
| `axim.devices()` | List targetable devices |
| `axim.hello()` | Print the device banner |

### SYNAXIM ops

| Function | Description |
|----------|-------------|
| `axim.int4_matvec(x, packed, scales, zeros, out_dim, in_dim, group_size, device)` | Fused INT4 matvec |
| `axim.rmsnorm(x, weight, eps, device)` | RMS normalization |
| `axim.silu(x, device)` | SiLU activation |
| `axim.swiglu(x, ...)` | Fused SwiGLU MLP stage |
| `axim.lowrank_retrieve(q, U, V, D, r, device)` | Low-rank memory read |

### Loader — `axim_compiler.loader`

| Function | Description |
|----------|-------------|
| `load_symb_int4(path)` | Load one INT4 `.symb` weight |
| `load_symb_fp16(path)` | Load an FP16 `.symb` vector |
| `load_config(model_dir)` | Load `config.symb.json` |
| `load_model(model_dir, max_layers)` | Load a full model |

## C — Backends

### CPU (`axim_cpu.h`)

`axim_cpu_add/mul/sub/silu`, `axim_cpu_rmsnorm`, `axim_cpu_int4_matvec`,
`axim_cpu_lowrank_retrieve`, `axim_cpu_detect_simd`, `axim_cpu_backend_name`.

### aximBLAS (`axim_blas.h`)

`axim_sgemv`, `axim_sgemm`, `axim_saxpy`, `axim_sdot`.

### aximDNN (`axim_dnn.h`)

`axim_softmax`, `axim_layernorm`, `axim_gelu`, `axim_attention_step`.

### GPU (`axim_gpu.h` / Metal)

`axim_metal_init`, `axim_metal_device_name`, `axim_metal_add/mul`,
`axim_metal_int4_matvec`.

### Graphics (`axim_gfx.h`)

`axim_gfx_init`, `axim_gfx_render_frame`, `axim_gfx_device_name`,
`axim_gfx_shutdown`.

### AXIM-HIP (`axim_hip.h`)

`aximMalloc`, `aximFree`, `aximMemcpy`, `aximGetDeviceCount`,
`aximSetDevice`, `aximLaunch`, `aximDeviceSynchronize`,
`aximGetErrorString`. Define `AXIM_HIP_CUDA_ALIASES` for CUDA aliases.

## Rust — `axim_orchestrator`

`dispatch_add`, `cpu_backend`, `gpu_backend`, `gpu_available`, `hello`.
PyO3 module `axim._native` exposes `add`, `cpu_backend_name`,
`gpu_backend_name`, `has_gpu`, `banner`.
