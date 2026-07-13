# Changelog

All notable changes to AXIM are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-12

First public release of **AXIM** — a CUDA-free compute and graphics runtime that runs
SYNAXIM AI models, HPC workloads, and games on NVIDIA, AMD, Intel, and Apple silicon.

### Added

- **AXIM IR** — SYNAXIM-native op set (ADD, MUL, SUB, SILU, MATVEC, INT4_MATVEC, RMSNORM,
  LOWRANK_UPDATE, LOWRANK_RETRIEVE, SWIGLU) with `Buffer`, `Kernel`, and `Module` types.
- **Python frontend** — `@axim.kernel` decorator, `run()`, `devices()`, and SYNAXIM ops
  (`int4_matvec`, `rmsnorm`, `silu`, `swiglu`, `lowrank_retrieve`).
- **CPU backend** — AVX-512 / AVX2 (x86) and NEON (ARM) SIMD with runtime ISA detection.
- **GPU backend (Metal)** — live on Apple silicon (verified on M3).
- **GPU backend (Vulkan)** — SPIR-V compute shaders authored for NVIDIA / AMD / Intel
  (runtime executor in progress).
- **Graphics** — Metal render pipeline sharing the device with AXIM compute.
- **aximBLAS / aximDNN** — tuned math libraries (sgemv, sgemm, saxpy, sdot; softmax,
  layernorm, gelu, attention_step) as cuBLAS/cuDNN-free equivalents.
- **AXIM-HIP** — CUDA-compat shim; port CUDA sources with a single `#include`.
- **`.symb` loader** — SYNAXIM / Magnus INT4 weight format loader.
- **`aximinfo`** — rocminfo-style device report tool.
- **Rust orchestrator** — device dispatch with a PyO3 Python bridge.
- **Documentation** — MkDocs site + GitHub wiki.
- **CI** — GitHub Actions for build (macOS Metal+NEON, Ubuntu Vulkan+AVX2, Rust), CodeQL,
  and docs; Dependabot for cargo/pip/actions.
- **Project files** — Apache-2.0 `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`.

### Verified (Apple M3, macOS arm64)

- CPU (NEON), GPU (Metal), and graphics render all live.
- **INT4 matvec: CPU (NEON) == GPU (Metal) exact match, zero CUDA.**
- Test suites: `test_pipeline.py` (12), `test_synaxim.py` (5), `test_loader.py` (7) —
  **24 checks passing**.

### Known limitations

- Vulkan runtime executor (SPIR-V loader) is work in progress; NVIDIA / AMD / Intel GPUs
  are not yet live end-to-end.
- Rust orchestrator requires a local Rust + maturin toolchain to build the wheel.
- Windows (DirectX 12 / Vulkan) support is planned, not yet implemented.

[1.0.0]: https://github.com/GRRN-MAKER/Aximcomp/releases/tag/v1.0.0
