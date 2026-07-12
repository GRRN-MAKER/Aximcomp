# Roadmap

## Done ✅

- AXIM IR (SYNAXIM-native op set)
- Python frontend (`@axim.kernel` + SYNAXIM ops)
- CPU backend — AVX-512 / AVX2 / NEON (NEON verified on M3)
- GPU backend — **Metal live on Apple M3**; Vulkan shaders written
- aximBLAS (cuBLAS-class) + aximDNN (cuDNN-class) tuned libraries
- AXIM-HIP CUDA-compat shim (CUDA source runs on AXIM)
- Graphics render pipeline (Metal live — game frame on M3)
- `.symb` model loader (SYNAXIM/Magnus)
- `aximinfo` device tool (rocminfo-style)
- Rust orchestrator + PyO3 packaging (maturin)
- GitHub Actions CI (macOS/Linux/Windows), CodeQL, Dependabot
- Full docs + wiki

## Next 🔷

- **Vulkan runtime executor** — live GPU on Nvidia/AMD/Intel (mirror Metal)
- **Graphics ↔ compute shared buffers** — zero-copy AI-in-games
- **`aximify`** — HIPIFY-style CUDA → AXIM source rewriter
- **Benchmarks** — aximBLAS/aximDNN vs cuBLAS/cuDNN/OpenBLAS
- **maturin wheels on PyPI** — `pip install axim`
- **INT4 GPU low-rank + RMSNorm shaders** — full SYNAXIM layer on GPU
- **Windowing integration** — SDL / GLFW / CAMetalLayer for games
- **Multi-device scheduling** — CPU + GPU heterogeneous split

## Longer term

- Rust CPU micro-kernels (replace Numba-style paths)
- Auto-tuning for tile sizes per CPU/GPU
- WebGPU backend (browser compute)
- Distributed multi-GPU
