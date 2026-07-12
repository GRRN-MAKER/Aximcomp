# Building with CI

**You never have to build AXIM on your own PC.** GitHub Actions builds every
backend on cloud runners — installing Rust, the Vulkan SDK, and using the
Xcode that ships on macOS runners.

## Workflows

| Workflow | File | What it does |
|----------|------|--------------|
| Build | `.github/workflows/build.yml` | Builds CPU/GPU/graphics/HIP + Rust wheel on macOS, Linux, Windows |
| CodeQL | `.github/workflows/codeql.yml` | Security code scanning (Python + C/C++) |
| Docs | `.github/workflows/docs.yml` | Publishes this documentation to GitHub Pages |

## What each runner builds

=== "macOS runner"
    - CPU backend (NEON) + aximBLAS + aximDNN → `.dylib`
    - Metal GPU backend (live) → `.dylib`
    - Metal graphics backend (live) → `.dylib`
    - Metal shaders → `.metallib` (Xcode pre-installed)
    - AXIM-HIP shim demo
    - All tests

=== "Ubuntu runner"
    - Installs **Vulkan SDK** (`glslang-tools`, `libvulkan-dev`)
    - CPU backend (AVX2) + tuned libs → `.so`
    - Vulkan compute shaders → **SPIR-V** `.spv`
    - GPU backend ABI (Vulkan path) → `.so`
    - AXIM-HIP shim demo
    - All tests

=== "Rust job (all OS)"
    - Installs **Rust** via `dtolnay/rust-toolchain`
    - Installs **maturin**
    - Builds the Rust orchestrator (`cargo build --release`)
    - Builds the Python wheel (`maturin build --release --features python`)

## Artifacts

Each run uploads downloadable artifacts:

- `axim-macos-metal` — Metal dylibs + metallib
- `axim-linux-vulkan` — Linux .so + SPIR-V shaders
- `axim-wheel-{os}` — Python wheels

## Run it manually

Go to the **Actions** tab → select **AXIM Build** → **Run workflow**. This
triggers all three OS builds without touching your machine.

## Security automation

- **CodeQL** scans Python and C/C++ on every push and weekly.
- **Dependabot** (`.github/dependabot.yml`) checks Rust, pip, and Actions
  dependencies weekly.
