# AXIM Notebooks (marimo)

Reactive [marimo](https://marimo.io) notebooks for exploring AXIM.

marimo notebooks are **pure Python files** — reactive (cells re-run based on
dependencies, no hidden execution-order state) and runnable as scripts or apps.

## Setup

```bash
pip install -r notebooks/requirements.txt
```

## `axim_bench.py` — Benchmark Explorer

Interactive comparison of AXIM backends (CPU-SIMD, GPU-Metal, GPU-Vulkan):
throughput and latency charts that update reactively when you edit the data.

```bash
# Edit interactively (opens in browser)
marimo edit notebooks/axim_bench.py

# Run as a read-only app
marimo run notebooks/axim_bench.py

# Or just execute it as a plain Python script
python notebooks/axim_bench.py
```

### Adding your real GPU numbers

1. On a machine with an NVIDIA/AMD/Intel GPU:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/GRRN-MAKER/Aximcomp/main/scripts/run_gpu_bench.sh | bash
   ```
2. Copy the `axim_vk_bench` table into the **data cell** of `axim_bench.py`.
3. The charts update automatically.

*Zero CUDA. The Vulkan path is vendor-neutral — the same SPIR-V runs on all vendors.*
