import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # AXIM Benchmark Explorer

        Interactive report for **AXIM** — the CUDA-free compute + graphics runtime.
        Compare correctness and performance of the same kernel across backends:

        - **CPU (SIMD)** — AVX-512 / AVX2 / NEON
        - **GPU (Metal)** — Apple silicon (verified on M3)
        - **GPU (Vulkan)** — NVIDIA / AMD / Intel (paste your `axim_vk_bench` numbers below)

        > Reactive notebook: edit the data cell and every chart updates automatically.
        > Run with `marimo edit notebooks/axim_bench.py` or `marimo run notebooks/axim_bench.py`.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 1. Paste benchmark data""")
    return


@app.cell
def _():
    # Each row: (backend, device, elements, avg_ms_per_dispatch)
    # Replace the Vulkan rows with the real output of `axim_vk_bench` on your
    # cloud NVIDIA instance (columns: elements, avg ms/dispatch).
    RAW = [
        # backend,   device,                    elements,  avg_ms
        ("CPU-NEON",  "Apple M3",                  4096,    0.0021),
        ("CPU-NEON",  "Apple M3",                 65536,    0.0298),
        ("CPU-NEON",  "Apple M3",               1048576,    0.4900),
        ("CPU-NEON",  "Apple M3",               4194304,    1.9800),
        ("GPU-Metal", "Apple M3",                  4096,    0.0180),
        ("GPU-Metal", "Apple M3",                 65536,    0.0230),
        ("GPU-Metal", "Apple M3",               1048576,    0.1400),
        ("GPU-Metal", "Apple M3",               4194304,    0.5200),
        # ── PASTE NVIDIA/AMD/Intel Vulkan numbers here ──
        # ("GPU-Vulkan", "NVIDIA RTX ....",        4096,    0.00xx),
        # ("GPU-Vulkan", "NVIDIA RTX ....",       65536,    0.00xx),
        # ("GPU-Vulkan", "NVIDIA RTX ....",     1048576,    0.0xxx),
        # ("GPU-Vulkan", "NVIDIA RTX ....",     4194304,    0.xxxx),
    ]
    return (RAW,)


@app.cell
def _(RAW):
    import pandas as pd

    df = pd.DataFrame(RAW, columns=["backend", "device", "elements", "avg_ms"])
    # Derived metric: millions of elements processed per second.
    df["m_elem_per_s"] = (df["elements"] / (df["avg_ms"] / 1000.0)) / 1e6
    return df, pd


@app.cell
def _(df, mo):
    mo.ui.table(df, label="Raw + derived metrics")
    return


@app.cell
def _(mo):
    mo.md(r"""## 2. Throughput by backend (higher is better)""")
    return


@app.cell
def _(df, mo):
    # marimo renders matplotlib figures natively; matplotlib is broadly available.
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    for _backend, _g in df.groupby("backend"):
        _g = _g.sort_values("elements")
        ax1.plot(_g["elements"], _g["m_elem_per_s"], marker="o", label=_backend)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("elements (log2)")
    ax1.set_ylabel("throughput (M elem/s)")
    ax1.set_title("AXIM throughput by backend")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1
    return plt,


@app.cell
def _(mo):
    mo.md(r"""## 3. Latency by backend (lower is better)""")
    return


@app.cell
def _(df, mo, plt):
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for _backend, _g in df.groupby("backend"):
        _g = _g.sort_values("elements")
        ax2.plot(_g["elements"], _g["avg_ms"], marker="s", label=_backend)
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("elements (log2)")
    ax2.set_ylabel("avg ms / dispatch (log)")
    ax2.set_title("AXIM per-dispatch latency by backend")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 4. How to get the Vulkan numbers

        On a machine with an NVIDIA / AMD / Intel GPU (e.g. your cloud instance):

        ```bash
        curl -fsSL https://raw.githubusercontent.com/GRRN-MAKER/Aximcomp/main/scripts/run_gpu_bench.sh | bash
        ```

        Copy the `axim_vk_bench` table (device name + elements + avg ms/dispatch) into the
        **data cell** above. The charts update automatically — no re-run needed.

        *Zero CUDA. The Vulkan path is vendor-neutral: the same SPIR-V runs on NVIDIA, AMD, and Intel.*
        """
    )
    return


if __name__ == "__main__":
    app.run()
