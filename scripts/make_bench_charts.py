#!/usr/bin/env python3
"""
Render AXIM benchmark charts (throughput + latency) to PNG for the paper/docs.
Uses the same data as notebooks/axim_bench.py. Zero CUDA.

Usage:
    python3 scripts/make_bench_charts.py
Outputs:
    docs/assets/axim_throughput.png
    docs/assets/axim_latency.png
"""
import os
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd

# Same data as the marimo notebook. Replace Vulkan rows with real
# axim_vk_bench output from the cloud NVIDIA instance when available.
RAW = [
    ("CPU-NEON",  "Apple M3",    4096,    0.0021),
    ("CPU-NEON",  "Apple M3",   65536,    0.0298),
    ("CPU-NEON",  "Apple M3", 1048576,    0.4900),
    ("CPU-NEON",  "Apple M3", 4194304,    1.9800),
    ("GPU-Metal", "Apple M3",    4096,    0.0180),
    ("GPU-Metal", "Apple M3",   65536,    0.0230),
    ("GPU-Metal", "Apple M3", 1048576,    0.1400),
    ("GPU-Metal", "Apple M3", 4194304,    0.5200),
]

df = pd.DataFrame(RAW, columns=["backend", "device", "elements", "avg_ms"])
df["m_elem_per_s"] = (df["elements"] / (df["avg_ms"] / 1000.0)) / 1e6

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outdir = os.path.join(here, "docs", "assets")
os.makedirs(outdir, exist_ok=True)

# ── Throughput (higher is better) ──
fig1, ax1 = plt.subplots(figsize=(7, 4))
for backend, g in df.groupby("backend"):
    g = g.sort_values("elements")
    ax1.plot(g["elements"], g["m_elem_per_s"], marker="o", label=backend)
ax1.set_xscale("log", base=2)
ax1.set_xlabel("elements (log2)")
ax1.set_ylabel("throughput (M elem/s)")
ax1.set_title("AXIM throughput by backend (Apple M3)")
ax1.legend()
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
p1 = os.path.join(outdir, "axim_throughput.png")
fig1.savefig(p1, dpi=150)

# ── Latency (lower is better) ──
fig2, ax2 = plt.subplots(figsize=(7, 4))
for backend, g in df.groupby("backend"):
    g = g.sort_values("elements")
    ax2.plot(g["elements"], g["avg_ms"], marker="s", label=backend)
ax2.set_xscale("log", base=2)
ax2.set_yscale("log")
ax2.set_xlabel("elements (log2)")
ax2.set_ylabel("avg ms / dispatch (log)")
ax2.set_title("AXIM per-dispatch latency by backend (Apple M3)")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
p2 = os.path.join(outdir, "axim_latency.png")
fig2.savefig(p2, dpi=150)

print("wrote:", p1)
print("wrote:", p2)
