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
        # AXIM GPU Benchmark — RTX Pro 6000 (Blackwell)

        Runs AXIM's **vendor-neutral Vulkan path** on a real NVIDIA GPU and
        (optionally) compares against **PyTorch/CUDA** on the same hardware.

        Run cells top to bottom. Everything happens in `/tmp/axim-bench-*`, and
        the setup cell installs only Vulkan tooling (glslang, loader, tools).

        > **Honesty note:** AXIM's Vulkan path and a CUDA/PyTorch matmul are not
        > the identical computation, so treat the comparison as *portability +
        > cost* evidence, not a claim of raw-throughput parity with hand-tuned
        > cuBLAS. All numbers below are measured live on this machine.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## 1. Environment — GPU, driver, Vulkan""")
    return


@app.cell
def _():
    import subprocess, shutil

    def run(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True, timeout=120).stdout.strip()
        except Exception as e:
            return f"(error: {e})"

    gpu = run("nvidia-smi --query-gpu=name,driver_version,memory.total "
              "--format=csv,noheader")
    has_vulkaninfo = shutil.which("vulkaninfo") is not None
    vk = run("vulkaninfo --summary 2>/dev/null | grep -iE 'deviceName|driverName|apiVersion' | head -6") \
        if has_vulkaninfo else "(vulkaninfo not installed yet — run setup cell)"
    print("GPU:", gpu)
    print("Vulkan:", vk)
    return run, shutil, subprocess


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. Setup (installs Vulkan tooling, clones AXIM, builds executor)

        Only run this once. On Ubuntu it uses `sudo apt-get`. It records the
        packages it installs so you can remove them later if you want.
        """
    )
    return


@app.cell
def _(run):
    setup = r"""
    set -e
    W=$(mktemp -d /tmp/axim-bench.XXXXXX); echo "WORKDIR=$W"
    dpkg --get-selections | awk '{print $1}' | sort > "$W/pkgs_before.txt" 2>/dev/null || true
    sudo apt-get update -qq 2>/dev/null || true
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      git build-essential glslang-tools libvulkan-dev vulkan-tools 2>/dev/null || true
    dpkg --get-selections | awk '{print $1}' | sort > "$W/pkgs_after.txt" 2>/dev/null || true
    comm -13 "$W/pkgs_before.txt" "$W/pkgs_after.txt" > "$W/pkgs_added.txt" 2>/dev/null || true
    echo "== packages added (remove later with: sudo apt-get purge <list>) =="
    cat "$W/pkgs_added.txt" 2>/dev/null || true

    cd "$W"
    git clone --depth 1 -q https://github.com/GRRN-MAKER/Aximcomp.git repo
    cd repo/axim_compiler/backend_gpu
    mkdir -p shaders/spirv build
    for f in shaders/vulkan/*.comp; do
      glslangValidator -V "$f" -o "shaders/spirv/$(basename "$f" .comp).spv"
    done
    c++ -std=c++17 -O3 -DAXIM_BUILD_VULKAN -Iinclude \
      src/axim_vulkan_executor.cpp bench_vulkan.cpp -lvulkan -o build/axim_vk_bench
    c++ -std=c++17 -O3 -DAXIM_BUILD_VULKAN -Iinclude \
      src/axim_vulkan_executor.cpp test_vulkan_executor.cpp -lvulkan -o build/axim_vk_selftest
    echo "WORKDIR_READY=$W"
    """
    out = run(setup)
    print(out)
    # capture the workdir for later cells
    workdir = None
    for line in out.splitlines():
        if line.startswith("WORKDIR_READY="):
            workdir = line.split("=", 1)[1]
    print("workdir =", workdir)
    return (workdir,)


@app.cell
def _(mo):
    mo.md(r"""## 3. Correctness — Vulkan self-test on the Blackwell GPU""")
    return


@app.cell
def _(run, workdir):
    if workdir:
        st = run(f"cd {workdir}/repo/axim_compiler/backend_gpu && ./build/axim_vk_selftest")
        print(st)
    else:
        print("Run the setup cell first.")
    return


@app.cell
def _(mo):
    mo.md(r"""## 4. Benchmark — AXIM Vulkan latency/throughput (real GPU)""")
    return


@app.cell
def _(run, workdir):
    if workdir:
        bench = run(f"cd {workdir}/repo/axim_compiler/backend_gpu && ./build/axim_vk_bench")
        print(bench)
    else:
        print("Run the setup cell first.")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 5. (Optional) Head-to-head vs PyTorch / CUDA

        If `torch` with CUDA is installed on this box, this times an equivalent
        elementwise op on the SAME GPU so you have an honest reference point.
        """
    )
    return


@app.cell
def _():
    import time
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            print("CUDA device:", torch.cuda.get_device_name(0))
            for n in (4096, 65536, 1048576, 4194304):
                a = torch.rand(n, device=dev)
                b = torch.rand(n, device=dev)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(200):
                    c = a + b
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) / 200 * 1000
                meps = (n / (ms / 1000.0)) / 1e6
                print(f"  torch/CUDA  n={n:>8}  {ms:.4f} ms  {meps:,.0f} M elem/s")
        else:
            print("torch present but CUDA not available.")
    except ImportError:
        print("torch not installed — skip. (AXIM Vulkan numbers above stand on their own.)")
    return


if __name__ == "__main__":
    app.run()
