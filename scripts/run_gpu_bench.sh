#!/usr/bin/env bash
# =============================================================================
# AXIM — Real-GPU Vulkan benchmark (NVIDIA / AMD / Intel), Ubuntu/Debian
# =============================================================================
# Copy this file to your cloud GPU instance and run it. It:
#   1. installs the Vulkan SDK + NVIDIA Vulkan ICD deps (no CUDA),
#   2. clones AXIM,
#   3. compiles the SPIR-V shaders + Vulkan executor,
#   4. runs the self-test (correctness) and the benchmark (timing),
#   5. prints the real GPU device name and numbers you can paste back.
#
# Usage on the instance:
#   curl -fsSL https://raw.githubusercontent.com/GRRN-MAKER/Aximcomp/main/scripts/run_gpu_bench.sh | bash
#   # or: scp this file over, then:  bash run_gpu_bench.sh
#
# Zero CUDA is installed or used. This runs the vendor-neutral Vulkan path.
set -euo pipefail

echo "== AXIM GPU benchmark: environment =="
uname -a || true
# NVIDIA driver + Vulkan ICD (the driver ships the Vulkan ICD; no CUDA needed).
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
fi

echo "== Installing Vulkan tooling (glslang, loader, tools) =="
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  git build-essential glslang-tools libvulkan-dev vulkan-tools

# On NVIDIA cloud images the Vulkan ICD is usually present with the driver.
# If vulkaninfo shows no device, install the vendor ICD package:
#   NVIDIA:  (ships with the driver — ensure libnvidia-gl-<ver> is installed)
#   AMD:     sudo apt-get install -y mesa-vulkan-drivers
#   Intel:   sudo apt-get install -y mesa-vulkan-drivers
echo "== Vulkan devices visible to the loader =="
vulkaninfo --summary 2>/dev/null | sed -n '1,40p' || \
  echo "(vulkaninfo unavailable; continuing — the executor will report the device)"

WORK="${AXIM_WORK:-$HOME/axim-bench}"
echo "== Cloning AXIM into $WORK =="
rm -rf "$WORK"
git clone --depth 1 https://github.com/GRRN-MAKER/Aximcomp.git "$WORK"
cd "$WORK/axim_compiler/backend_gpu"

echo "== Compiling SPIR-V shaders =="
mkdir -p shaders/spirv build
for f in shaders/vulkan/*.comp; do
  n=$(basename "$f" .comp)
  glslangValidator -V "$f" -o "shaders/spirv/$n.spv"
  echo "  built $n.spv"
done

echo "== Building Vulkan self-test + benchmark =="
c++ -std=c++17 -O3 -DAXIM_BUILD_VULKAN -Iinclude \
  src/axim_vulkan_executor.cpp test_vulkan_executor.cpp \
  -lvulkan -o build/axim_vk_selftest
c++ -std=c++17 -O3 -DAXIM_BUILD_VULKAN -Iinclude \
  src/axim_vulkan_executor.cpp bench_vulkan.cpp \
  -lvulkan -o build/axim_vk_bench

echo
echo "############### CORRECTNESS (self-test) ###############"
./build/axim_vk_selftest

echo
echo "############### BENCHMARK (real GPU timing) ###############"
./build/axim_vk_bench

echo
echo "== DONE. Copy the device name + numbers above back to the AXIM paper. =="
