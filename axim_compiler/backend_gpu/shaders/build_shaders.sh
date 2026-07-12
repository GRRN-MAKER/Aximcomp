#!/usr/bin/env bash
# AXIM Shader Build — compiles all compute shaders to portable bytecode.
#   Vulkan:  GLSL .comp  → SPIR-V .spv   (glslangValidator)
#   Metal:   MSL .metal  → .metallib     (xcrun metal)
# Zero CUDA. Run from this directory.
set -e
cd "$(dirname "$0")"

echo "== AXIM Shader Build =="

# ── Vulkan SPIR-V ──
if command -v glslangValidator >/dev/null 2>&1; then
    mkdir -p spirv
    for f in vulkan/*.comp; do
        name=$(basename "$f" .comp)
        echo "  [Vulkan] $name.comp -> spirv/$name.spv"
        glslangValidator -V "$f" -o "spirv/$name.spv"
    done
else
    echo "  [Vulkan] glslangValidator not found — install Vulkan SDK to build SPIR-V."
fi

# ── Metal metallib ──
if xcrun -sdk macosx metal --version >/dev/null 2>&1; then
    echo "  [Metal] axim_kernels.metal -> metal/axim_kernels.metallib"
    xcrun -sdk macosx metal   -c metal/axim_kernels.metal -o metal/axim_kernels.air
    xcrun -sdk macosx metallib   metal/axim_kernels.air   -o metal/axim_kernels.metallib
else
    echo "  [Metal] metal toolchain not found — install full Xcode to build .metallib."
fi

echo "== Done =="
