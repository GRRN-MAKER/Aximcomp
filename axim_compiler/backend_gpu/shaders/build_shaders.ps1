# AXIM Shader Build (Windows / PowerShell)
#   Vulkan: GLSL .comp -> SPIR-V .spv  (glslangValidator, from the Vulkan SDK)
# Zero CUDA. Run from this directory:  ./build_shaders.ps1
#
# On Windows, AXIM's GPU path is Vulkan (NVIDIA/AMD/Intel drivers all ship it).
# A DirectX 12 backend can consume the same SPIR-V via DXC/SPIRV-Cross; the
# Vulkan path is the default and requires no extra tooling beyond the SDK.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "== AXIM Shader Build (Windows) =="

if (Get-Command glslangValidator -ErrorAction SilentlyContinue) {
    New-Item -ItemType Directory -Force -Path spirv | Out-Null
    Get-ChildItem vulkan/*.comp | ForEach-Object {
        $name = $_.BaseName
        Write-Host "  [Vulkan] $($_.Name) -> spirv/$name.spv"
        glslangValidator -V $_.FullName -o "spirv/$name.spv"
    }
} else {
    Write-Host "  [Vulkan] glslangValidator not found - install the Vulkan SDK (LunarG)."
}

Write-Host "== Done =="
