// AXIM Orchestrator build script.
// Compiles and links the vendor-neutral C++ CPU/GPU backends so the Rust
// orchestrator (and the PyO3 `axim._native` module) can call them via FFI.
// Zero CUDA — only SIMD (CPU) and Vulkan/Metal (GPU) sources are built.

use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();

    let cpu_src = root.join("backend_cpu/src/axim_cpu.cpp");
    let cpu_inc = root.join("backend_cpu/include");
    let gpu_src = root.join("backend_gpu/src/axim_gpu.cpp");
    let gpu_inc = root.join("backend_gpu/include");

    // ── CPU backend (SIMD) ──
    let mut cpu = cc::Build::new();
    cpu.cpp(true).std("c++14").opt_level(3)
        .include(&cpu_inc)
        .file(&cpu_src);
    if cfg!(target_arch = "x86_64") {
        cpu.flag_if_supported("-mavx2");
        cpu.flag_if_supported("-mfma");
    }
    cpu.compile("axim_cpu_static");

    // ── GPU backend (Vulkan/Metal ABI; emulated fallback if none) ──
    cc::Build::new()
        .cpp(true).std("c++14").opt_level(3)
        .include(&gpu_inc)
        .file(&gpu_src)
        .compile("axim_gpu_static");

    println!("cargo:rerun-if-changed={}", cpu_src.display());
    println!("cargo:rerun-if-changed={}", gpu_src.display());
}
