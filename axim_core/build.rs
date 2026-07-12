fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("../axim_foundation/src/vision.cpp")
        .file("../axim_foundation/src/kv_cache.cpp")
        .file("../axim_foundation/src/edge_rail_thermal.cpp")
        .file("../axim_foundation/src/ghost_touch.cpp")
        .include("../axim_foundation/include")
        .flag_if_supported("-std=c++14")
        .compile("axim_foundation");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/display_manager.rs");
    println!("cargo:rerun-if-changed=src/thunderbolt_handshake.rs");
    println!("cargo:rerun-if-changed=../axim_foundation/src/vision.cpp");
    println!("cargo:rerun-if-changed=../axim_foundation/include/vision.h");
    println!("cargo:rerun-if-changed=../axim_foundation/src/kv_cache.cpp");
    println!("cargo:rerun-if-changed=../axim_foundation/include/kv_cache.h");
    println!("cargo:rerun-if-changed=../axim_foundation/src/edge_rail_thermal.cpp");
    println!("cargo:rerun-if-changed=../axim_foundation/include/edge_rail_thermal.h");
    println!("cargo:rerun-if-changed=../axim_foundation/src/ghost_touch.cpp");
    println!("cargo:rerun-if-changed=../axim_foundation/include/ghost_touch.h");
}
