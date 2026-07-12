cmake_minimum_required(VERSION 3.10)
project(AximFoundation)

set(CMAKE_CXX_STANDARD 14)

# Placeholder for CMake configuration if built standalone
# Currently configured to be compiled by cxx via build.rs in Rust.

# To build standalone:
add_library(axim_foundation STATIC
    src/vision.cpp
    src/kv_cache.cpp
    src/edge_rail_thermal.cpp
    src/ghost_touch.cpp
)
target_include_directories(axim_foundation PUBLIC include)
