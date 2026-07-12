cmake_minimum_required(VERSION 3.12)
project(axim_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Vendor-neutral SIMD flags. No CUDA, no nvcc.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    # Apple silicon / ARM — NEON is baseline on aarch64
    add_compile_options(-O3)
else()
    # x86_64 — enable AVX2; AVX-512 is selected at runtime via CPUID
    add_compile_options(-O3 -mavx2 -mfma)
endif()

add_library(axim_cpu SHARED
    src/axim_cpu.cpp
)
target_include_directories(axim_cpu PUBLIC include)

# Output a predictable name the Python ctypes loader expects.
set_target_properties(axim_cpu PROPERTIES
    OUTPUT_NAME "axim_cpu"
    PREFIX "lib"
)
