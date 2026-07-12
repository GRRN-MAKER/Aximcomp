# Getting Started

## Install / Run (no build required)

AXIM's Python frontend runs immediately in pure-Python fallback mode:

```python
import axim_compiler as axim

axim.hello()                       # shows CPU/GPU backends
axim.run(add, [1,2,3], [4,5,6])    # → [5.0, 7.0, 9.0]
```

## Build the native backends (for speed)

You do **not** need to build on your own PC — GitHub Actions builds every
backend in the cloud (see [Building with CI](ci.md)). To build locally:

=== "macOS (Apple silicon)"
    ```bash
    # CPU (NEON) + tuned libs
    cd axim_compiler/backend_cpu && mkdir -p build
    c++ -std=c++14 -O3 -fPIC -shared -Iinclude \
        src/axim_cpu.cpp src/axim_blas.cpp src/axim_dnn.cpp \
        -o build/libaxim_cpu.dylib

    # GPU (Metal) — live compute
    cd ../backend_gpu && mkdir -p build
    clang++ -std=c++17 -ObjC++ -O3 -fPIC -shared \
        -framework Metal -framework Foundation \
        -Iinclude src/axim_metal.mm -o build/libaxim_metal.dylib

    # Graphics (Metal) — game render
    cd ../graphics && mkdir -p build
    clang++ -std=c++17 -ObjC++ -O3 -fPIC -shared \
        -framework Metal -framework Foundation \
        -Iinclude src/axim_gfx_metal.mm -o build/libaxim_gfx.dylib
    ```

=== "Linux (Nvidia/AMD/Intel)"
    ```bash
    # CPU (AVX2) + tuned libs
    cd axim_compiler/backend_cpu && mkdir -p build
    c++ -std=c++14 -O3 -mavx2 -mfma -fPIC -shared -Iinclude \
        src/axim_cpu.cpp src/axim_blas.cpp src/axim_dnn.cpp \
        -o build/libaxim_cpu.so

    # GPU shaders (Vulkan SPIR-V)
    sudo apt-get install -y glslang-tools libvulkan-dev
    cd ../backend_gpu/shaders && ./build_shaders.sh
    ```

## Check your devices

```bash
python3 axim_compiler/tools/aximinfo.py
```

Output on an Apple M3:

```
Devices AXIM can target:
  [0] CPU  backend=neon   vendor=apple  name=arm
  [1] GPU  backend=metal  vendor=apple  name=Apple GPU (Metal)
Native backends:
  CPU  : libaxim_cpu  loaded  → SIMD=NEON
  GPU  : libaxim_metal loaded → device=Apple M3
CUDA          : NOT USED (AXIM is CUDA-free by design)
```

## Run the tests

```bash
python3 axim_compiler/tests/test_pipeline.py   # IR + dispatch
python3 axim_compiler/tests/test_synaxim.py    # SYNAXIM ops
python3 axim_compiler/tests/test_loader.py     # .symb loader
```

Next: [Architecture →](architecture.md)
