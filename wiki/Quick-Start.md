# Quick Start

## Run without building

```python
import axim_compiler as axim
axim.hello()                          # shows CPU/GPU backends
print(axim.run(add, [1,2,3], [4,5,6]))  # → [5.0, 7.0, 9.0]
```

## Check your hardware

```bash
python3 axim_compiler/tools/aximinfo.py
```

## Build native backends (optional — CI can do this for you)

**macOS (Apple silicon):**
```bash
cd axim_compiler/backend_cpu && mkdir -p build
c++ -std=c++14 -O3 -fPIC -shared -Iinclude \
    src/axim_cpu.cpp src/axim_blas.cpp src/axim_dnn.cpp \
    -o build/libaxim_cpu.dylib

cd ../backend_gpu && mkdir -p build
clang++ -std=c++17 -ObjC++ -O3 -fPIC -shared \
    -framework Metal -framework Foundation \
    -Iinclude src/axim_metal.mm -o build/libaxim_metal.dylib
```

**Linux (Nvidia/AMD/Intel):**
```bash
cd axim_compiler/backend_cpu && mkdir -p build
c++ -std=c++14 -O3 -mavx2 -mfma -fPIC -shared -Iinclude \
    src/axim_cpu.cpp src/axim_blas.cpp src/axim_dnn.cpp \
    -o build/libaxim_cpu.so
```

## Run tests

```bash
python3 axim_compiler/tests/test_pipeline.py
python3 axim_compiler/tests/test_synaxim.py
python3 axim_compiler/tests/test_loader.py
```

See the full [documentation site](https://grrn-maker.github.io/Aximcomp/).
