"""Quick test: run AXIM kernels on the real Apple GPU via Metal."""
import ctypes, os

_here = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(_here, "build", "libaxim_metal.dylib"))

lib.axim_metal_device_name.restype = ctypes.c_char_p
lib.axim_metal_init.restype = ctypes.c_int
lib.axim_metal_add.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_size_t]

rc = lib.axim_metal_init()
print(f"Metal init rc = {rc}")
print(f"GPU device    = {lib.axim_metal_device_name().decode()}")

n = 8
a = (ctypes.c_float * n)(*[1.0 * i for i in range(n)])
b = (ctypes.c_float * n)(*[10.0 * i for i in range(n)])
out = (ctypes.c_float * n)()
fp = lambda x: ctypes.cast(x, ctypes.POINTER(ctypes.c_float))
lib.axim_metal_add(fp(a), fp(b), fp(out), n)
result = list(out)
print(f"GPU add result = {result}")
expected = [11.0 * i for i in range(n)]
assert result == expected, f"mismatch: {result} != {expected}"
print("✅ AXIM ran on the real Apple GPU via Metal — CUDA-free.")
