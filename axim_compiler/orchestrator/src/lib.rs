//! AXIM Orchestrator — Device Dispatch Layer
//! ==========================================
//! Routes AXIM IR operations to the correct vendor-neutral backend:
//!   - CPU: AVX-512 / AVX2 / NEON SIMD  (backend_cpu)
//!   - GPU: Vulkan Compute / Metal      (backend_gpu)
//!
//! Zero CUDA. The orchestrator picks the best available device, falls
//! back from GPU to CPU automatically, and exposes a stable C ABI so the
//! Python frontend (and SYNAXIM) can drive it on any machine.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

// ── FFI to the C++ CPU backend (backend_cpu) ──
extern "C" {
    fn axim_cpu_backend_name() -> *const c_char;
    fn axim_cpu_add(a: *const f32, b: *const f32, out: *mut f32, n: usize);
    fn axim_cpu_mul(a: *const f32, b: *const f32, out: *mut f32, n: usize);
    fn axim_cpu_hello(out: *mut c_char, cap: usize) -> c_int;
}

// ── FFI to the C++ GPU backend (backend_gpu) ──
extern "C" {
    fn axim_gpu_api() -> c_int;
    fn axim_gpu_backend_name() -> *const c_char;
    fn axim_gpu_init() -> c_int;
    fn axim_gpu_add(a: *const f32, b: *const f32, out: *mut f32, n: usize) -> c_int;
    fn axim_gpu_hello(out: *mut c_char, cap: usize) -> c_int;
}

/// Target device kinds AXIM can dispatch to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    Auto,
}

/// Returns the active CPU SIMD backend name (e.g. "AVX-512", "NEON").
pub fn cpu_backend() -> String {
    unsafe {
        CStr::from_ptr(axim_cpu_backend_name())
            .to_string_lossy()
            .into_owned()
    }
}

/// Returns the active GPU backend name (e.g. "Vulkan", "Metal", "none").
pub fn gpu_backend() -> String {
    unsafe {
        CStr::from_ptr(axim_gpu_backend_name())
            .to_string_lossy()
            .into_owned()
    }
}

/// True if a real (non-emulated) GPU backend is available.
pub fn gpu_available() -> bool {
    unsafe { axim_gpu_api() != 0 }
}

/// Resolve `Auto` to a concrete device: GPU if available, else CPU.
fn resolve(kind: DeviceKind) -> DeviceKind {
    match kind {
        DeviceKind::Auto => {
            if gpu_available() {
                DeviceKind::Gpu
            } else {
                DeviceKind::Cpu
            }
        }
        other => other,
    }
}

/// Elementwise add, dispatched to the chosen device with CPU fallback.
pub fn dispatch_add(a: &[f32], b: &[f32], out: &mut [f32], kind: DeviceKind) {
    let n = a.len();
    match resolve(kind) {
        DeviceKind::Gpu => unsafe {
            let _ = axim_gpu_init();
            let rc = axim_gpu_add(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n);
            if rc != 0 {
                // Fallback to CPU on any GPU failure.
                axim_cpu_add(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n);
            }
        },
        _ => unsafe {
            axim_cpu_add(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n);
        },
    }
}

/// Elementwise mul (CPU path; GPU shader added as backend matures).
pub fn dispatch_mul(a: &[f32], b: &[f32], out: &mut [f32], _kind: DeviceKind) {
    unsafe { axim_cpu_mul(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len()) }
}

/// Print a full AXIM device banner (used by the hello-world scaffold).
pub fn hello() -> String {
    let mut cpu_buf = vec![0i8; 256];
    let mut gpu_buf = vec![0i8; 256];
    unsafe {
        axim_cpu_hello(cpu_buf.as_mut_ptr(), cpu_buf.len());
        axim_gpu_hello(gpu_buf.as_mut_ptr(), gpu_buf.len());
        let cpu = CStr::from_ptr(cpu_buf.as_ptr()).to_string_lossy();
        let gpu = CStr::from_ptr(gpu_buf.as_ptr()).to_string_lossy();
        format!(
            "AXIM Orchestrator ready.\n  {}\n  {}\n  GPU available: {}",
            cpu, gpu, gpu_available()
        )
    }
}

// ══════════════════════════════════════════════════════════════
// C ABI — exported for the Python frontend / SYNAXIM to call.
// ══════════════════════════════════════════════════════════════

/// C-callable add. `kind`: 0=cpu, 1=gpu, 2=auto.
#[no_mangle]
pub extern "C" fn axim_dispatch_add(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    n: usize,
    kind: c_int,
) {
    let a = unsafe { std::slice::from_raw_parts(a, n) };
    let b = unsafe { std::slice::from_raw_parts(b, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, n) };
    let k = match kind {
        0 => DeviceKind::Cpu,
        1 => DeviceKind::Gpu,
        _ => DeviceKind::Auto,
    };
    dispatch_add(a, b, out, k);
}

/// C-callable banner into a caller-provided buffer.
#[no_mangle]
pub extern "C" fn axim_orchestrator_hello(out: *mut c_char, cap: usize) -> c_int {
    let banner = hello();
    let bytes = banner.as_bytes();
    let n = bytes.len().min(cap.saturating_sub(1));
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, out, n);
        *out.add(n) = 0;
    }
    n as c_int
}

// ══════════════════════════════════════════════════════════════
// PyO3 module — `import axim._native` (built with maturin, feature "python")
// ══════════════════════════════════════════════════════════════
#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;

    /// Elementwise add on the chosen device: device 0=cpu, 1=gpu, 2=auto.
    #[pyfunction]
    fn add(a: Vec<f32>, b: Vec<f32>, device: i32) -> Vec<f32> {
        let mut out = vec![0.0f32; a.len()];
        let k = match device {
            0 => DeviceKind::Cpu,
            1 => DeviceKind::Gpu,
            _ => DeviceKind::Auto,
        };
        dispatch_add(&a, &b, &mut out, k);
        out
    }

    /// CPU SIMD backend name (e.g. "AVX-512", "NEON").
    #[pyfunction]
    fn cpu_backend_name() -> String {
        cpu_backend()
    }

    /// GPU backend name (e.g. "Vulkan", "Metal", "none").
    #[pyfunction]
    fn gpu_backend_name() -> String {
        gpu_backend()
    }

    /// True if a real GPU backend is available.
    #[pyfunction]
    fn has_gpu() -> bool {
        gpu_available()
    }

    /// Full AXIM device banner.
    #[pyfunction]
    fn banner() -> String {
        hello()
    }

    #[pymodule]
    fn _native(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(add, m)?)?;
        m.add_function(wrap_pyfunction!(cpu_backend_name, m)?)?;
        m.add_function(wrap_pyfunction!(gpu_backend_name, m)?)?;
        m.add_function(wrap_pyfunction!(has_gpu, m)?)?;
        m.add_function(wrap_pyfunction!(banner, m)?)?;
        Ok(())
    }
}
