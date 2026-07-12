# Security Policy

AXIM is a vendor-neutral, CUDA-free compute and graphics runtime. Because it
executes low-level kernels on CPU (SIMD) and GPU (Vulkan/Metal), we take
security seriously and welcome responsible disclosure.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ Yes    |
| < 0.1   | ❌ No     |

## Reporting a Vulnerability

**Please report vulnerabilities privately — do not open a public issue.**

Use GitHub's **private vulnerability reporting**:

1. Go to the **Security** tab of this repository
2. Click **Report a vulnerability**
3. Provide a clear description, reproduction steps, and impact assessment

Alternatively, email the maintainer at the address listed on the GRRN
profile. We aim to acknowledge reports within **72 hours** and provide a
remediation timeline within **7 days**.

## What to Report

- Memory-safety issues in the C/C++/Objective-C++ backends
  (`backend_cpu`, `backend_gpu`, `graphics`, `hip`)
- Buffer overflows in the INT4 unpacking / matvec kernels
- Unsafe FFI boundaries in the Rust orchestrator
- Malicious `.symb` model files that could trigger out-of-bounds reads
- Shader injection via untrusted GLSL/MSL sources
- Supply-chain issues in dependencies (also tracked by Dependabot)

## Scope

**In scope:** all code under `axim_compiler/`, `axim_core/`,
`axim_foundation/`, and the GitHub Actions workflows.

**Out of scope:** third-party GPU drivers (Vulkan/Metal), the host OS,
and hardware vulnerabilities in specific GPUs.

## Security Hardening in AXIM

- **No CUDA / proprietary runtime** — reduces closed-source attack surface
- **Bounds-checked INT4 unpacking** — nibble extraction stays within buffers
- **CodeQL scanning** — automated on every push (Python + C/C++)
- **Dependabot** — weekly dependency vulnerability checks
- **Rust orchestrator** — memory-safe device dispatch layer

## Disclosure Policy

We follow **coordinated disclosure**. Once a fix is available, we will
publish a security advisory crediting the reporter (unless anonymity is
requested).
