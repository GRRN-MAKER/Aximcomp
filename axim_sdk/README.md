# Axim SDK

Python SDK for the Axim agent.

## Installation (development)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Optional: Rust core
If you build the Rust extension in `../axim-core` (using maturin or setuptools-rust), it will provide `axim_core.ShellExecutor` for faster shell execution. Until then, the SDK falls back to Python's `subprocess`.
