# Contribute to AXIM

We welcome and encourage contributions to both the AXIM code and documentation. If you
want to contribute to the AXIM repository, please first review the following guidance.

AXIM is a CUDA-free compute and graphics runtime made up of a Python frontend, a Rust
orchestrator, and C++/Objective-C++ backends (CPU SIMD, Metal, Vulkan). Some components
build on external projects (such as the Vulkan SDK and platform graphics APIs); those
follow their own upstream contribution guidelines. All AXIM components follow the
workflow described below.

---

## Development workflow

AXIM uses GitHub to host code, collaborate, and manage version control. We use pull
requests (PRs) for all changes within the repository, and GitHub issues to track known
issues such as bugs.

### Issue tracking

Before filing a new issue, search the
[existing issues](https://github.com/GRRN-MAKER/Aximcomp/issues) to make sure your issue
isn't already listed.

General issue guidelines:

- Use your best judgement for issue creation. If your issue is already listed, upvote the
  issue and comment to provide additional details, such as how you reproduced it.
- If you're not sure whether your issue is the same, err on the side of caution and file
  it. Add a comment linking the similar issue's number. If we evaluate it as a duplicate,
  we'll close it.
- If your issue doesn't exist, use the issue template to file a new one.
- When filing an issue, provide as much information as possible — including command
  output, your OS, CPU/GPU vendor, and driver versions — so we can reproduce it quickly.
- Check your issue regularly, as we may require additional information.

### Pull requests

When you create a pull request, target the default branch (`main`).

Use the following process (individual areas may add project-specific steps):

1. Identify the issue you want to fix.
2. Target the default branch (`main`) for integration.
3. Ensure your code builds successfully on at least one backend (see
   [Building from source](README.md#building-from-source)).
4. Run the relevant test suite and include the log of the successful run in your PR:
   ```bash
   python3 axim_compiler/tests/test_pipeline.py
   python3 axim_compiler/tests/test_synaxim.py
   python3 axim_compiler/tests/test_loader.py
   ```
5. Do not break existing test cases.
6. New functionality is only merged with new tests. If your PR includes a new feature,
   provide an example or test so we can ensure the feature works and stays valid.
7. Tests must have good coverage of the new code path.
8. Submit your PR and work with the reviewer/maintainer to get it approved.
9. All CI checks (Build, CodeQL) must pass before merge.
10. Once approved and green, a maintainer merges it. We'll inform you once your change is
    committed.

> **Important**
> By creating a PR, you agree to license your contribution under the terms of the
> [LICENSE](LICENSE) file (Apache License 2.0).

---

## Coding standards

- **Python** — target 3.9+, PEP 8, type hints where practical.
- **Rust** — `cargo fmt` and `cargo clippy` clean; edition 2021, Rust 1.70+.
- **C++ / Objective-C++** — C++17 (backends), no CUDA headers or CUDA dependencies of any
  kind. Keep vendor-neutral paths (Vulkan/SIMD) portable across x86 and ARM.
- Keep the CPU and GPU results **bit-for-bit consistent** where the op defines it (e.g.
  INT4 matvec CPU == GPU).

---

## New feature development

Use [GitHub Discussions](https://github.com/GRRN-MAKER/Aximcomp/discussions) to propose
new features (for example a new backend target, kernel, or graphics feature). Maintainers
are happy to provide direction and feedback before you invest in implementation.

---

## Documentation

AXIM documentation lives in [`docs/`](docs/) (MkDocs) and [`wiki/`](wiki/). You must
update the documentation related to any new feature or API contribution.

Preview the docs locally:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

---

## Security

Do **not** report security vulnerabilities through public issues or PRs. See our
[Security Policy](SECURITY.md) and use GitHub's private vulnerability reporting.

---

## Future development workflow

The current AXIM development workflow is GitHub-based. If we change platforms in the
future, the tools and links may change, and we will update this guide accordingly.
