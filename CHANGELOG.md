# Changelog

## 0.1.0rc1 - 2026-09-08

### Changed

- Continue the project as the [Borda/torch-hungarian](https://github.com/Borda/torch-hungarian) fork, published on PyPI as `torch-hungarian`. Ivan Karpukhin remains the author of the original project and of the `0.0.x` compiled CUDA line; Jirka Borovec is the maintainer of this fork.
- Keep the import name `torch_linear_assignment`, so code written against the upstream project and against `torch-hungarian==0.1.0rc0` needs no change.
- Point the package metadata, README badges, install command, and Colab reproduction cells at the fork, and add source, changelog, and upstream project URLs.
- Resolve the installed version from `torch-hungarian` first and from `torch-linear-assignment` second in `tests/benchmark.py` and `make benchmark`, so the fork and the frozen `0.0.6` upstream baseline are both recognized.
- Uninstall `torch-hungarian` in `make install-legacy` before installing the `0.0.6` baseline. Both distributions provide `torch_linear_assignment`, so without this the target could report the fork's version while running the legacy backend.

### Fixed

- Restrict the `triton` runtime dependency to Linux `x86_64`, matching the marker PyTorch itself uses. Triton publishes no aarch64 wheel and no source distribution, so the previous Linux-wide marker made installation fail outright on ARM Linux instead of falling back to SciPy.
- Pin `triton` in the hosted CI snapshot. The CPU-only PyTorch wheel omits the `triton` dependency that the CUDA wheel carries, so `pip check` failed on every matrix job.
- Upload the CPU-only benchmark diagnostic only when the benchmark step actually ran, so a failing install no longer reports a second, misleading missing-artifact error.

## 0.1.0rc0 - 2026-08-20

### Added

- Add a lazy Triton CUDA backend for Linux NVIDIA GPUs with compute capability 8.0 or newer and Torch 2.4 or newer.
- Add exact SciPy parity, dtype, autocast, non-finite input, non-default stream, and large-batch regression coverage for the Triton solver.
- Add reproducible author-run GPU validation and JSONL benchmark reporting with paired SciPy and legacy `0.0.6` comparisons.
- Add a fresh-process benchmark harness: every case runs in its own interpreter with a unique empty Triton cache, retains raw cold and warm samples, and enforces exact SciPy parity on each measured call.
- Add self-authenticating JSONL evidence that binds every record to the benchmark Git revision and source digest, the imported package origin and digest, the executed backend module digest, the input digest, the CPU model, dependency versions, and available `nvidia-smi` state.
- Add `make benchmark-evidence` for paired `off`, `nonfinite_only`, `infeasibility_flag_only`, and `full` validation-overhead, memory, and isolated cold-compilation records across five process rounds.
- Add `make validate-gpu-multi`, a fail-closed two-GPU acceptance gate, and a compiled-backend regression test that keeps stream, current-device, and workspace state isolated across two CUDA devices.
- Add a pinned `requirements-ci.txt` hosted validation snapshot, a Python 3.10, 3.12, and 3.13 test matrix, a CPU-only benchmark diagnostic artifact, and a pre-commit gate in hosted CI.

### Changed

- Ship a pure-source package by default; normal installation no longer compiles the legacy CUDA extension.
- Require Python 3.10 or newer.
- Preserve `float64` solver precision, promote `float16`, `bfloat16`, and integer inputs to `float32`, and match SciPy's infeasible and invalid-cost behavior.
- Fall back to SciPy on unsupported CUDA runtimes and return the assignment to the input device.
- Make `make benchmark` fail closed by default: missing GPU or backend execution, out-of-memory, worker failure, package or implementation identity mismatch, incomplete process rounds, and parity failure all exit nonzero. `--allow-incomplete` is only for CPU-only diagnostics, and its rows remain ineligible for performance acceptance.
- Pin `make install-legacy` to the exact PyPI `0.0.6` baseline and fail if the checkout shadows that installed distribution.
- Split the publish workflow into separate build and publish jobs, pin action digests, publish through a protected environment with trusted publishing, and verify both the wheel and the source distribution against the release tag from outside the workspace.

### Migration

- Existing `batch_linear_assignment` and `assignment_to_indices` calls do not require API changes.
- Users who require the compiled CUDA implementation, including execution on T4 (`sm_75`), should remain on the maintenance-only `0.0.x` release line. Release `0.0.6` can be installed with `python -m pip install "torch-linear-assignment==0.0.6"`.
- The Triton path requires Linux, an NVIDIA GPU with compute capability 8.0 or newer, Torch 2.4 or newer, and Triton. Other supported environments continue through SciPy.

### Validation status

- Exact assignment parity and all GPU tests pass on NVIDIA L4 (`sm_89`), A100 (`sm_80`), and RTX PRO 6000 Blackwell (`sm_120`) author-run environments.
- H100 validation and isolated fresh-process cold-start measurements remain pending. Published warm timings are paired with same-machine SciPy and legacy baselines; they are not generalized across hardware.
- The published tables predate this benchmark harness and carry no JSONL provenance, so they are provisional author-run evidence. Publication-grade cold claims require a source-bound, five-process rerun on each advertised GPU class.
