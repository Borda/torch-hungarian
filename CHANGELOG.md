# Changelog

## 0.1.0rc0 - 2026-08-20

### Added

- Add a lazy Triton CUDA backend for Linux NVIDIA GPUs with compute capability 8.0 or newer and Torch 2.4 or newer.
- Add exact SciPy parity, dtype, autocast, non-finite input, non-default stream, and large-batch regression coverage for the Triton solver.
- Add reproducible author-run GPU validation and JSONL benchmark reporting with paired SciPy and legacy `0.0.6` comparisons.

### Changed

- Ship a pure-source package by default; normal installation no longer compiles the legacy CUDA extension.
- Require Python 3.10 or newer.
- Preserve `float64` solver precision, promote `float16`, `bfloat16`, and integer inputs to `float32`, and match SciPy's infeasible and invalid-cost behavior.
- Fall back to SciPy on unsupported CUDA runtimes and return the assignment to the input device.

### Migration

- Existing `batch_linear_assignment` and `assignment_to_indices` calls do not require API changes.
- Users who require the compiled CUDA implementation, including execution on T4 (`sm_75`), should remain on the maintenance-only `0.0.x` release line. Release `0.0.6` can be installed with `python -m pip install "torch-linear-assignment==0.0.6"`.
- The Triton path requires Linux, an NVIDIA GPU with compute capability 8.0 or newer, Torch 2.4 or newer, and Triton. Other supported environments continue through SciPy.

### Validation status

- Exact assignment parity and all GPU tests pass on NVIDIA L4 (`sm_89`), A100 (`sm_80`), and RTX PRO 6000 Blackwell (`sm_120`) author-run environments.
- H100 validation and isolated fresh-process cold-start measurements remain pending. Published warm timings are paired with same-machine SciPy and legacy baselines; they are not generalized across hardware.
