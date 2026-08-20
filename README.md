# Batch linear assignment for PyTorch

[![PyPI version](https://badge.fury.io/py/torch-linear-assignment.svg)](https://badge.fury.io/py/torch-linear-assignment) [![Build Status](https://github.com/ivan-chai/torch-linear-assignment/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/ivan-chai/torch-linear-assignment/actions) [![Downloads](https://img.shields.io/pypi/dm/torch-linear-assignment)](https://pepy.tech/project/torch-linear-assignment) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<h4 align="left">
    <p>
        <a href="#install">Installation</a> |
        <a href="#backend-and-release-policy">Backend policy</a> |
        <a href="#example">Usage</a> |
        <a href="#support">Support</a> |
        <a href="#legacy-cuda-implementation-00x">Legacy CUDA</a> |
        <a href="#acknowledgments">Acknowledgments</a> |
        <a href="#citation">Citation</a>
    <p>
</h4>
Batch computation of the linear assignment problem with PyTorch. The active
`0.1.0+` release line is pure source: CPU calls use SciPy, and supported CUDA
calls use the lazy Triton backend.

## Backend and release policy

The active release line is `0.1.0+`. It uses Triton on validated Linux NVIDIA GPUs with compute capability 8.0 or newer and SciPy everywhere else. Unsupported CUDA inputs are solved by the SciPy CPU fallback and returned to the input device. The compiled CUDA extension belongs to the maintenance-only `0.0.x` line and is documented separately in [Legacy CUDA implementation (`0.0.x`)](#legacy-cuda-implementation-00x). If that backend needs a compatibility or correctness fix, the project can issue another `0.0.x` release without adopting the Triton overhaul.

## Install

Python 3.10 or newer is required. Install the `0.1.0` release candidate from PyPI:

```bash
python -m pip install "torch-linear-assignment==0.1.0rc0"
```

No editable install, local CUDA compilation, or `--no-build-isolation` flag is needed for normal use. On Linux, the package declares Triton through a platform marker and imports it lazily only for an eligible CUDA input. macOS and Windows installs remain usable through SciPy without requiring Triton.

## Example

```python
import torch
from torch_linear_assignment import batch_linear_assignment

cost = torch.tensor([
    8, 4, 7,
    5, 2, 3,
    9, 6, 7,
    9, 4, 8,
]).reshape(1, 4, 3)

assignment = batch_linear_assignment(cost)
print(assignment)
```

The output is:

```py
tensor([[ 0,  2, -1,  1]])
```

To get indices in the SciPy's format:

```py
from torch_linear_assignment import assignment_to_indices

row_ind, col_ind = assignment_to_indices(assignment)
print(row_ind)
print(col_ind)
```

The output is:

```py
tensor([[0, 1, 3]])
tensor([[0, 2, 1]])
```

## Support

The public API is `batch_linear_assignment(cost)` with cost shape `(B, W, T)`. Assignments have shape `(B, W)`, use `torch.long`, and contain `-1` for an unmatched worker.

| Input and runtime                                                                                                                                         | Backend and behavior                                                               |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| CPU input on any supported OS                                                                                                                             | SciPy on CPU. This remains the reference path.                                     |
| Linux NVIDIA GPU with compute capability 8.0 or newer, Torch >= 2.4, and Triton available                                                                 | Lazy Triton CUDA backend. The first call can include Triton compilation.           |
| CUDA input on an unsupported OS; non-NVIDIA Linux accelerator; NVIDIA GPU below compute capability 8.0 (including T4); or missing/ineligible Triton/Torch | Warn once, solve with SciPy on CPU, and return the assignment on the input device. |

The solver promotes `float16`, `bfloat16`, and integer costs to `float32`; `float64` costs remain `float64`. Triton execution disables autocast, so precision already lost before entry cannot be recovered.

Non-finite costs follow the SciPy contract. `NaN` and `-inf` raise `ValueError("matrix contains invalid numeric entries")`. `+inf` represents a forbidden edge when a perfect matching still exists; if the matrix is infeasible, the call raises `ValueError("cost matrix is infeasible")` instead of terminating the process with a device assertion.

## Validation and performance status

GPU validation is author-run rather than hosted CI. The validation targets are Linux NVIDIA GPUs with compute capability 8.0 or newer (for example, L4, A100, H100, or RTX PRO 6000 Blackwell):

```bash
make validate   # correctness once; prints GPU metadata when available
make benchmark  # SciPy CPU and installed public CUDA; writes JSONL and a cold/warm table
```

`make validate` prints Python/platform, Torch/CUDA, Triton, and GPU compute capability metadata when a CUDA device is visible. `make benchmark` separately measures the SciPy CPU reference and installed public CUDA backend, writes complete JSONL evidence, and prints a compact cold/warm table. A clean GPU metadata skip on macOS or a machine without CUDA is expected, but is not GPU evidence.

The first Triton call may be slower because it includes cold compilation; measure cold compilation separately from warm execution. Do not generalize performance across GPU models or workloads. The active `0.1.0+` line requires the correctness, AMP/dtype, fallback, packaging, GPU, and warm-performance gates to pass on each advertised GPU class.

Author-run evidence captured on 2026-08-20 at commit `eee95ffc` (`perf(triton): terminate completed augmentations early`) supports this matrix:

| GPU                                      | SciPy reference comparison      | Triton validation  |
| ---------------------------------------- | ------------------------------- | ------------------ |
| NVIDIA L4 (`sm_89`)                      | Exact in all 30 benchmark cases | `78/78` tests pass |
| NVIDIA A100 (`sm_80`)                    | Exact in all 30 benchmark cases | `78/78` tests pass |
| NVIDIA H100 (`sm_90`)                    | ?                               | ?                  |
| NVIDIA RTX PRO 6000 Blackwell (`sm_120`) | Exact in all 30 benchmark cases | `78/78` tests pass |

Representative batch-208 float32 performance is shown below. Each timing cell is `transpose / square / direct` in milliseconds for `300 x 100`, `300 x 300`, and `300 x 600`. Ratios above `1.0x` mean Triton is faster. Values are rounded to one decimal; `~` marks ratios derived from the displayed timings.

| GPU / timing       | SciPy CPU (ms)       | Triton (ms)      | SciPy / Triton          |
| ------------------ | -------------------- | ---------------- | ----------------------- |
| L4 (first\*)       | 71.8 / 788.6 / 421.9 | 1.0 / 7.6 / 4.3  | ~71.8x / 103.8x / 98.1x |
| L4 (warm)          | 71.3 / 782.5 / 422.8 | 1.0 / 7.6 / 4.4  | 71.3x / 103.5x / 96.4x  |
| A100 (first\*)     | 74.8 / 788.6 / 430.3 | 1.3 / 11.6 / 4.3 | ~57.5x / 68.0x / 100.1x |
| A100 (warm)        | 73.9 / 788.4 / 431.9 | 1.3 / 11.6 / 4.4 | 56.7x / 67.7x / 99.1x   |
| H100 (first\*)     | ?                    | ?                | ?                       |
| H100 (warm)        | ?                    | ?                | ?                       |
| RTX 6000 (first\*) | 32.1 / 409.8 / 204.8 | 0.7 / 5.5 / 2.2  | ~45.9x / 74.5x / 93.1x  |
| RTX 6000 (warm)    | 32.3 / 410.1 / 204.8 | 0.5 / 5.4 / 2.2  | 60.8x / 75.8x / 94.4x   |

The Triton implementation parallelizes batch items and vectorizes each current candidate-column scan. Commit `eee95ffc` also terminates the shortest-path search when an augmentation finds an unmatched sink. This removes the post-completion iterations that dominated direct matrices. Against the same-machine SciPy CPU reference, representative warm Triton speedups span `56.7x`--`103.5x` while retaining exact assignment parity.

`first*` means the first call for that case in the shared benchmark process. It is not an isolated compilation measurement: earlier shapes and dtypes can populate driver, process, and Triton disk caches. The three notebooks also predate the fresh-process benchmark-integrity harness, so these measurements are provisional author-run evidence rather than the final cold-performance record. Publication-grade cold claims require a source-bound, five-process rerun.

Do not publish a Triton-only timing as a speedup. Each performance report pairs the same seeded workload with its same-machine SciPy CPU baseline.

For direct benchmark-script options, the underlying command writes complete JSONL evidence to a run-specific file and prints a compact table containing the backend, validation mode, shape, status, cold/warm latency, and exact-parity result:

```bash
make benchmark
python tests/benchmark.py --backends scipy,public_cuda --output benchmark-results.jsonl
```

The command exits nonzero if any invocation fails exact parity. The JSONL file retains replay diagnostics that the console table intentionally omits.

## Legacy CUDA implementation (`0.0.x`)

Release `0.0.6` contains the frozen, compiled plain-CUDA implementation. It is not the active default and should be pinned only when that legacy GPU path is explicitly required, including T4 (`sm_75`) execution:

```bash
python -m pip install "torch-linear-assignment==0.0.6"
```

The extension needs a compatible CUDA build environment and must include `sm_75`; the project does not promise a portable T4 wheel. Pinning it is also workload-dependent: in the recorded T4 comparison, SciPy was faster through batch 50, while legacy CUDA was materially faster for the tested batch-208 transpose and direct cases.

For a private contributor comparison from a current checkout, the legacy extension can instead be built explicitly:

```bash
TLA_BUILD_LEGACY_CUDA=1 python -m pip install -e . --no-build-isolation
```

This opt-in is not required for normal `0.1.0+` use and does not expose a public backend selector. For a clean cross-version benchmark, copy `tests/benchmark.py` outside the checkout first. In a separate baseline environment, `make install-legacy` installs the newest PyPI release matching `torch-linear-assignment<0.1.0`; the copied runner then measures its installed public CUDA backend. Pair those results with the same shapes, dtypes, seed, GPU, and validation mode from the current Triton run.

The following author-run batch-208 FP32 comparison uses the same `transpose / square / direct` ordering as the main table. Ratios above `1.0x` mean Triton is faster; values are rounded to one decimal, and `~` marks ratios derived from separately displayed legacy and current-package timings.

| GPU / timing       | Legacy 0.0.6 (ms)    | Triton (ms)      | Legacy / Triton          |
| ------------------ | -------------------- | ---------------- | ------------------------ |
| T4 (first\*)       | 37.0 / 879.5 / 273.9 | —                | —                        |
| T4 (warm)          | 37.7 / 738.6 / 272.6 | —                | —                        |
| L4 (first\*)       | 37.9 / 648.5 / 274.5 | 1.0 / 7.6 / 4.3  | ~37.9x / 85.3x / 63.8x   |
| L4 (warm)          | 37.8 / 652.5 / 273.3 | 1.0 / 7.6 / 4.4  | ~37.8x / 85.9x / 62.1x   |
| A100 (first\*)     | 36.2 / 755.5 / 271.7 | 1.3 / 11.6 / 4.3 | ~27.8x / 65.1x / 63.2x   |
| A100 (warm)        | 36.2 / 731.3 / 271.6 | 1.3 / 11.6 / 4.4 | ~27.8x / 63.0x / 61.7x   |
| H100 (first\*)     | ?                    | ?                | ?                        |
| H100 (warm)        | ?                    | ?                | ?                        |
| RTX 6000 (first\*) | 35.4 / 607.1 / 248.0 | 0.7 / 5.5 / 2.2  | ~50.6x / 110.4x / 112.7x |
| RTX 6000 (warm)    | 35.4 / 607.6 / 249.3 | 0.5 / 5.4 / 2.2  | ~70.8x / 112.5x / 113.3x |

Triton is faster in all 30 paired warm cases on each measured supported GPU; the weakest displayed ratio is about `6.0x`. T4 has no Triton timing because it is below the supported compute capability, and H100 remains unmeasured. The `first*` caveat from the main table applies equally here.

## Acknowledgments

The `0.1.0+` pure-source Triton backend, GPU validation workflow, and benchmark revamp were implemented by [@Borda](https://github.com/Borda).

## Citation

The code was originally developed for the [HoTPP Benchmark](https://github.com/ivan-chai/hotpp-benchmark). If you use this code in your research project, please cite one of the following papers:

```
@article{karpukhin2024hotppbenchmark,
  title={HoTPP Benchmark: Are We Good at the Long Horizon Events Forecasting?},
  author={Karpukhin, Ivan and Shipilov, Foma and Savchenko, Andrey},
  journal={arXiv preprint arXiv:2406.14341},
  year={2024},
  url ={https://arxiv.org/abs/2406.14341}
}

@article{karpukhin2024detpp,
  title={DeTPP: Leveraging Object Detection for Robust Long-Horizon Event Prediction},
  author={Karpukhin, Ivan and Savchenko, Andrey},
  journal={arXiv preprint arXiv:2408.13131},
  year={2024},
  url ={https://arxiv.org/abs/2408.13131}
}
```
