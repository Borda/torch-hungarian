# Batch linear assignment for PyTorch

[![PyPI version](https://badge.fury.io/py/torch-linear-assignment.svg)](https://badge.fury.io/py/torch-linear-assignment)
[![Build Status](https://github.com/ivan-chai/torch-linear-assignment/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/ivan-chai/torch-linear-assignment/actions)
[![Downloads](https://img.shields.io/pypi/dm/torch-linear-assignment)](https://pepy.tech/project/torch-linear-assignment)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<h4 align="left">
    <p>
        <a href="#Install">Installation</a> |
        <a href="#backend-and-release-policy">Backend policy</a> |
        <a href="#Example">Usage</a> |
        <a href="#Support">Support</a> |
        <a href="#Citation">Citation</a>
    <p>
</h4>
Batch computation of the linear assignment problem with PyTorch. The Issue 32
Triton rewrite prepares the `0.1.0` backend cutover: CPU calls remain
SciPy-backed, and supported CUDA calls use the lazy, pure-source Triton path.

## Backend and release policy

The backend cutover is an explicit release boundary:

| Release line      | Primary GPU path               | NVIDIA T4 behavior                                                                   | Maintenance policy                                                                                         |
| ----------------- | ------------------------------ | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `0.0.6`           | Compiled legacy CUDA extension | Legacy CUDA can be built for `sm_75` when T4 GPU execution is specifically required. | Frozen legacy release. If a necessary legacy fix appears, branch from the `0.0.6` tag and release `0.0.7`. |
| `0.1.0` and later | Triton on validated GPUs       | T4 uses the SciPy CPU fallback and returns the assignment to the input device.       | Active release line; Triton is the high-performance backend.                                               |

Once `0.1.0` is released, T4 users should use the active `0.1.x` line unless
they explicitly need the legacy GPU backend. Pin `0.0.6` for that legacy path:

```bash
python -m pip install "torch-linear-assignment==0.0.6"
```

The `0.0.6` extension needs a compatible CUDA build environment and must
include `sm_75`; the project does not promise a portable T4 wheel. Pinning the
legacy release is also workload-dependent rather than a general performance
recommendation: in the recorded T4 comparison, SciPy was faster through batch
50, while legacy CUDA was materially faster for the tested batch-208 transpose
and direct cases.

The largest performance improvements are expected from Triton on validated
Ampere-or-newer GPUs. T4 remains usable on `0.1.0` and later through the SciPy
fallback, so newer hardware is a performance tier rather than a package
requirement.

## Install

Python 3.10 or newer is required. Install the pure-source package from PyPI:

```bash
python -m pip install torch-linear-assignment
```

Install a checkout in editable mode:

```bash
python -m pip install -e . --no-build-isolation
```

Starting with `0.1.0`, the default install does not compile the legacy CUDA
extension. On Linux, installation declares Triton through a platform marker,
and the package imports it lazily only when a supported CUDA input is
dispatched. macOS and Windows installs therefore remain usable without Triton.

For a private comparison during `0.1.0` development only, opt in explicitly
while installing from a checkout:

```bash
TLA_BUILD_LEGACY_CUDA=1 python -m pip install -e . --no-build-isolation
```

This is not required for normal use and does not expose a public backend
selector. End users who require the legacy T4 GPU path should use the frozen
`0.0.6` release instead.

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

The public API is `batch_linear_assignment(cost)` with cost shape `(B, W, T)`.
Assignments have shape `(B, W)`, use `torch.long`, and contain `-1` for an
unmatched worker.

| Input and runtime                                                                                                                                                  | Backend and behavior                                                                                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| CPU input on any supported OS                                                                                                                                      | SciPy on CPU. This remains the CPU reference path.                                                                          |
| `0.1.0+` on Linux, validated NVIDIA GPU with compute capability 8.0 or newer, Torch >= 2.4, and Triton available                                                   | Lazy Triton CUDA backend. The first call can include Triton compilation.                                                    |
| `0.1.0+` CUDA input on an unsupported OS; non-NVIDIA Linux accelerator; NVIDIA GPU below compute capability 8.0 (including T4); or missing/ineligible Triton/Torch | Warn once, solve with SciPy on CPU, and return the assignment on the input device.                                          |
| `0.0.6` on T4 with a compatible extension built for `sm_75`                                                                                                        | Frozen legacy CUDA backend. Use only when that GPU path is explicitly required; it is not the current release-line default. |

The solver promotes `float16`, `bfloat16`, and integer costs to `float32`;
`float64` costs remain `float64`. Triton execution disables autocast, so
precision already lost before entry cannot be recovered.

Non-finite costs follow the SciPy contract. `NaN` and `-inf` raise
`ValueError("matrix contains invalid numeric entries")`. `+inf` represents a
forbidden edge when a perfect matching still exists; if the matrix is
infeasible, the call raises `ValueError("cost matrix is infeasible")` instead
of terminating the process with a device assertion.

## Validation and performance status

GPU validation is author-run rather than hosted CI. The validation targets are
Linux NVIDIA GPUs with compute capability 8.0 or newer (for example, A100,
H100, or RTX 6000 Ada):

```bash
make validate   # correctness once; prints GPU metadata when available
make benchmark  # SciPy CPU and installed public CUDA; writes JSONL and a cold/warm table
```

`make validate` prints Python/platform, Torch/CUDA, Triton, and GPU compute
capability metadata when a CUDA device is visible. `make benchmark` separately
measures the SciPy CPU reference and installed public CUDA backend, writes
complete JSONL evidence, and prints a compact cold/warm table. A clean GPU
metadata skip on macOS or a machine without CUDA is expected, but is not GPU
evidence.

The first Triton call may be slower because it includes cold compilation;
measure cold compilation separately from warm execution. Do not generalize
performance across GPU models or workloads. The `0.1.0` cutover requires the
correctness, AMP/dtype, fallback, packaging, GPU, and warm-performance gates to
pass on each advertised GPU class. The development checkout retains the legacy
extension only for comparison; the public `0.1.0+` policy remains Triton on
validated GPUs and SciPy fallback elsewhere.

Author-run evidence captured on 2026-08-19 currently supports this matrix:

| GPU                                      | Backend under test  | Validation result                                                                                              |
| ---------------------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------- |
| NVIDIA T4 (`sm_75`)                      | `0.0.6` legacy CUDA | Legacy is exact in all 30 benchmark cases. Triton is unsupported; current public CUDA uses the SciPy fallback. |
| NVIDIA A100 (`sm_80`)                    | Triton              | `77/77` tests pass; exact in all 30 benchmark cases.                                                           |
| NVIDIA H100 (`sm_90`)                    | Triton              | ?                                                                                                              |
| NVIDIA L4 (`sm_89`)                      | Triton              | `77/77` tests pass; exact in all 30 benchmark cases.                                                           |
| NVIDIA RTX PRO 6000 Blackwell (`sm_120`) | Triton              | `77/77` tests pass; exact in all 30 benchmark cases.                                                           |

Representative batch-208 float32 performance is shown below. Each timing cell
is `transpose / square / direct` in milliseconds for `300 x 100`, `300 x 300`,
and `300 x 600`. Ratios above `1.0x` mean Triton is faster. Failed parity cases
have no publishable timing.

| GPU / timing    | SciPy CPU (ms)       | Legacy 0.0.6 (ms)    | Triton (ms)          | SciPy / Triton     | Legacy / Triton    |
| --------------- | -------------------- | -------------------- | -------------------- | ------------------ | ------------------ |
| T4 (cold)       | 69.7 / 730.4 / 398.6 | 37.0 / 879.5 / 273.9 | —                    | —                  | —                  |
| T4 (warm)       | 66.7 / 733.0 / 397.8 | 37.7 / 738.6 / 272.6 | —                    | —                  | —                  |
| L4 (cold)       | 74.0 / 782.8 / 426.4 | 37.4 / 635.3 / 270.2 | 35.5 / 119.8 / 352.1 | 2.1x / 6.5x / 1.2x | 1.1x / 5.3x / 0.8x |
| L4 (warm)       | 73.3 / 784.8 / 426.8 | 37.3 / 643.7 / 271.5 | 35.5 / 119.8 / 352.1 | 2.1x / 6.6x / 1.2x | 1.1x / 5.4x / 0.8x |
| A100 (cold)     | 72.5 / 792.9 / 432.2 | 36.3 / 766.6 / 273.4 | 51.2 / 197.3 / 620.4 | 1.4x / 4.0x / 0.7x | 0.7x / 3.9x / 0.4x |
| A100 (warm)     | 72.5 / 789.9 / 429.8 | 36.4 / 731.9 / 273.2 | 51.2 / 173.2 / 620.4 | 1.4x / 4.6x / 0.7x | 0.7x / 4.2x / 0.4x |
| H100 (cold)     | ?                    | ?                    | ?                    | ?                  | ?                  |
| H100 (warm)     | ?                    | ?                    | ?                    | ?                  | ?                  |
| RTX 6000 (cold) | 31.8 / 408.5 / 199.9 | 35.1 / 599.6 / 248.0 | 29.1 / 92.5 / 343.1  | 1.1x / 4.4x / 0.6x | 1.2x / 6.5x / 0.7x |
| RTX 6000 (warm) | 32.9 / 408.1 / 204.0 | 35.0 / 599.7 / 246.5 | 29.0 / 92.4 / 343.1  | 1.1x / 4.4x / 0.6x | 1.2x / 6.5x / 0.7x |

The Triton implementation parallelizes batch items and vectorizes each current
candidate-column scan. The row, shortest-path, and augmentation loops remain
sequential within each matrix. The corrected kernel passes parity, but the
performance gate remains open: square matrices are 4.2x--6.5x faster than
legacy in this representative warm workload, while direct matrices are only
0.4x--0.8x as fast.

Cold means the first call in the benchmark process. It is not isolated from
driver, process, or Triton disk caches populated by the preceding validation.

Do not publish a Triton-only timing as a speedup. Each performance report pairs
the same seeded workload with its same-machine SciPy CPU baseline. For a
cross-version report, additionally run a copied `tests/benchmark.py` against
the installed `torch-linear-assignment==0.0.6` public CUDA backend, then report
both `SciPy / Triton` and `0.0.6 / Triton` ratios for matching cold or warm
measurements.

In a separate baseline environment, `make install-legacy` installs the newest
PyPI release matching `torch-linear-assignment<0.1.0`; `make benchmark` then
measures that installed legacy public CUDA backend without reinstalling the
development checkout.

For direct benchmark-script options, the underlying command writes complete
JSONL evidence to a run-specific file and prints a compact table containing the
backend, validation mode, shape, status, cold/warm latency, and exact-parity
result:

```bash
make benchmark
python tests/benchmark.py --backends scipy,public_cuda --output benchmark-results.jsonl
```

The command exits nonzero if any invocation fails exact parity. The JSONL file
retains replay diagnostics that the console table intentionally omits.

# Citation

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
