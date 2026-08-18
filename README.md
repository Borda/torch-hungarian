# Batch linear assignment for PyTorch

[![PyPI version](https://badge.fury.io/py/torch-linear-assignment.svg)](https://badge.fury.io/py/torch-linear-assignment)
[![Build Status](https://github.com/ivan-chai/torch-linear-assignment/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/ivan-chai/torch-linear-assignment/actions)
[![Downloads](https://img.shields.io/pypi/dm/torch-linear-assignment)](https://pepy.tech/project/torch-linear-assignment)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<h4 align="left">
    <p>
        <a href="#Install">Installation</a> |
        <a href="#Example">Usage</a> |
        <a href="#Support">Support</a> |
        <a href="#Citation">Citation</a>
    <p>
</h4>
Batch computation of the linear assignment problem with PyTorch. The Issue 32
Triton rewrite is currently staged, not the final backend cutover: CPU calls
remain SciPy-backed, and supported CUDA calls use the lazy, pure-source Triton
path.

## Install

Python 3.10 or newer is required. Install the pure-source package from PyPI:

```bash
python -m pip install torch-linear-assignment
```

Install a checkout in editable mode:

```bash
python -m pip install -e . --no-build-isolation
```

The default install does not compile the legacy CUDA extension. On Linux,
installation declares Triton through a platform marker, and the package imports
it lazily only when a supported CUDA input is dispatched. macOS and Windows
installs therefore remain usable without Triton.

For a private comparison or rollback against the old CUDA implementation only,
opt in explicitly while installing from a checkout:

```bash
TLA_BUILD_LEGACY_CUDA=1 python -m pip install -e . --no-build-isolation
```

This is not required for normal use and does not expose a public backend
selector. The extension remains available for comparison until the GPU
correctness, performance, packaging, and upstream-review gates pass.

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

| Input and runtime                                                                                                                                  | Backend and behavior                                                                                                     |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| CPU input on any supported OS                                                                                                                      | SciPy on CPU. This remains the CPU reference path.                                                                       |
| Linux, NVIDIA GPU with compute capability 8.0 or newer, Torch >= 2.4, and Triton available                                                         | Lazy Triton CUDA backend. The staged patch supports Ampere-or-newer GPUs; the first call can include Triton compilation. |
| CUDA input on an unsupported OS; non-NVIDIA Linux accelerator; NVIDIA GPU below compute capability 8.0; or missing/ineligible Triton/Torch runtime | Warn once, solve with SciPy on CPU, and return the assignment on the input device.                                       |

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
make validate-gpu  # tests and benchmark; skips when no CUDA device is visible
make validate      # CPU gate, then the GPU lane
```

`validate-gpu` prints Python/platform, Torch/CUDA, Triton, and GPU compute
capability metadata. Include that output, the exact-parity test result, and the
benchmark output when reporting GPU evidence. A clean skip on macOS or a
machine without CUDA is expected, but is not GPU evidence.

The first Triton call may be slower because it includes cold compilation;
measure cold compilation separately from warm execution. This staged patch
makes no performance claim. The compiled extension is retained until the
correctness, AMP/dtype, fallback, packaging, GPU, and warm-performance gates
pass; it is not yet a final cutover.

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
