"""Host eligibility for tests that must run the compiled Triton kernel, shared by both suites.

On an unsupported platform the public API falls back to SciPy. That is useful behavior but not
evidence that a Triton kernel compiled or executed, so kernel tests skip with the exact reason.
"""

import importlib
import re
import sys

import pytest
import torch

MINIMUM_TORCH_VERSION = (2, 4)


def compiled_triton_skip_reason() -> str | None:
    """Return why compiled-kernel tests cannot honestly run on this host."""
    if sys.platform != "linux":
        return "compiled Triton acceptance requires Linux; macOS fallback is not kernel evidence"
    if not torch.cuda.is_available():
        return "compiled Triton acceptance requires an available CUDA device"
    if torch.version.cuda is None:
        return "compiled Triton acceptance requires a CUDA-enabled PyTorch build"
    version = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if version is None or tuple(map(int, version.groups())) < MINIMUM_TORCH_VERSION:
        return "compiled Triton acceptance requires PyTorch 2.4 or newer"
    try:
        importlib.import_module("triton")
    except ModuleNotFoundError as error:
        if error.name == "triton":
            return "compiled Triton acceptance requires the optional triton package"
        raise
    return None


def compiled_device_params() -> "list[pytest.ParameterSet]":
    """Parametrize every supported CUDA device or report the exact ineligibility."""
    reason = compiled_triton_skip_reason()
    if reason is not None:
        return [pytest.param(None, marks=pytest.mark.skip(reason=reason), id="ineligible")]

    params: list[pytest.ParameterSet] = []
    for device_index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(device_index)
        device = torch.device(f"cuda:{device_index}")
        if capability < (8, 0):
            params.append(
                pytest.param(
                    device,
                    marks=pytest.mark.skip(
                        reason=(
                            "compiled Triton acceptance requires NVIDIA compute "
                            f"capability >= 8.0; cuda:{device_index} is {capability}"
                        )
                    ),
                    id=f"cuda-{device_index}-sm-{capability[0]}{capability[1]}",
                )
            )
            continue
        params.append(
            pytest.param(
                device,
                id=f"cuda-{device_index}-sm-{capability[0]}{capability[1]}",
            )
        )
    return params
