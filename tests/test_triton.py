"""Compiled-GPU acceptance tests for the private Triton assignment backend.

These tests deliberately call the private wrapper rather than the public API:
on an unsupported platform the public API falls back to SciPy, which is useful
behavior but is not evidence that a Triton kernel compiled or executed.
"""

import importlib
import re
import sys

import pytest
import torch
from scipy.optimize import linear_sum_assignment

_MINIMUM_TORCH_VERSION = (2, 4)


def _compiled_triton_skip_reason() -> str | None:
    """Return why compiled-kernel tests cannot honestly run on this host."""
    if sys.platform != "linux":
        return "compiled Triton acceptance requires Linux; macOS fallback is not kernel evidence"
    if not torch.cuda.is_available():
        return "compiled Triton acceptance requires an available CUDA device"
    if torch.version.cuda is None:
        return "compiled Triton acceptance requires a CUDA-enabled PyTorch build"
    version = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if version is None or tuple(map(int, version.groups())) < _MINIMUM_TORCH_VERSION:
        return "compiled Triton acceptance requires PyTorch 2.4 or newer"
    try:
        importlib.import_module("triton")
    except ModuleNotFoundError as error:
        if error.name == "triton":
            return "compiled Triton acceptance requires the optional triton package"
        raise
    return None


def _compiled_device_params():
    """Parametrize every supported CUDA device or report the exact ineligibility."""
    reason = _compiled_triton_skip_reason()
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


@pytest.fixture(params=_compiled_device_params())
def triton_backend_device(
    request: pytest.FixtureRequest,
) -> tuple[object, torch.device]:
    """Provide the compiled private backend and one eligible CUDA device."""
    backend = importlib.import_module("torch_linear_assignment._triton")
    return backend, request.param


def _scipy_assignment(cost: torch.Tensor) -> torch.Tensor:
    """Return the public assignment oracle after Triton's documented promotion."""
    solver_dtype = torch.float64 if cost.dtype == torch.float64 else torch.float32
    cpu_cost = cost.detach().to(device="cpu", dtype=solver_dtype).contiguous()
    batch_size, workers, _ = cpu_cost.shape
    expected = torch.full((batch_size, workers), -1, dtype=torch.long)
    for batch_index, matrix in enumerate(cpu_cost):
        row_indices, column_indices = linear_sum_assignment(matrix.numpy())
        expected[batch_index, torch.from_numpy(row_indices)] = torch.from_numpy(column_indices)
    return expected


@pytest.mark.parametrize(
    "cost",
    [
        pytest.param(
            torch.tensor(
                [
                    [[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]],
                    [[8.0, 7.0, 2.0], [6.0, 4.0, 3.0], [5.0, 9.0, 1.0]],
                ]
            ),
            id="square-batch-isolated",
        ),
        pytest.param(
            torch.tensor([[[9.0, 2.0, 7.0, 8.0], [6.0, 4.0, 3.0, 7.0]]]),
            id="wide-workers-unassigned",
        ),
        pytest.param(
            torch.tensor([[[8.0, 4.0], [5.0, 2.0], [9.0, 6.0], [9.0, 4.0]]]),
            id="tall-transpose-path",
        ),
        pytest.param(torch.zeros((1, 3, 3)), id="exact-ties"),
        pytest.param(
            torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]),
            id="evolving-scan-order-ties",
        ),
        pytest.param(
            torch.tensor([[[-4.0, -1.0, -3.0], [-2.0, -5.0, -6.0], [-7.0, -8.0, -9.0]]]),
            id="negative-costs",
        ),
    ],
)
def test_batch_linear_assignment_compiled_matches_scipy_deterministic_costs(
    triton_backend_device: tuple[object, torch.device], cost: torch.Tensor
) -> None:
    """Prevent direct, transpose, tie, sign, and batch-index kernel regressions."""
    backend, device = triton_backend_device
    cuda_cost = cost.to(device)
    expected = _scipy_assignment(cuda_cost)

    actual = backend.batch_linear_assignment(cuda_cost)

    assert actual.device == device
    assert actual.dtype == torch.long
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        pytest.param((0, 3, 5), (0, 3), id="empty-batch"),
        pytest.param((1, 0, 3), (1, 0), id="zero-workers"),
        pytest.param((1, 3, 0), (1, 3), id="zero-tasks"),
        pytest.param((1, 0, 0), (1, 0), id="zero-workers-and-tasks"),
    ],
)
def test_batch_linear_assignment_compiled_preserves_empty_contract(
    triton_backend_device: tuple[object, torch.device],
    shape: tuple[int, int, int],
    expected: tuple[int, int],
) -> None:
    """Prevent zero-size calls from launching invalid kernels or losing sentinels."""
    backend, device = triton_backend_device
    cost = torch.empty(shape, device=device)

    actual = backend.batch_linear_assignment(cost)

    assert actual.shape == expected
    assert actual.dtype == torch.long
    if shape[2] == 0:
        assert torch.equal(actual.cpu(), torch.full(expected, -1, dtype=torch.long))


def test_batch_linear_assignment_compiled_matches_seeded_batched_oracle(
    triton_backend_device: tuple[object, torch.device],
) -> None:
    """Prevent one seeded batch item from changing another item's assignment."""
    backend, device = triton_backend_device
    generator = torch.Generator().manual_seed(32)
    cost = torch.rand((7, 4, 6), generator=generator).to(device)
    expected = _scipy_assignment(cost)

    actual = backend.batch_linear_assignment(cost)

    assert actual.dtype == torch.long
    assert torch.equal(actual.cpu(), expected)


def test_batch_linear_assignment_compiled_matches_large_batched_oracle(
    triton_backend_device: tuple[object, torch.device],
) -> None:
    """Prevent large-vector execution failures exposed by saved batch-624 GPU runs."""
    backend, device = triton_backend_device
    generator = torch.Generator().manual_seed(32)
    cost = torch.randn((624, 300, 300), generator=generator)
    expected = _scipy_assignment(cost)

    actual = backend.batch_linear_assignment(cost.to(device))

    assert actual.dtype == torch.long
    assert torch.equal(actual.cpu(), expected)


def test_batch_linear_assignment_compiled_matches_direct_benchmark_oracle(
    triton_backend_device: tuple[object, torch.device],
) -> None:
    """Prevent 1024-lane direct search termination from changing assignments."""
    backend, device = triton_backend_device
    generator = torch.Generator().manual_seed(32)
    cost = torch.randn((208, 300, 600), generator=generator)
    expected = _scipy_assignment(cost)

    actual = backend.batch_linear_assignment(cost.to(device))

    assert actual.dtype == torch.long
    assert torch.equal(actual.cpu(), expected)


def test_batch_linear_assignment_compiled_handles_non_contiguous_cost(
    triton_backend_device: tuple[object, torch.device],
) -> None:
    """Prevent the wrapper or kernel from treating a strided input as contiguous."""
    backend, device = triton_backend_device
    cost = (
        torch.tensor([[[9.0, 5.0, 1.0], [4.0, 8.0, 3.0], [7.0, 2.0, 6.0], [0.0, 10.0, 11.0]]])
        .transpose(1, 2)
        .to(device)
    )
    assert not cost.is_contiguous()
    expected = _scipy_assignment(cost)

    actual = backend.batch_linear_assignment(cost)

    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, id="float16-promoted-to-float32"),
        pytest.param(torch.bfloat16, id="bfloat16-promoted-to-float32"),
        pytest.param(torch.int64, id="integer-promoted-to-float32"),
    ],
)
def test_batch_linear_assignment_compiled_matches_promoted_scipy_oracle(
    triton_backend_device: tuple[object, torch.device], dtype: torch.dtype
) -> None:
    """Prevent promoted inputs from using the wrong compute dtype or index type."""
    backend, device = triton_backend_device
    cost = torch.tensor(
        [[[8.0, 1.0, 5.0], [3.0, 7.0, 2.0], [6.0, 4.0, 9.0]]],
        dtype=dtype,
        device=device,
    )
    expected = _scipy_assignment(cost)

    actual = backend.batch_linear_assignment(cost)

    assert actual.dtype == torch.long
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, id="float32"),
        pytest.param(torch.float64, id="float64"),
    ],
)
def test_batch_linear_assignment_compiled_matches_floating_scipy_oracle(
    triton_backend_device: tuple[object, torch.device], dtype: torch.dtype
) -> None:
    """Prevent normal floating-point inputs from changing exact assignment indices."""
    backend, device = triton_backend_device
    cost = torch.tensor(
        [[[4.25, 1.5, 3.0], [2.0, 0.125, 5.0], [3.0, 2.0, 2.5]]],
        dtype=dtype,
        device=device,
    )
    expected = _scipy_assignment(cost)

    actual = backend.batch_linear_assignment(cost)

    assert actual.dtype == torch.long
    assert torch.equal(actual.cpu(), expected)


def test_batch_linear_assignment_compiled_preserves_float64_precision(
    triton_backend_device: tuple[object, torch.device],
) -> None:
    """Prevent the prior silent FP64-to-FP32 downcast from changing assignments."""
    backend, device = triton_backend_device
    cost = torch.tensor(
        [[[1.0 + 1e-8, 1.0], [1.0, 1.0 + 1e-8]]],
        dtype=torch.float64,
        device=device,
    )
    expected = _scipy_assignment(cost)
    downcast_expected = _scipy_assignment(cost.to(torch.float32))
    assert torch.equal(expected, torch.tensor([[1, 0]]))
    assert not torch.equal(expected, downcast_expected)

    actual = backend.batch_linear_assignment(cost)

    assert actual.dtype == torch.long
    assert torch.equal(actual.cpu(), expected)


def test_batch_linear_assignment_compiled_disables_cuda_autocast(
    triton_backend_device: tuple[object, torch.device],
) -> None:
    """Prevent ambient AMP from changing private solver assignment indices."""
    backend, device = triton_backend_device
    cost = torch.tensor(
        [[[4.25, 1.5, 3.0], [2.0, 0.125, 5.0], [3.0, 2.0, 2.5]]],
        device=device,
    )
    expected = _scipy_assignment(cost)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        actual = backend.batch_linear_assignment(cost)

    assert actual.dtype == torch.long
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize(
    ("cost", "error_message"),
    [
        pytest.param(
            torch.tensor([[[0.0, float("nan")]]]),
            "matrix contains invalid numeric entries",
            id="nan",
        ),
        pytest.param(
            torch.tensor([[[0.0, float("-inf")]]]),
            "matrix contains invalid numeric entries",
            id="negative-infinity",
        ),
        pytest.param(
            torch.full((1, 2, 2), float("inf")),
            "cost matrix is infeasible",
            id="infeasible-all-infinity-row",
        ),
    ],
)
def test_batch_linear_assignment_compiled_rejects_invalid_costs(
    triton_backend_device: tuple[object, torch.device],
    cost: torch.Tensor,
    error_message: str,
) -> None:
    """Prevent invalid costs from causing a device fault or a vague exception."""
    backend, device = triton_backend_device

    with pytest.raises(ValueError, match=error_message):
        backend.batch_linear_assignment(cost.to(device))


def test_batch_linear_assignment_compiled_accepts_positive_infinity_edges(
    triton_backend_device: tuple[object, torch.device],
) -> None:
    """Prevent valid forbidden-edge matrices from being rejected as non-finite."""
    backend, device = triton_backend_device
    cost = torch.tensor([[[float("inf"), 1.0], [2.0, float("inf")]]], device=device)
    expected = _scipy_assignment(cost)

    actual = backend.batch_linear_assignment(cost)

    assert torch.equal(actual.cpu(), expected)


def test_batch_linear_assignment_compiled_obeys_non_default_stream(
    triton_backend_device: tuple[object, torch.device],
) -> None:
    """Prevent workspace allocation or validation from assuming the default stream."""
    backend, device = triton_backend_device
    cost = torch.tensor([[[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]]], device=device)
    expected = _scipy_assignment(cost)
    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        actual = backend.batch_linear_assignment(cost)
    stream.synchronize()

    assert actual.dtype == torch.long
    assert torch.equal(actual.cpu(), expected)
