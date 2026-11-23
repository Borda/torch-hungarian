"""Compiled-GPU acceptance tests for the private Triton assignment backend.

Most tests call the private wrapper to exercise the compiled backend directly. Tests that
exercise public CUDA dispatch use the eligible-device fixture or assert that the compiled
solve ran, since the public API can fall back to SciPy.
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from scipy.optimize import linear_sum_assignment

from tests._triton_eligibility import compiled_device_params, compiled_triton_skip_reason
from torch_linear_assignment import batch_linear_assignment, batch_linear_assignment_indices


@pytest.fixture(params=compiled_device_params())
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


@pytest.mark.parametrize("validation", ["full", "off", "nonfinite_only", "infeasibility_flag_only"])
@pytest.mark.parametrize(
    "cost",
    [
        pytest.param(torch.tensor([[[4 + 1j, 1 + 2j], [2 + 3j, 3 + 4j]]]), id="complex64"),
        pytest.param(torch.tensor([[[complex(1, float("nan"))]]], dtype=torch.complex128), id="imaginary-nan"),
        pytest.param(torch.empty((0, 2, 2), dtype=torch.complex64), id="empty-complex-batch"),
    ],
)
def test_batch_linear_assignment_compiled_rejects_complex_costs(
    triton_backend_device: tuple[object, torch.device], cost: torch.Tensor, validation: str
) -> None:
    """Never solve a real projection of complex costs, even with validation off."""
    backend, device = triton_backend_device

    with pytest.raises(TypeError, match="Complex costs are not supported"):
        backend.batch_linear_assignment(cost.to(device), validation=validation)


def test_batch_linear_assignment_compiled_accepts_costs_requiring_grad(
    triton_backend_device: tuple[object, torch.device],
) -> None:
    """Accept model-derived costs while keeping discrete assignments detached."""
    backend, device = triton_backend_device
    cost = torch.tensor([[[4.0, 1.0], [2.0, 3.0]]], device=device, requires_grad=True)

    actual = backend.batch_linear_assignment(cost)

    assert torch.equal(actual.cpu(), torch.tensor([[1, 0]], dtype=torch.long))
    assert cost.requires_grad
    assert not actual.requires_grad


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


def test_batch_linear_assignment_compiled_isolates_two_cuda_devices() -> None:
    """Prevent stream, current-device, and workspace state from leaking across GPUs."""
    reason = compiled_triton_skip_reason()
    if reason is not None:
        pytest.skip(reason)

    devices = [
        torch.device(f"cuda:{device_index}")
        for device_index in range(torch.cuda.device_count())
        if torch.cuda.get_device_capability(device_index) >= (8, 0)
    ]
    if len(devices) < 2:
        pytest.skip("compiled multi-device acceptance requires two CUDA devices with capability >= 8.0")

    backend = importlib.import_module("torch_linear_assignment._triton")
    device_costs = [
        (
            devices[0],
            torch.rand((1, 3, 3), generator=torch.Generator().manual_seed(41))
            + torch.tensor([[[0.0, 10.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 0.0]]]),
        ),
        (
            devices[1],
            torch.rand((1, 3, 3), generator=torch.Generator().manual_seed(42))
            + torch.tensor([[[10.0, 0.0, 10.0], [10.0, 10.0, 0.0], [0.0, 10.0, 10.0]]]),
        ),
    ]
    streams = [torch.cuda.Stream(device=device) for device, _ in device_costs]
    expected = [_scipy_assignment(cost) for _, cost in device_costs]
    actual: list[torch.Tensor] = []

    for (device, cost), stream in zip(device_costs, streams):
        with torch.cuda.device(device), torch.cuda.stream(stream):
            assert torch.cuda.current_device() == device.index
            actual.append(backend.batch_linear_assignment(cost.to(device)))

    for device, stream in zip((device for device, _ in device_costs), streams):
        with torch.cuda.device(device):
            stream.synchronize()

    assert not torch.equal(expected[0], expected[1])
    for (device, _), result, scipy_expected in zip(device_costs, actual, expected):
        assert result.device == device
        assert result.dtype == torch.long
        assert torch.equal(result.cpu(), scipy_expected)


def test_public_cuda_solve_runs_on_the_cost_device_without_a_device_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent a cost on a non-current GPU from launching on the current one."""
    reason = compiled_triton_skip_reason()
    if reason is not None:
        pytest.skip(reason)

    devices = [
        torch.device(f"cuda:{device_index}")
        for device_index in range(torch.cuda.device_count())
        if torch.cuda.get_device_capability(device_index) >= (8, 0)
    ]
    if len(devices) < 2:
        pytest.skip("compiled multi-device acceptance requires two CUDA devices with capability >= 8.0")

    backend = importlib.import_module("torch_linear_assignment._triton")
    launched_on: list[int] = []
    original_solve = backend._solve

    def recording_solve(cost: torch.Tensor, *args: object, **kwargs: object) -> object:
        launched_on.append(torch.cuda.current_device())
        return original_solve(cost, *args, **kwargs)

    monkeypatch.setattr(backend, "_solve", recording_solve)
    # Eight problems stay on the GPU; up to four this small are routed to SciPy.
    host_cost = torch.rand((8, 5, 5), generator=torch.Generator().manual_seed(51))

    # The other GPU is current, and the call opens no device context of its own.
    with torch.cuda.device(devices[0]):
        actual = batch_linear_assignment(host_cost.to(devices[1]))

    assert launched_on == [devices[1].index]
    assert actual.device == devices[1]
    assert torch.equal(actual.cpu(), _scipy_assignment(host_cost))


def test_batch_linear_assignment_compiled_offsets_stay_exact_past_two_to_the_31_elements(
    triton_backend_device: tuple[object, torch.device],
) -> None:
    """Prevent 32-bit per-batch offsets from wrapping once a batch holds more than 2**31 elements."""
    backend, device = triton_backend_device
    size = 64
    batch_size = 2**31 // (size * size) + 1024
    cost_bytes = batch_size * size * size * 4
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < cost_bytes * 5 // 4 + 2**30:
        pytest.skip(f"the oversized batch needs about {cost_bytes * 5 // 4 / 2**30:.0f} GiB of free GPU memory")

    generator = torch.Generator(device=device).manual_seed(7)
    cost = torch.rand((batch_size, size, size), generator=generator, device=device)
    assert cost.numel() > 2**31

    # The private backend launches on the current device, which the fixture does not set.
    with torch.cuda.device(device):
        actual = backend.batch_linear_assignment(cost, validation="off")

        # Compare each end of the batch with a small solve that cannot overflow its offsets.
        window = 2048
        for batch_slice in (slice(0, window), slice(batch_size - window, batch_size)):
            expected = backend.batch_linear_assignment(cost[batch_slice].contiguous(), validation="off")
            assert torch.equal(actual[batch_slice], expected)
    assert torch.equal(actual[-4:].cpu(), _scipy_assignment(cost[-4:]))


@pytest.mark.parametrize("offset", [1.0, 1e2, 1e4, 1e6])
@pytest.mark.parametrize("size", [32, 100, 256])
def test_batch_linear_assignment_compiled_float32_offsets_match_scipy(
    triton_backend_device: tuple[object, torch.device], size: int, offset: float
) -> None:
    """Prevent float32 potentials on large cost offsets from returning a strictly worse assignment."""
    backend, device = triton_backend_device
    generator = torch.Generator().manual_seed(size + int(offset))
    cost = offset + torch.rand((8, size, size), generator=generator)
    expected = _scipy_assignment(cost)

    with torch.cuda.device(device):
        actual = backend.batch_linear_assignment(cost.to(device)).cpu()

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("matrix", "dtype"),
    [
        pytest.param([[-1e8, 1], [-1e8, 2]], torch.float32, id="signed-float32"),
        pytest.param([[1.375, 16777216], [0.625, 16777216]], torch.float32, id="wide-float32"),
        pytest.param([[-1e308, 1e308], [-1e308, 1e308]], torch.float64, id="finite-float64-overflow"),
        pytest.param([[-1e308, float("inf")], [-1e308, 1e308]], torch.float64, id="finite-and-forbidden-float64"),
        pytest.param(
            [[1.375, 16777216, float("inf")], [0.625, 16777216, float("inf")], [8, 9, 7]],
            torch.float32,
            id="mixed-safe-and-unsafe-float32",
        ),
        pytest.param(
            [[1024.5, float("inf"), 1024.25], [1024.75, 1024.125, float("inf")]],
            torch.float32,
            id="common-offset-with-padding-and-forbidden-edges",
        ),
    ],
)
def test_batch_linear_assignment_compiled_preserves_finite_costs(
    triton_backend_device: tuple[object, torch.device], matrix: list[list[float]], dtype: torch.dtype
) -> None:
    """Preserve the SciPy assignment when a row shift could round or overflow finite costs."""
    backend, device = triton_backend_device
    cost = torch.tensor([matrix], dtype=dtype)
    expected = _scipy_assignment(cost)

    with torch.cuda.device(device):
        actual = backend.batch_linear_assignment(cost.to(device)).cpu()

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int32])
def test_batch_linear_assignment_compiled_matches_scipy_on_random_shapes(
    triton_backend_device: tuple[object, torch.device], dtype: torch.dtype
) -> None:
    """Prevent the runtime-dimension kernel from drifting on any batch, row or column count."""
    backend, device = triton_backend_device
    generator = torch.Generator().manual_seed(40)
    for _ in range(40):
        batch_size, workers, tasks = (int(value) for value in torch.randint(1, 150, (3,), generator=generator))
        batch_size = batch_size % 16 + 1
        raw = torch.rand((batch_size, workers, tasks), generator=generator) * 50
        cost = raw.floor().to(dtype) if dtype == torch.int32 else raw.to(dtype)

        with torch.cuda.device(device):
            actual = backend.batch_linear_assignment(cost.to(device)).cpu()

        assert torch.equal(actual, _scipy_assignment(cost)), f"{dtype} {tuple(cost.shape)}"


@pytest.mark.parametrize("tasks", [1, 7, 30])
def test_batch_linear_assignment_compiled_matches_scipy_on_tied_costs_at_matcher_shapes(
    triton_backend_device: tuple[object, torch.device], tasks: int
) -> None:
    """Prevent the in-kernel row reduction from moving a tie-break on RF-DETR's tall, tied batches."""
    backend, device = triton_backend_device
    generator = torch.Generator().manual_seed(tasks)
    cost = torch.randint(0, 3, (13, 300, tasks), generator=generator).float()
    expected = _scipy_assignment(cost)

    with torch.cuda.device(device):
        actual = backend.batch_linear_assignment(cost.to(device)).cpu()

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("shapes", "expected_binaries"),
    [
        pytest.param(
            [(3, 300, 1), (5, 300, 2), (7, 300, 5), (11, 300, 9), (13, 300, 17), (17, 300, 30)],
            1,
            id="one-bucket",
        ),
        pytest.param(
            [(3, 40, 20), (5, 100, 50), (7, 200, 150), (9, 300, 300), (11, 60, 20), (13, 500, 400), (15, 129, 40)],
            4,
            id="mixed-buckets",
        ),
        pytest.param(
            [(3, 127, 300), (5, 300, 126), (7, 128, 300), (9, 300, 129)],
            2,
            id="same-bucket-different-warp-counts",
        ),
    ],
)
def test_batch_linear_assignment_compiled_reuses_binaries_within_bucket_and_warp_count(
    triton_backend_device: tuple[object, torch.device],
    tmp_path: Path,
    shapes: list[tuple[int, int, int]],
    expected_binaries: int,
) -> None:
    """Reuse cubins across runtime dimensions when the column bucket and warp count match."""
    backend, device = triton_backend_device
    backend_path = Path(backend.__file__).resolve()
    program = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "backend_path = Path(sys.argv[3])\n"
        "sys.path.insert(0, str(backend_path.parent.parent))\n"
        "import torch\n"
        "from torch_linear_assignment import _triton as backend\n"
        "if Path(backend.__file__).resolve() != backend_path:\n"
        "    raise RuntimeError(f'Unexpected Triton backend: {backend.__file__}')\n"
        "torch.cuda.set_device(int(sys.argv[1]))\n"
        "for shape in json.loads(sys.argv[2]):\n"
        "    backend.batch_linear_assignment(torch.rand(shape, device='cuda'), validation='off')\n"
        "torch.cuda.synchronize()\n"
    )
    # A fresh cache makes every binary this run needs appear here, whatever earlier tests compiled.
    env = {**os.environ, "TRITON_CACHE_DIR": str(tmp_path)}
    subprocess.run(
        [sys.executable, "-B", "-c", program, str(device.index or 0), json.dumps(shapes), str(backend_path)],
        check=True,
        env=env,
        timeout=900,
    )

    assert len(list(tmp_path.rglob("*.cubin"))) == expected_binaries


@pytest.mark.parametrize(
    ("rows", "block_n", "expected"),
    [
        pytest.param(30, 512, 1, id="matcher-shape-few-rows"),
        pytest.param(300, 32, 1, id="matcher-shape-narrow"),
        pytest.param(300, 256, 1, id="below-block-threshold"),
        pytest.param(127, 512, 1, id="below-row-threshold"),
        pytest.param(128, 512, 4, id="at-both-thresholds"),
        pytest.param(300, 1024, 4, id="large-square-like"),
    ],
)
def test_warp_count_uses_several_warps_only_for_large_shapes(
    triton_backend_device: tuple[object, torch.device], rows: int, block_n: int, expected: int
) -> None:
    """Keep small and RF-DETR shapes on one warp, and use several only past both thresholds."""
    backend, _ = triton_backend_device

    assert backend._warp_count(rows, block_n) == expected


# (batch, rows, columns, tied). Small shapes carry the item count; the large ones are where
# several warps pay off, and the 30x300 batch is the RF-DETR matcher shape after transposing.
_WARP_SWEEP_CASES = {
    "tiny-continuous": (25000, 8, 12, False),
    "small-tied": (25000, 24, 40, True),
    "square-continuous": (25000, 30, 30, False),
    "wide-tied": (25000, 16, 64, True),
    "matcher-continuous": (2000, 30, 300, False),
    "large-square": (200, 300, 300, False),
    "largest-square": (100, 512, 512, False),
    "large-wide-tied": (200, 300, 600, True),
}


@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
def test_batch_linear_assignment_compiled_is_exact_at_every_warp_count(
    triton_backend_device: tuple[object, torch.device], num_warps: int
) -> None:
    """Match both SciPy assignment views for over 100k problems at each warp count."""
    backend, device = triton_backend_device
    generator = torch.Generator().manual_seed(1000 + num_warps)
    problems = 0

    for case_id, (batch_size, rows, columns, tied) in _WARP_SWEEP_CASES.items():
        shape = (batch_size, rows, columns)
        cost = (
            torch.randint(0, 3, shape, generator=generator).float() if tied else torch.rand(shape, generator=generator)
        )

        with torch.cuda.device(device):
            col4row, row4col = backend._solve(cost.to(device), "off", num_warps=num_warps)
        expected = _scipy_assignment(cost)
        # Scatter only indices from SciPy. A bad kernel index remains a host assertion failure.
        inverse = torch.full((batch_size, columns), -1, dtype=torch.long)
        inverse.scatter_(1, expected, torch.arange(rows).expand(batch_size, rows))
        assert torch.equal(col4row.cpu().long(), expected), f"{case_id}: row assignment at {num_warps} warps"
        assert torch.equal(row4col.cpu().long(), inverse), f"{case_id}: column assignment at {num_warps} warps"
        problems += batch_size

    assert problems >= 100_000


@pytest.mark.parametrize("num_warps", [1, 2, 4, 8])
def test_batch_linear_assignment_compiled_preserves_float64_at_each_warp_count(
    triton_backend_device: tuple[object, torch.device], num_warps: int
) -> None:
    """Keep a sub-float32 assignment difference in the large float64 kernel."""
    backend, device = triton_backend_device
    cost = torch.full((1, 128, 257), torch.inf, dtype=torch.float64)
    cost[0].fill_diagonal_(0.0)
    cost[0, :2, :2] = torch.tensor([[1.0 + 1e-8, 1.0], [1.0, 1.0 + 1e-8]], dtype=torch.float64)
    expected = _scipy_assignment(cost)
    assert torch.equal(expected[0, :2], torch.tensor([1, 0]))
    assert not torch.equal(expected, _scipy_assignment(cost.float()))

    with torch.cuda.device(device):
        col4row, _ = backend._solve(cost.to(device), "full", num_warps=num_warps)

    assert torch.equal(col4row.cpu().long(), expected)


def _long_augmenting_path_cost() -> torch.Tensor:
    """Build a four-warp cost whose last solver row must displace every previous assignment."""
    cost = torch.full((128, 257), torch.inf)
    rows = torch.arange(127)
    cost[rows, 2 * rows] = 0.0
    cost[rows, 2 * (rows + 1)] = 1.0
    cost[-1, 0] = 0.0
    return cost


def test_batch_linear_assignment_compiled_matches_both_public_views_after_long_tall_augmentation(
    triton_backend_device: tuple[object, torch.device],
) -> None:
    """Compare both public tall results after a four-warp path through forbidden edges."""
    _, device = triton_backend_device
    cost = _long_augmenting_path_cost().T.repeat(2, 1, 1)
    expected_assignment = _scipy_assignment(cost)
    expected_rows, expected_cols = _scipy_indices(cost)

    with torch.cuda.device(device):
        cuda_cost = cost.to(device)
        assignment = batch_linear_assignment(cuda_cost)
        rows, cols = batch_linear_assignment_indices(cuda_cost)

    assert torch.equal(assignment.cpu(), expected_assignment)
    assert torch.equal(rows.cpu(), expected_rows)
    assert torch.equal(cols.cpu(), expected_cols)


def test_batch_linear_assignment_compiled_rejects_large_infeasible_augmenting_path(
    triton_backend_device: tuple[object, torch.device],
) -> None:
    """Raise on a four-warp search with no free column and a finite edge in every solver row."""
    _, device = triton_backend_device
    cost = _long_augmenting_path_cost()
    cost[-2, 254] = torch.inf
    tall_cost = cost.T.repeat(2, 1, 1).to(device)

    with torch.cuda.device(device), pytest.raises(ValueError, match="cost matrix is infeasible"):
        batch_linear_assignment(tall_cost)


def _scipy_indices(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return SciPy's row-sorted ``(row_ind, col_ind)`` stacked over the batch, after the solver's promotion."""
    solver_dtype = torch.float64 if cost.dtype == torch.float64 else torch.float32
    pairs = [linear_sum_assignment(matrix.numpy()) for matrix in cost.detach().to(device="cpu", dtype=solver_dtype)]
    rows = torch.stack([torch.from_numpy(row_ind) for row_ind, _ in pairs]).long()
    cols = torch.stack([torch.from_numpy(col_ind) for _, col_ind in pairs]).long()
    return rows, cols


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((5, 40, 40), id="square"),
        pytest.param((5, 300, 7), id="tall"),
        pytest.param((5, 7, 300), id="wide"),
        pytest.param((6, 300, 30), id="matcher-shape"),
        pytest.param((5, 1, 9), id="one-worker"),
        pytest.param((5, 9, 1), id="one-task"),
    ],
)
@pytest.mark.parametrize("validation", ["full", "off"])
def test_batch_linear_assignment_indices_compiled_match_scipy_pairs(
    triton_backend_device: tuple[object, torch.device], shape: tuple[int, int, int], validation: str
) -> None:
    """Return SciPy's row-sorted pairs of static shape (B, min(W, T)) in either validation mode."""
    backend, device = triton_backend_device
    cost = torch.rand(shape, generator=torch.Generator().manual_seed(sum(shape)))

    with torch.cuda.device(device):
        rows, cols = backend.batch_linear_assignment_indices(cost.to(device), validation=validation)

    expected_rows, expected_cols = _scipy_indices(cost)
    assert rows.device == cols.device == device
    assert rows.dtype == cols.dtype == torch.long
    assert rows.shape == cols.shape == (shape[0], min(shape[1:]))
    assert torch.equal(rows.cpu(), expected_rows)
    assert torch.equal(cols.cpu(), expected_cols)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64, torch.int32])
def test_batch_linear_assignment_indices_compiled_match_scipy_for_promoted_dtypes(
    triton_backend_device: tuple[object, torch.device], dtype: torch.dtype
) -> None:
    """Promote the solver dtype exactly as the assignment does, so both views agree with SciPy."""
    backend, device = triton_backend_device
    raw = torch.rand((6, 50, 20), generator=torch.Generator().manual_seed(9)) * 40
    cost = raw.floor().to(dtype) if dtype == torch.int32 else raw.to(dtype)

    with torch.cuda.device(device):
        rows, cols = backend.batch_linear_assignment_indices(cost.to(device))

    expected_rows, expected_cols = _scipy_indices(cost)
    assert torch.equal(rows.cpu(), expected_rows)
    assert torch.equal(cols.cpu(), expected_cols)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        pytest.param((0, 4, 3), (0, 3), id="no-batch"),
        pytest.param((2, 0, 3), (2, 0), id="no-workers"),
        pytest.param((2, 4, 0), (2, 0), id="no-tasks"),
    ],
)
def test_batch_linear_assignment_indices_compiled_keep_the_static_shape_for_empty_costs(
    triton_backend_device: tuple[object, torch.device], shape: tuple[int, int, int], expected: tuple[int, int]
) -> None:
    """Return empty long pairs of shape (B, min(W, T)) without launching a kernel."""
    backend, device = triton_backend_device

    rows, cols = backend.batch_linear_assignment_indices(torch.zeros(shape, device=device))

    assert rows.shape == cols.shape == expected
    assert rows.dtype == cols.dtype == torch.long
    assert rows.device == cols.device == device


@pytest.mark.parametrize(
    ("cost", "error", "message"),
    [
        pytest.param(torch.tensor([[[0.0, float("nan")]]]), ValueError, "invalid numeric entries", id="nan"),
        pytest.param(
            torch.tensor([[[0.0, float("-inf")]]]), ValueError, "invalid numeric entries", id="negative-infinity"
        ),
        pytest.param(torch.full((1, 2, 2), float("inf")), ValueError, "infeasible", id="infeasible"),
        pytest.param(torch.zeros(1, 2, 2, dtype=torch.complex64), TypeError, "Complex costs", id="complex"),
    ],
)
def test_batch_linear_assignment_indices_compiled_raise_the_assignment_errors(
    triton_backend_device: tuple[object, torch.device], cost: torch.Tensor, error: type[Exception], message: str
) -> None:
    """Reject, under full validation, the costs the assignment rejects, with the same exception types."""
    backend, device = triton_backend_device

    with pytest.raises(error, match=message), torch.cuda.device(device):
        backend.batch_linear_assignment_indices(cost.to(device))


def test_batch_linear_assignment_indices_compiled_disable_cuda_autocast(
    triton_backend_device: tuple[object, torch.device],
) -> None:
    """Prevent ambient AMP from changing the pairs the solver returns."""
    backend, device = triton_backend_device
    cost = torch.rand((5, 12, 6), generator=torch.Generator().manual_seed(3))

    with torch.cuda.device(device), torch.amp.autocast("cuda", dtype=torch.float16):
        rows, cols = backend.batch_linear_assignment_indices(cost.to(device))

    expected_rows, expected_cols = _scipy_indices(cost)
    assert torch.equal(rows.cpu(), expected_rows)
    assert torch.equal(cols.cpu(), expected_cols)


def test_public_indices_replay_in_a_cuda_graph_with_validation_off(
    triton_backend_device: tuple[object, torch.device],
) -> None:
    """Capture the unchecked solve once after warm-up and replay it on new costs with exact pairs."""
    _, device = triton_backend_device
    shape = (13, 300, 7)
    generator = torch.Generator().manual_seed(21)
    with torch.cuda.device(device):
        static_cost = torch.rand(shape, generator=generator).to(device)
        # Compilation and workspace allocation happen on a side stream before capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            batch_linear_assignment_indices(static_cost, validate="off")
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            rows, cols = batch_linear_assignment_indices(static_cost, validate="off")

        for _ in range(3):
            fresh = torch.rand(shape, generator=generator)
            static_cost.copy_(fresh)
            graph.replay()
            expected_rows, expected_cols = _scipy_indices(fresh)
            assert torch.equal(rows.cpu(), expected_rows)
            assert torch.equal(cols.cpu(), expected_cols)


@pytest.mark.parametrize(
    ("shape", "launches"),
    [
        pytest.param((1, 300, 300), 0, id="single-problem-goes-to-scipy"),
        pytest.param((4, 30, 30), 0, id="four-small-problems-go-to-scipy"),
        pytest.param((5, 30, 30), 1, id="five-small-problems-launch-the-kernel"),
        pytest.param((13, 300, 7), 1, id="matcher-shape-launches-the-kernel"),
    ],
)
def test_public_calls_route_small_batches_to_scipy(
    triton_backend_device: tuple[object, torch.device],
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int, int],
    launches: int,
) -> None:
    """Launch the kernel only for batches the routing rule leaves on the GPU, with SciPy's answer either way."""
    backend, device = triton_backend_device
    launched: list[tuple[int, ...]] = []
    original_solve = backend._solve

    def recording_solve(cost: torch.Tensor, *args: object, **kwargs: object) -> object:
        launched.append(tuple(cost.shape))
        return original_solve(cost, *args, **kwargs)

    monkeypatch.setattr(backend, "_solve", recording_solve)
    host_cost = torch.rand(shape, generator=torch.Generator().manual_seed(sum(shape)))

    cost = host_cost.to(device)
    with torch.cuda.device(device):
        assignment = batch_linear_assignment(cost)
        rows, cols = batch_linear_assignment_indices(cost)

    assert len(launched) == 2 * launches
    assert torch.equal(assignment.cpu(), _scipy_assignment(host_cost))
    expected_rows, expected_cols = _scipy_indices(host_cost)
    assert torch.equal(rows.cpu(), expected_rows)
    assert torch.equal(cols.cpu(), expected_cols)
