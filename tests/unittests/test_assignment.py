import hashlib
import subprocess
import sys
import textwrap
import warnings
from types import SimpleNamespace
from unittest import TestCase

import pytest
import torch
from scipy.optimize import linear_sum_assignment

import torch_linear_assignment.assignment as assignment_module
from torch_linear_assignment import batch_linear_assignment, batch_linear_assignment_indices


def scipy_assignment(cost):
    """Return the documented CPU assignment result for a batched cost tensor."""
    batch_size, workers, _ = cost.shape
    expected = torch.full((batch_size, workers), -1, dtype=torch.long)
    for batch_index, matrix in enumerate(cost):
        row_ind, col_ind = linear_sum_assignment(matrix.numpy(), maximize=False)
        expected[batch_index, torch.from_numpy(row_ind)] = torch.from_numpy(col_ind)
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
        pytest.param(
            torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            id="exact-ties",
        ),
        pytest.param(
            torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]]),
            id="evolving-scan-order-ties",
        ),
        pytest.param(
            torch.tensor([[[-4.0, -1.0, -3.0], [-2.0, -5.0, -6.0], [-7.0, -8.0, -9.0]]]),
            id="negative-costs",
        ),
        pytest.param(
            torch.tensor([[[9, 1, 8], [2, 7, 3], [6, 4, 5]]], dtype=torch.int64),
            id="integer-costs",
        ),
    ],
)
def test_batch_linear_assignment_cpu_matches_scipy_deterministic_costs(cost):
    """Prevent CPU regressions in batching, rectangular handling, and tie selection."""
    expected = scipy_assignment(cost)

    actual = batch_linear_assignment(cost)

    assert actual.dtype == torch.long
    assert torch.equal(actual, expected)


def test_batch_linear_assignment_cpu_matches_scipy_for_non_contiguous_cost():
    """Prevent a CPU solver from assuming a contiguous cost layout."""
    cost = torch.tensor([[[9.0, 5.0, 1.0], [4.0, 8.0, 3.0], [7.0, 2.0, 6.0], [0.0, 10.0, 11.0]]]).transpose(1, 2)
    assert not cost.is_contiguous()
    expected = scipy_assignment(cost)

    actual = batch_linear_assignment(cost)

    assert torch.equal(actual, expected)


def test_batch_linear_assignment_empty_batch_preserves_worker_dimension():
    """Prevent empty batches from losing their public assignment shape or dtype."""
    cost = torch.empty((0, 3, 5), dtype=torch.float64)

    actual = batch_linear_assignment(cost)

    assert actual.shape == (0, 3)
    assert actual.dtype == torch.long


@pytest.mark.parametrize(
    "cost, expected",
    [
        pytest.param(
            torch.empty((1, 0, 3)),
            torch.empty((1, 0), dtype=torch.long),
            id="zero-workers",
        ),
        pytest.param(
            torch.empty((1, 3, 0)),
            torch.full((1, 3), -1, dtype=torch.long),
            id="zero-tasks",
        ),
        pytest.param(
            torch.empty((1, 0, 0)),
            torch.empty((1, 0), dtype=torch.long),
            id="zero-workers-and-tasks",
        ),
    ],
)
def test_batch_linear_assignment_cpu_handles_scipy_permitted_zero_dimensions(cost, expected):
    """Pin SciPy-permitted zero-dimension assignment results."""
    actual = batch_linear_assignment(cost)

    assert torch.equal(actual, expected)


def test_batch_linear_assignment_rejects_non_batched_cost():
    """Prevent rank-two costs from silently being treated as a batch."""
    cost = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError, match="Need 3-dimensional tensor with shape \\(B, W, T\\)\\."):
        batch_linear_assignment(cost)


@pytest.mark.parametrize(
    "cost, error_message",
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
            torch.tensor([[[float("inf"), 1.0], [2.0, float("inf")]]]),
            None,
            id="positive-infinity-forbidden-edges",
        ),
        pytest.param(
            torch.full((1, 2, 2), float("inf")),
            "cost matrix is infeasible",
            id="infeasible-all-infinity-row",
        ),
    ],
)
def test_batch_linear_assignment_cpu_matches_scipy_non_finite_contract(cost, error_message):
    """Prevent CPU non-finite handling from diverging from SciPy semantics."""
    if error_message is not None:
        with pytest.raises(ValueError, match=error_message):
            batch_linear_assignment(cost)
        return

    expected = scipy_assignment(cost)
    actual = batch_linear_assignment(cost)

    assert torch.equal(actual, expected)


def test_package_import_does_not_require_optional_backend_modules():
    """Prevent project-private accelerator modules from becoming eager imports."""
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockOptionalBackend(importlib.abc.MetaPathFinder):
            blocked = {"torch_linear_assignment._triton"}

            def find_spec(self, fullname, path=None, target=None):
                if fullname in self.blocked:
                    raise ModuleNotFoundError(f"blocked optional dependency: {fullname}", name=fullname)
                return None

        sys.meta_path.insert(0, BlockOptionalBackend())
        import torch
        from torch_linear_assignment import batch_linear_assignment

        cost = torch.tensor([[[4.0, 1.0], [2.0, 3.0]]])
        expected = torch.tensor([[1, 0]], dtype=torch.long)
        assert torch.equal(batch_linear_assignment(cost), expected)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_batch_linear_assignment_cpu_promotes_bfloat16_cost_for_scipy():
    """Prevent CPU BF16 costs from reaching SciPy's unsupported NumPy boundary."""
    cost = torch.tensor(
        [[[8.0, 1.0, 5.0], [3.0, 7.0, 2.0], [6.0, 4.0, 9.0]]],
        dtype=torch.bfloat16,
    )
    expected = scipy_assignment(cost.to(torch.float32))

    actual = batch_linear_assignment(cost)

    assert actual.dtype == torch.long
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, id="fp32"),
        pytest.param(torch.float64, id="fp64"),
        pytest.param(torch.bfloat16, id="bf16"),
    ],
)
def test_batch_linear_assignment_cpu_accepts_costs_requiring_grad(dtype):
    """Solve model-derived costs without altering the input's autograd graph."""
    leaf = torch.tensor([[[4.0, 1.0], [2.0, 3.0]]], dtype=dtype, requires_grad=True)
    cost = leaf * 2
    original = cost.detach().clone()

    actual = batch_linear_assignment(cost)

    assert torch.equal(actual, torch.tensor([[1, 0]], dtype=torch.long))
    assert not actual.requires_grad
    assert cost.requires_grad
    assert torch.equal(cost.detach(), original)
    cost.sum().backward()
    assert torch.equal(leaf.grad, torch.full_like(leaf, 2))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")),
    ],
)
@pytest.mark.parametrize(
    "cost",
    [
        pytest.param(torch.tensor([[[4 + 1j, 1 + 2j], [2 + 3j, 3 + 4j]]]), id="complex64"),
        pytest.param(torch.tensor([[[complex(1, float("nan"))]]], dtype=torch.complex128), id="imaginary-nan"),
        pytest.param(torch.empty((0, 2, 2), dtype=torch.complex64), id="empty-complex-batch"),
    ],
)
def test_batch_linear_assignment_rejects_complex_costs(cost, device):
    """Reject undefined complex objectives, including empty and invalid batches."""
    with pytest.raises(TypeError, match="Complex costs are not supported"):
        batch_linear_assignment(cost.to(device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.int64, id="integer"),
        pytest.param(torch.bfloat16, id="bfloat16"),
        pytest.param(torch.float32, id="float32-requires-grad"),
    ],
)
def test_batch_linear_assignment_cuda_fallback_uses_promoted_scipy_once(monkeypatch, dtype):
    """Keep fallback promotion, autograd inputs, and warn-once behavior consistent."""
    cpu_cost = torch.tensor(
        [[[8.0, 1.0, 5.0], [3.0, 7.0, 2.0], [6.0, 4.0, 9.0]]],
        dtype=dtype,
        requires_grad=dtype == torch.float32,
    )
    cost = cpu_cost.cuda()
    expected = scipy_assignment(cpu_cost.detach().to(torch.float32))
    monkeypatch.setattr(assignment_module, "_CUDA_FALLBACK_WARNING_EMITTED", False)
    monkeypatch.setattr(assignment_module, "_cuda_uses_triton", lambda _: False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = batch_linear_assignment(cost)
        second = batch_linear_assignment(cost)

    assert len(caught) == 1
    assert "Triton linear-assignment support" in str(caught[0].message)
    assert first.device == cost.device
    assert first.dtype == torch.long
    assert torch.equal(first.cpu(), expected)
    assert torch.equal(second.cpu(), expected)
    if dtype == torch.float32:
        assert cost.requires_grad
        cost.sum().backward()
        assert torch.equal(cpu_cost.grad, torch.ones_like(cpu_cost))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("shape", "seed"),
    [
        pytest.param((16, 20, 40), 32, id="batched-direct"),
        pytest.param((1, 30, 10), 33, id="transpose"),
        pytest.param((1, 20, 40), 34, id="single-direct"),
        pytest.param((0, 5, 5), 35, id="empty-batch"),
    ],
)
def test_batch_linear_assignment_cuda_integer_parity_is_reproducible(shape, seed):
    """Pin CUDA integer parity to replayable matrices with objective diagnostics."""
    generator = torch.Generator().manual_seed(seed)
    cost = torch.randint(-10, 10, shape, generator=generator)
    expected = batch_linear_assignment(cost)
    actual = batch_linear_assignment(cost.cuda()).cpu()

    assert expected.shape == actual.shape
    assert expected.dtype == actual.dtype
    if torch.equal(actual, expected):
        return

    matched_workers = expected >= 0
    workers = torch.arange(shape[1]).expand(shape[0], -1)
    batch = torch.arange(shape[0]).unsqueeze(1).expand_as(workers)
    safe_expected = expected.clamp_min(0)
    safe_actual = actual.clamp_min(0)
    expected_objective = torch.where(
        matched_workers,
        cost[batch, workers, safe_expected],
        0,
    ).sum(dim=1)
    actual_objective = torch.where(
        actual >= 0,
        cost[batch, workers, safe_actual],
        0,
    ).sum(dim=1)
    mismatched_batches = (actual != expected).any(dim=1).nonzero().flatten().tolist()
    digest = hashlib.sha256(cost.numpy().tobytes()).hexdigest()
    pytest.fail(
        f"CUDA integer parity mismatch: shape={shape}, seed={seed}, sha256={digest}, "
        f"batches={mismatched_batches}, expected={expected.tolist()}, actual={actual.tolist()}, "
        f"expected_objective={expected_objective.tolist()}, actual_objective={actual_objective.tolist()}"
    )


class _DeviceProxy(torch.Tensor):
    """A host tensor that reports another device, so device handling runs without that hardware."""

    reported_device: torch.device

    @property
    def device(self) -> torch.device:
        """Report the device the tensor stands in for."""
        return self.reported_device

    def cpu(self, *args, **kwargs) -> torch.Tensor:
        """Return the plain host tensor, as a copy off a real accelerator would."""
        return self.as_subclass(torch.Tensor)


def _device_proxy(cost: torch.Tensor, device: str) -> _DeviceProxy:
    """Wrap a host cost so it reports ``device``."""
    proxy = cost.as_subclass(_DeviceProxy)
    proxy.reported_device = torch.device(device)
    return proxy


def test_batch_linear_assignment_solves_other_accelerators_on_host(monkeypatch):
    """Prevent an MPS, XPU or other non-CUDA input from reaching the host solver off its device."""
    host_cost = torch.tensor([[[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]]])
    solved_on = []

    def host_solver(cost):
        solved_on.append(cost.device)
        return scipy_assignment(cost)

    monkeypatch.setattr(assignment_module, "batch_linear_assignment_cpu", host_solver)

    result = batch_linear_assignment(_device_proxy(host_cost, "meta"))

    assert solved_on == [torch.device("cpu")]
    assert result.device.type == "meta"
    assert result.dtype == torch.long
    assert result.shape == (1, 3)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((2, 4, 4), id="square"),
        pytest.param((1, 6, 3), id="tall"),
        pytest.param((1, 3, 6), id="wide"),
    ],
)
def test_batch_linear_assignment_solves_mps_costs_on_host(shape):
    """Return SciPy's assignment on the MPS device for square, tall and wide costs."""
    cost = torch.rand(shape, generator=torch.Generator().manual_seed(11))

    result = batch_linear_assignment(cost.to("mps"))

    assert result.device.type == "mps"
    assert torch.equal(result.cpu(), scipy_assignment(cost))


@pytest.mark.parametrize(
    ("current_index", "cost_index", "entered"),
    [
        pytest.param(0, 1, [1], id="other-device"),
        pytest.param(1, 1, [], id="current-device"),
    ],
)
def test_batch_linear_assignment_cuda_launches_on_the_cost_device(monkeypatch, current_index, cost_index, entered):
    """Prevent a cost on a non-current GPU from launching on the current one, without a guard on the common path."""
    state = {"current": current_index}
    entered_indices = []
    launched_on = []

    class FakeDeviceContext:
        """Stand in for ``torch.cuda.device`` by moving a fake current-device index."""

        def __init__(self, device):
            self.index = torch.device(device).index

        def __enter__(self):
            entered_indices.append(self.index)
            self.previous, state["current"] = state["current"], self.index

        def __exit__(self, *exc_info):
            state["current"] = self.previous

    def launch(cost):
        launched_on.append(state["current"])
        return "assignment"

    monkeypatch.setattr(torch.cuda, "current_device", lambda: state["current"])
    monkeypatch.setattr(torch.cuda, "device", FakeDeviceContext)
    monkeypatch.setattr(
        assignment_module, "_load_triton_backend", lambda: SimpleNamespace(batch_linear_assignment=launch)
    )

    cost = _device_proxy(torch.zeros(1, 2, 2), f"cuda:{cost_index}")

    assert assignment_module.batch_linear_assignment_cuda(cost) == "assignment"
    assert launched_on == [cost_index]
    assert entered_indices == entered
    assert state["current"] == current_index


def scipy_indices(cost):
    """Return SciPy's ``(row_ind, col_ind)`` stacked over the batch, the oracle for the indices API."""
    pairs = [linear_sum_assignment(matrix.numpy(), maximize=False) for matrix in cost]
    rows = torch.stack([torch.from_numpy(row_ind) for row_ind, _ in pairs]).long()
    cols = torch.stack([torch.from_numpy(col_ind) for _, col_ind in pairs]).long()
    return rows, cols


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((3, 4, 4), id="square"),
        pytest.param((2, 6, 3), id="tall"),
        pytest.param((2, 3, 6), id="wide"),
        pytest.param((1, 1, 5), id="one-worker"),
        pytest.param((4, 5, 1), id="one-task"),
        pytest.param((2, 40, 7), id="very-tall"),
        pytest.param((2, 7, 40), id="very-wide"),
    ],
)
def test_batch_linear_assignment_indices_match_scipy_pairs(shape):
    """Return SciPy's row-sorted pairs, of static shape (B, min(W, T)), for every orientation."""
    cost = torch.rand(shape, generator=torch.Generator().manual_seed(shape[1] * 31 + shape[2]))

    rows, cols = batch_linear_assignment_indices(cost)

    expected_rows, expected_cols = scipy_indices(cost)
    assert rows.shape == cols.shape == (shape[0], min(shape[1:]))
    assert rows.dtype == cols.dtype == torch.long
    assert torch.equal(rows, expected_rows)
    assert torch.equal(cols, expected_cols)


def test_batch_linear_assignment_indices_agree_with_the_assignment():
    """Place every returned pair where the assignment tensor puts it."""
    cost = torch.rand((3, 9, 5), generator=torch.Generator().manual_seed(8))
    assignment = batch_linear_assignment(cost)

    rows, cols = batch_linear_assignment_indices(cost)

    assert torch.equal(assignment.gather(1, rows), cols)
    assert int((assignment >= 0).sum()) == rows.numel()


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        pytest.param((0, 4, 3), (0, 3), id="no-batch"),
        pytest.param((2, 0, 3), (2, 0), id="no-workers"),
        pytest.param((2, 4, 0), (2, 0), id="no-tasks"),
        pytest.param((0, 0, 0), (0, 0), id="all-empty"),
    ],
)
def test_batch_linear_assignment_indices_keep_the_static_shape_for_empty_costs(shape, expected):
    """Return empty pairs whose shape is still (B, min(W, T))."""
    rows, cols = batch_linear_assignment_indices(torch.zeros(shape))

    assert rows.shape == cols.shape == expected
    assert rows.dtype == cols.dtype == torch.long


@pytest.mark.parametrize("default_device", ["cpu", "meta"])
def test_empty_indices_use_the_input_device(default_device):
    """Return CPU pairs for an explicit CPU cost under a different default device."""
    cost = torch.empty((0, 4, 3), device="cpu")

    with torch.device(default_device):
        rows, cols = batch_linear_assignment_indices(cost)

    assert rows.device == cols.device == cost.device
    assert rows.shape == cols.shape == (0, 3)
    assert rows.dtype == cols.dtype == torch.long


@pytest.mark.parametrize("validate", ["full", "off"])
def test_batch_linear_assignment_indices_accept_both_validate_modes_on_the_host(validate):
    """Give the same pairs under either mode, since SciPy validates on every host path."""
    cost = torch.rand((2, 5, 5), generator=torch.Generator().manual_seed(2))

    rows, cols = batch_linear_assignment_indices(cost, validate=validate)

    expected_rows, expected_cols = scipy_indices(cost)
    assert torch.equal(rows, expected_rows)
    assert torch.equal(cols, expected_cols)


def test_batch_linear_assignment_indices_accept_costs_requiring_grad():
    """Solve a gradient-bearing cost without touching its autograd graph."""
    cost = torch.rand((2, 4, 6), generator=torch.Generator().manual_seed(4), requires_grad=True)

    rows, cols = batch_linear_assignment_indices(cost)

    assert not rows.requires_grad
    assert not cols.requires_grad
    assert cost.requires_grad
    assert torch.equal(rows, scipy_indices(cost.detach())[0])


@pytest.mark.parametrize(
    ("cost", "error", "message"),
    [
        pytest.param(torch.zeros(3, 3), ValueError, "Need 3-dimensional tensor", id="rank"),
        pytest.param(torch.zeros(1, 2, 2, dtype=torch.complex64), TypeError, "Complex costs", id="complex"),
        pytest.param(torch.zeros(0, 2, 2, dtype=torch.complex64), TypeError, "Complex costs", id="complex-empty"),
        pytest.param(torch.tensor([[[0.0, float("nan")]]]), ValueError, "invalid numeric entries", id="nan"),
        pytest.param(torch.full((1, 2, 2), float("inf")), ValueError, "infeasible", id="infeasible"),
    ],
)
def test_batch_linear_assignment_indices_raise_the_assignment_errors(cost, error, message):
    """Reject the inputs batch_linear_assignment rejects, with the same exception types."""
    with pytest.raises(error, match=message):
        batch_linear_assignment_indices(cost)


class _UnhashableMode(str):
    """A string value whose type does not support hashing."""

    __hash__ = None


@pytest.mark.parametrize(
    "validate",
    [
        "deferred",
        pytest.param([], id="list"),
        pytest.param({}, id="dict"),
        pytest.param(_UnhashableMode("deferred"), id="unhashable-string"),
    ],
)
def test_batch_linear_assignment_indices_reject_an_unknown_validate_mode(validate):
    """Name the bad mode instead of treating it as one of the two."""
    with pytest.raises(ValueError, match="Unknown validate mode:"):
        batch_linear_assignment_indices(torch.zeros(1, 2, 2), validate=validate)


def test_batch_linear_assignment_indices_solve_other_accelerators_on_host(monkeypatch):
    """Return the pairs on the input's device after a host solve, as the assignment does."""
    host_cost = torch.rand((2, 4, 6), generator=torch.Generator().manual_seed(6))
    solved_on = []
    real_solver = assignment_module.batch_linear_assignment_cpu

    def host_solver(cost):
        solved_on.append(cost.device)
        return real_solver(cost)

    monkeypatch.setattr(assignment_module, "batch_linear_assignment_cpu", host_solver)

    rows, cols = batch_linear_assignment_indices(_device_proxy(host_cost, "meta"))

    assert solved_on == [torch.device("cpu")]
    assert rows.device.type == cols.device.type == "meta"
    assert rows.shape == cols.shape == (2, 4)


@pytest.mark.parametrize(
    ("batch_size", "side", "host"),
    [
        (1, 4, True),
        (1, 4096, True),
        (2, 64, True),
        (2, 65, False),
        (3, 32, True),
        (3, 33, False),
        (4, 32, True),
        (4, 33, False),
        (5, 8, False),
        (208, 300, False),
        (0, 300, False),
    ],
)
@pytest.mark.parametrize("orientation", ["tall", "wide"])
def test_prefers_host_follows_the_measured_routing_rule(batch_size, side, host, orientation):
    """Route one problem, two up to 64 wide, and up to four up to 32 wide to SciPy, whichever side is larger."""
    workers, tasks = (side, 2) if orientation == "tall" else (2, side)

    assert assignment_module._prefers_host(batch_size, workers, tasks) is host


@pytest.mark.parametrize(
    ("shape", "eligible", "uses_triton", "warns"),
    [
        pytest.param((8, 300, 30), True, True, False, id="large-batch-goes-to-triton"),
        pytest.param((1, 300, 30), True, False, False, id="single-problem-routes-to-host-silently"),
        pytest.param((8, 300, 30), False, False, True, id="ineligible-gpu-warns-and-falls-back"),
        pytest.param((1, 300, 30), False, False, True, id="ineligible-gpu-warns-even-for-one-problem"),
    ],
)
def test_cuda_cost_dispatch_routes_small_batches_to_the_host_without_a_warning(
    monkeypatch, shape, eligible, uses_triton, warns
):
    """Warn about the fallback only when Triton is unavailable, never for a deliberate small-batch route."""
    monkeypatch.setattr(assignment_module, "_CUDA_FALLBACK_WARNING_EMITTED", False)
    monkeypatch.setattr(assignment_module, "_cuda_uses_triton", lambda _: eligible)
    cost = SimpleNamespace(is_cuda=True, shape=shape)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decision = assignment_module._uses_triton(cost)

    assert decision is uses_triton
    assert bool(caught) is warns


@pytest.mark.parametrize("public_call", [batch_linear_assignment, batch_linear_assignment_indices])
def test_cuda_fallback_warning_points_at_the_caller_of_the_public_function(monkeypatch, public_call):
    """Attribute the once-only fallback warning to the user's call, not to the package's dispatch frames."""
    monkeypatch.setattr(assignment_module, "_CUDA_FALLBACK_WARNING_EMITTED", False)
    monkeypatch.setattr(assignment_module, "_cuda_uses_triton", lambda _: False)
    monkeypatch.setattr(assignment_module, "_solve_on_host", lambda _cost: torch.zeros((8, 2), dtype=torch.long))
    monkeypatch.setattr(
        assignment_module, "assignment_to_indices", lambda _assignment: (torch.zeros(8, 2), torch.zeros(8, 2))
    )
    cost = SimpleNamespace(is_cuda=True, ndim=3, shape=(8, 2, 2), device=torch.device("meta"), is_complex=lambda: False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        public_call(cost)

    assert len(caught) == 1
    assert caught[0].filename == __file__


def test_batch_linear_assignment_indices_pass_validate_to_the_triton_backend(monkeypatch):
    """Send the validate mode to the backend and run it with the cost's GPU current."""
    seen = []

    def backend_indices(cost, *, validation):
        seen.append(validation)
        return "pairs"

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(assignment_module, "_uses_triton", lambda _: True)
    monkeypatch.setattr(
        assignment_module,
        "_require_triton_backend",
        lambda: SimpleNamespace(batch_linear_assignment_indices=backend_indices),
    )
    cost = _device_proxy(torch.zeros(8, 2, 2), "cuda:0")

    assert batch_linear_assignment_indices(cost) == "pairs"
    assert batch_linear_assignment_indices(cost, validate="off") == "pairs"
    assert seen == ["full", "off"]


class TestAssignment(TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_simple(self):
        cost = torch.tensor([8, 4, 7, 5, 2, 3, 9, 6, 7, 9, 4, 8]).reshape(1, 4, 3).to(self.device)
        gt_assignment = torch.tensor([0, 2, -1, 1]).reshape(1, 4)
        result = batch_linear_assignment(cost).cpu()
        print(result)
        self.assertTrue((result == gt_assignment).all())
