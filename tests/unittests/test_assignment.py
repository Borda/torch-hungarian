import hashlib
import subprocess
import sys
import textwrap
import warnings
from unittest import TestCase

import pytest
import torch
from scipy.optimize import linear_sum_assignment

import torch_linear_assignment.assignment as assignment_module
from torch_linear_assignment import batch_linear_assignment


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16], ids=["fp32", "fp64", "bf16"])
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


class TestAssignment(TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_simple(self):
        cost = torch.tensor([8, 4, 7, 5, 2, 3, 9, 6, 7, 9, 4, 8]).reshape(1, 4, 3).to(self.device)
        gt_assignment = torch.tensor([0, 2, -1, 1]).reshape(1, 4)
        result = batch_linear_assignment(cost).cpu()
        print(result)
        self.assertTrue((result == gt_assignment).all())
