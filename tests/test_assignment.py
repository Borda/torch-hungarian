import subprocess
import sys
import textwrap
from unittest import TestCase

import pytest
import torch
from scipy.optimize import linear_sum_assignment
from torch_linear_assignment import batch_linear_assignment
import torch_linear_assignment.assignment as assignment_module


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
            torch.tensor(
                [[[-4.0, -1.0, -3.0], [-2.0, -5.0, -6.0], [-7.0, -8.0, -9.0]]]
            ),
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
    cost = torch.tensor(
        [[[9.0, 5.0, 1.0], [4.0, 8.0, 3.0], [7.0, 2.0, 6.0], [0.0, 10.0, 11.0]]]
    ).transpose(1, 2)
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
def test_batch_linear_assignment_cpu_handles_scipy_permitted_zero_dimensions(
    cost, expected
):
    """Pin SciPy-permitted zero-dimension assignment results."""
    actual = batch_linear_assignment(cost)

    assert torch.equal(actual, expected)


def test_batch_linear_assignment_rejects_non_batched_cost():
    """Prevent rank-two costs from silently being treated as a batch."""
    cost = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(
        ValueError, match="Need 3-dimensional tensor with shape \\(B, W, T\\)\\."
    ):
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
def test_batch_linear_assignment_cpu_matches_scipy_non_finite_contract(
    cost, error_message
):
    """Prevent CPU non-finite handling from diverging from SciPy semantics."""
    if error_message is not None:
        with pytest.raises(ValueError, match=error_message):
            batch_linear_assignment(cost)
        return

    expected = scipy_assignment(cost)
    actual = batch_linear_assignment(cost)

    assert torch.equal(actual, expected)


def test_package_import_and_cuda_fallback_do_not_require_backend_or_triton():
    """Prevent eager optional imports and repeated warnings when the GPU path is unavailable."""
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        import warnings

        class BlockOptionalBackend(importlib.abc.MetaPathFinder):
            blocked = {"torch_linear_assignment._backend", "torch_linear_assignment._triton", "triton"}

            def find_spec(self, fullname, path=None, target=None):
                if fullname in self.blocked:
                    raise ModuleNotFoundError(f"blocked optional dependency: {fullname}", name=fullname)
                return None

        sys.meta_path.insert(0, BlockOptionalBackend())
        import torch
        from torch_linear_assignment import batch_linear_assignment

        class SimulatedCudaTensor(torch.Tensor):
            @property
            def is_cuda(self):
                return True

        cost = torch.tensor([[[4.0, 1.0], [2.0, 3.0]]]).as_subclass(SimulatedCudaTensor)
        expected = torch.tensor([[1, 0]], dtype=torch.long)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = batch_linear_assignment(cost)
            second = batch_linear_assignment(cost)

        assert torch.equal(first, expected)
        assert torch.equal(second, expected)
        assert len(caught) == 1
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_batch_linear_assignment_cuda_fallback_promotes_integer_cost_for_scipy(
    monkeypatch,
):
    """Prevent an unsupported CUDA integer input from bypassing the SciPy oracle."""

    class SimulatedUnsupportedCudaTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    cpu_cost = torch.tensor([[[8, 1, 5], [3, 7, 2], [6, 4, 9]]], dtype=torch.int64)
    cost = cpu_cost.as_subclass(SimulatedUnsupportedCudaTensor)
    expected = scipy_assignment(cpu_cost.to(torch.float32))
    monkeypatch.setattr(assignment_module, "_CUDA_FALLBACK_WARNING_EMITTED", False)

    with pytest.warns(RuntimeWarning, match="Triton linear-assignment support"):
        actual = batch_linear_assignment(cost)

    assert actual.dtype == torch.long
    assert torch.equal(actual, expected)


def test_batch_linear_assignment_cuda_fallback_promotes_bfloat16_cost_for_scipy(
    monkeypatch,
):
    """Prevent unsupported CUDA BF16 inputs from failing at SciPy's NumPy boundary."""

    class SimulatedUnsupportedCudaTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    cpu_cost = torch.tensor(
        [[[8.0, 1.0, 5.0], [3.0, 7.0, 2.0], [6.0, 4.0, 9.0]]],
        dtype=torch.bfloat16,
    )
    cost = cpu_cost.as_subclass(SimulatedUnsupportedCudaTensor)
    expected = scipy_assignment(cpu_cost.to(torch.float32))
    monkeypatch.setattr(assignment_module, "_CUDA_FALLBACK_WARNING_EMITTED", False)

    with pytest.warns(RuntimeWarning, match="Triton linear-assignment support"):
        actual = batch_linear_assignment(cost)

    assert actual.dtype == torch.long
    assert torch.equal(actual, expected)


class TestAssignment(TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_simple(self):
        cost = (
            torch.tensor([8, 4, 7, 5, 2, 3, 9, 6, 7, 9, 4, 8])
            .reshape(1, 4, 3)
            .to(self.device)
        )
        gt_assignment = torch.tensor([0, 2, -1, 1]).reshape(1, 4)
        result = batch_linear_assignment(cost).cpu()
        print(result)
        self.assertTrue((result == gt_assignment).all())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_cuda_equal_to_cpu(self):
        for bs, rows, cols in [(16, 20, 40), (1, 30, 10), (0, 5, 5)]:
            cost = torch.randint(-10, 10, (bs, rows, cols))
            matching_cpu = batch_linear_assignment(cost)
            matching_gpu = batch_linear_assignment(cost.to(self.device)).cpu()
            self.assertEqual(matching_cpu.shape, matching_gpu.shape)
            self.assertEqual(matching_cpu.dtype, matching_gpu.dtype)
            self.assertTrue((matching_cpu == matching_gpu).all())
