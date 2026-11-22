"""RF-DETR integration parity on the host path.

RF-DETR folds every ``(layer, image, group)`` matching problem of one training step into a single
padded call to this package (``rfdetr.models._assignment.assign_many_bucketed``). These tests
compare that call with RF-DETR's own per-problem SciPy oracle,
``HungarianMatcher._assign_compact_cost_matrix``, at the shapes RF-DETR trains with.

Both functions are private to RF-DETR, so the ``rfdetr`` dependency group pins it exactly; a break
after a version bump means re-pinning, not a regression here. Off CUDA this package solves with
SciPy, so this file covers the package contract plus RF-DETR's padding and unpacking logic, not the
GPU kernel; ``test_rfdetr_gpu.py`` covers the kernel. Install with ``make install-rfdetr``.
"""

from pathlib import Path

import pytest
import torch

import torch_linear_assignment
from tests.integrations._rfdetr_support import (
    COMPACT_PROBLEMS,
    RFDETR_GROUP_DETR,
    RFDETR_GROUP_WIDTH,
    CompactProblem,
    CostKind,
    assert_same_objectives,
    assert_same_pairs,
    assert_valid_matching,
    compact_costs,
    detection_batch,
)

rfdetr_assignment = pytest.importorskip("rfdetr.models._assignment")
rfdetr_matcher = pytest.importorskip("rfdetr.models.matcher")
HungarianMatcher = rfdetr_matcher.HungarianMatcher


def test_rfdetr_imports_this_checkout_not_a_pinned_release() -> None:
    """Fail when an rfdetr install replaced the package under test with its own pinned release."""
    checkout = Path(__file__).resolve().parents[2]
    origin = Path(torch_linear_assignment.__file__).resolve()
    expected = checkout / "src" / "torch_linear_assignment" / "__init__.py"
    assert origin == expected, f"torch_linear_assignment imported from {origin}, not {expected}"


def test_checkout_check_rejects_a_wheel_inside_the_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject an installed wheel even when its environment is inside the checkout."""
    checkout = Path(__file__).resolve().parents[2]
    origin = checkout / ".venv" / "lib" / "site-packages" / "torch_linear_assignment" / "__init__.py"
    monkeypatch.setattr(torch_linear_assignment, "__file__", str(origin))

    with pytest.raises(AssertionError, match="not"):
        test_rfdetr_imports_this_checkout_not_a_pinned_release()


def test_float64_cost_preserves_a_small_assignment_difference() -> None:
    """Preserve a float64 optimum that disappears when costs are narrowed to float32."""
    cost = torch.tensor([[0.0, 1.0], [1.0, 2.0 + 1e-8]], dtype=torch.float64)

    actual = rfdetr_assignment.assign_many_bucketed([cost], [2], 1)[0][0]

    assert actual[0].tolist() == [0, 1]
    assert actual[1].tolist() == [1, 0]


@pytest.mark.parametrize("problem", list(COMPACT_PROBLEMS.values()), ids=list(COMPACT_PROBLEMS))
def test_batched_solve_returns_the_scipy_oracle_pairs(problem: CompactProblem) -> None:
    """Continuous costs have one optimum, so the batched solve must return the oracle's exact pairs."""
    costs = compact_costs(problem, kind="continuous", seed=1)

    actual = rfdetr_assignment.assign_many_bucketed(costs, list(problem.sizes), problem.group_detr)

    assert len(actual) == len(costs)
    for cost, layer_pairs in zip(costs, actual, strict=True):
        expected = HungarianMatcher._assign_compact_cost_matrix(cost, list(problem.sizes), problem.group_detr)
        assert_valid_matching(cost, problem.sizes, problem.group_detr, layer_pairs)
        assert_same_pairs(layer_pairs, expected)


@pytest.mark.parametrize("kind", ["tied", "constant"])
@pytest.mark.parametrize("problem", list(COMPACT_PROBLEMS.values()), ids=list(COMPACT_PROBLEMS))
def test_batched_solve_is_optimal_on_tied_costs(problem: CompactProblem, kind: CostKind) -> None:
    """Tied costs admit many optima, and padding can change SciPy's tie-break, so compare objectives."""
    costs = compact_costs(problem, kind=kind, seed=2)

    actual = rfdetr_assignment.assign_many_bucketed(costs, list(problem.sizes), problem.group_detr)

    for cost, layer_pairs in zip(costs, actual, strict=True):
        expected = HungarianMatcher._assign_compact_cost_matrix(cost, list(problem.sizes), problem.group_detr)
        assert_same_objectives(
            assert_valid_matching(cost, problem.sizes, problem.group_detr, layer_pairs),
            assert_valid_matching(cost, problem.sizes, problem.group_detr, expected),
        )


def test_matcher_forward_on_host_matches_the_batched_solve() -> None:
    """The host matcher and the batched solve agree on the cost the matcher itself builds."""
    sizes = (3, 0, 17, 30)
    num_queries = RFDETR_GROUP_WIDTH * RFDETR_GROUP_DETR
    outputs, targets = detection_batch(sizes, num_queries, num_classes=91, seed=3)
    matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)

    matched = matcher(outputs, targets, group_detr=RFDETR_GROUP_DETR)

    cost = matcher._compute_compact_detection_cost_matrix(outputs, targets).float()
    batched = rfdetr_assignment.assign_many_bucketed([cost], list(sizes), RFDETR_GROUP_DETR)[0]
    assert_same_pairs(batched, matched)


def test_batched_solve_accepts_non_float32_costs() -> None:
    """RF-DETR casts to float32, but the package contract also covers half and double costs."""
    problem = COMPACT_PROBLEMS["rfdetr-step"]
    for dtype in (torch.float16, torch.bfloat16, torch.float64):
        costs = [cost.to(dtype) for cost in compact_costs(problem, kind="continuous", seed=4)]

        actual = rfdetr_assignment.assign_many_bucketed(costs, list(problem.sizes), problem.group_detr)

        for cost, layer_pairs in zip(costs, actual, strict=True):
            # Every half value is exact in float64, so the oracle sees the costs the solver sees.
            expected = HungarianMatcher._assign_compact_cost_matrix(
                cost.double(), list(problem.sizes), problem.group_detr
            )
            assert_same_objectives(
                assert_valid_matching(cost, problem.sizes, problem.group_detr, layer_pairs),
                assert_valid_matching(cost, problem.sizes, problem.group_detr, expected),
            )
