"""RF-DETR integration parity on the compiled Triton kernel.

The same comparisons as ``test_rfdetr.py`` with CUDA costs, plus RF-DETR's ``HungarianMatcher`` on
CUDA against the same matcher on the host. On an ineligible GPU the package falls back to SciPy,
which would turn every comparison here into SciPy against SciPy, so each test asserts that the
Triton backend actually solved the batch. Index pairs are not compared: the float32 kernel may
return a different, equally optimal answer, so parity is a valid matching with the oracle's
objective. Install with ``make install-rfdetr``; the Modal GPU lane runs this file.
"""

import importlib
from collections.abc import Callable
from typing import Any

import pytest
import torch

from tests._triton_eligibility import compiled_device_params
from tests.integrations._rfdetr_support import (
    COMPACT_PROBLEMS,
    COST_KINDS,
    RFDETR_GROUP_DETR,
    RFDETR_GROUP_WIDTH,
    CompactProblem,
    CostKind,
    assert_same_objectives,
    assert_valid_matching,
    compact_costs,
    detection_batch,
)

rfdetr_assignment = pytest.importorskip("rfdetr.models._assignment")
rfdetr_matcher = pytest.importorskip("rfdetr.models.matcher")
HungarianMatcher = rfdetr_matcher.HungarianMatcher


def _count_calls(monkeypatch: pytest.MonkeyPatch, owner: object, name: str) -> list[int]:
    """Wrap ``owner.name`` so the test can prove the call was reached; return the call log."""
    calls: list[int] = []
    original: Callable[..., Any] = getattr(owner, name)

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, counted)
    return calls


@pytest.fixture(params=compiled_device_params())
def triton_device(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> tuple[torch.device, list[Any]]:
    """Provide an eligible CUDA device and a log of compiled-backend solves."""
    backend = importlib.import_module("torch_linear_assignment._triton")
    # The kernel is under test here, so keep the small-batch route to SciPy out of the way.
    monkeypatch.setattr(
        importlib.import_module("torch_linear_assignment.assignment"), "_prefers_host", lambda *_: False
    )
    calls: list[Any] = []
    original = backend._solve

    def counted(cost: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        calls.append(tuple(cost.shape[1:]))
        return original(cost, *args, **kwargs)

    monkeypatch.setattr(backend, "_solve", counted)
    return request.param, calls


@pytest.mark.parametrize("kind", COST_KINDS)
@pytest.mark.parametrize("problem", list(COMPACT_PROBLEMS.values()), ids=list(COMPACT_PROBLEMS))
def test_batched_kernel_solve_matches_the_scipy_oracle(
    problem: CompactProblem, kind: CostKind, triton_device: tuple[torch.device, list[Any]]
) -> None:
    """The kernel-backed batched solve returns optimal matchings for every RF-DETR problem shape."""
    device, kernel_calls = triton_device
    host_costs = compact_costs(problem, kind=kind, seed=5)

    actual = rfdetr_assignment.assign_many_bucketed(
        [cost.to(device) for cost in host_costs], list(problem.sizes), problem.group_detr
    )

    if any(problem.sizes):
        widths = set(problem.group_widths)
        if len(widths) > 1 or max(problem.sizes) > min(widths):
            # `_solve` is reached after the wrapper transposes a tall problem, so it always sees
            # (rows, cols) with rows <= cols, which is the sorted pair whichever way round RF-DETR
            # built the bucket.
            shapes = {tuple(sorted((width, size))) for width in widths for size in problem.sizes if size}
            assert set(kernel_calls) == shapes, "a nonempty bucket did not launch the Triton kernel"
        else:
            assert len(kernel_calls) == 1, "the padded batch did not launch the Triton kernel"
    for cost, layer_pairs in zip(host_costs, actual, strict=True):
        expected = HungarianMatcher._assign_compact_cost_matrix(cost, list(problem.sizes), problem.group_detr)
        assert_same_objectives(
            assert_valid_matching(cost, problem.sizes, problem.group_detr, layer_pairs),
            assert_valid_matching(cost, problem.sizes, problem.group_detr, expected),
        )


def _host_and_device_inputs(
    device: torch.device, seed: int
) -> tuple[tuple[int, ...], dict[str, torch.Tensor], list[dict[str, torch.Tensor]], dict[str, Any], list[Any]]:
    sizes = (3, 0, 17, 30)
    outputs, targets = detection_batch(sizes, RFDETR_GROUP_WIDTH * RFDETR_GROUP_DETR, num_classes=91, seed=seed)
    device_outputs = {key: value.to(device) for key, value in outputs.items()}
    device_targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
    return sizes, outputs, targets, device_outputs, device_targets


def test_matcher_forward_on_cuda_matches_the_host_matcher(
    triton_device: tuple[torch.device, list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """RF-DETR's CUDA matcher routes through the batched kernel and stays optimal on the host's costs."""
    device, kernel_calls = triton_device
    sizes, outputs, targets, device_outputs, device_targets = _host_and_device_inputs(device, seed=6)
    matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)
    batched_calls = _count_calls(monkeypatch, rfdetr_assignment, "assign_many_bucketed")

    on_device = matcher(device_outputs, device_targets, group_detr=RFDETR_GROUP_DETR)
    on_host = matcher(outputs, targets, group_detr=RFDETR_GROUP_DETR)

    assert batched_calls == [1], "the CUDA matcher declined its compact batched route"
    assert kernel_calls, "the Triton backend never ran"
    # Costs built on different devices may round differently, so score both answers on the host's.
    cost = matcher._compute_compact_detection_cost_matrix(outputs, targets).float()
    assert_same_objectives(
        assert_valid_matching(cost, sizes, RFDETR_GROUP_DETR, on_device),
        assert_valid_matching(cost, sizes, RFDETR_GROUP_DETR, on_host),
    )


def test_multi_layer_match_on_cuda_matches_the_host_matcher(
    triton_device: tuple[torch.device, list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training matches the final and auxiliary layers in one folded batch; every layer stays optimal."""
    device, kernel_calls = triton_device
    layers = 3
    layer_inputs = [_host_and_device_inputs(device, seed=7 + layer) for layer in range(layers)]
    sizes, _, targets, _, device_targets = layer_inputs[0]
    # Layers share one target list, as in training; only predictions differ per layer.
    host_outputs = [outputs for _, outputs, _, _, _ in layer_inputs]
    device_outputs = [outputs for _, _, _, outputs, _ in layer_inputs]
    matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)
    batched_calls = _count_calls(monkeypatch, rfdetr_assignment, "assign_many_bucketed")

    on_device = matcher._match_many(device_outputs, device_targets, RFDETR_GROUP_DETR)

    assert on_device is not None, "the CUDA matcher declined the folded multi-layer route"
    assert batched_calls == [1], "the folded route did not reach the batched assignment exactly once"
    assert kernel_calls, "the folded route did not launch the Triton kernel"
    for outputs, layer_pairs in zip(host_outputs, on_device, strict=True):
        on_host = matcher(outputs, targets, group_detr=RFDETR_GROUP_DETR)
        cost = matcher._compute_compact_detection_cost_matrix(outputs, targets).float()
        assert_same_objectives(
            assert_valid_matching(cost, sizes, RFDETR_GROUP_DETR, layer_pairs),
            assert_valid_matching(cost, sizes, RFDETR_GROUP_DETR, on_host),
        )
