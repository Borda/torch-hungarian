"""Shared RF-DETR problem builders and parity checks for the RF-DETR integration tests.

RF-DETR solves one ``[num_queries, sum(sizes)]`` "compact" cost matrix per decoder layer: the
queries split into ``group_detr`` groups of ``group_width`` rows, and the columns are the images'
target blocks side by side. Image ``b`` in group ``g`` is the problem
``cost[g * group_width : (g + 1) * group_width, offset_b : offset_b + sizes[b]]``. Both solvers
under comparison return, per image, row indices into all ``num_queries`` rows and column indices
local to that image's target block.
"""

from dataclasses import dataclass
from typing import Literal

import torch

#: Queries per group and number of groups used by RF-DETR training defaults.
RFDETR_GROUP_WIDTH = 300
RFDETR_GROUP_DETR = 13

#: Cost families: ``continuous`` has one optimum, ``tied`` many, ``constant`` makes every
#: matching optimal.
CostKind = Literal["continuous", "tied", "constant"]
COST_KINDS: tuple[CostKind, ...] = ("continuous", "tied", "constant")


@dataclass(frozen=True)
class ProblemScore:
    """Objective of one ``(image, group)`` matching, in float64, with its float32 rounding scale."""

    objective: float
    magnitude: float
    matches: int


@dataclass(frozen=True)
class CompactProblem:
    """One RF-DETR matcher step: per-layer group widths, per-image target counts, group count."""

    group_widths: tuple[int, ...]
    sizes: tuple[int, ...]
    group_detr: int


#: Shapes RF-DETR reaches in training, plus the two inputs that force its bucketed fallback.
COMPACT_PROBLEMS = {
    # Final + two auxiliary layers, mixed target counts including an empty image.
    "rfdetr-step": CompactProblem((RFDETR_GROUP_WIDTH,) * 3, (0, 1, 5, 12, 30, 3), RFDETR_GROUP_DETR),
    "single-layer-single-image": CompactProblem((RFDETR_GROUP_WIDTH,), (7,), RFDETR_GROUP_DETR),
    "one-group": CompactProblem((RFDETR_GROUP_WIDTH,) * 2, (2, 30, 17), 1),
    "every-image-empty": CompactProblem((RFDETR_GROUP_WIDTH,) * 2, (0, 0), RFDETR_GROUP_DETR),
    # More targets than queries per group: padding cannot separate real columns, so RF-DETR
    # solves one bucket per distinct shape instead, including wide problems.
    "more-targets-than-group-width": CompactProblem((4, 4), (6, 1, 0, 4), RFDETR_GROUP_DETR),
    # Layers with different query counts cannot share one padded batch either.
    "mixed-query-counts": CompactProblem((RFDETR_GROUP_WIDTH, 200), (3, 9), RFDETR_GROUP_DETR),
}


def compact_costs(problem: CompactProblem, *, kind: CostKind, seed: int) -> list[torch.Tensor]:
    """Build one seeded float32 compact cost matrix per layer.

    Continuous costs have a unique optimum with probability one, so index pairs are comparable.
    Tied costs draw from three integer levels and constant costs are all equal, so many optima
    exist and only the objective is comparable.
    """
    generator = torch.Generator().manual_seed(seed)
    total_targets = sum(problem.sizes)
    costs = []
    for group_width in problem.group_widths:
        shape = (group_width * problem.group_detr, total_targets)
        if kind == "constant":
            costs.append(torch.full(shape, 0.5, dtype=torch.float32))
        elif kind == "tied":
            costs.append(torch.randint(0, 3, shape, generator=generator).to(torch.float32))
        else:
            costs.append(torch.rand(shape, generator=generator, dtype=torch.float32) * 10)
    return costs


def _offsets(sizes: tuple[int, ...] | list[int]) -> list[int]:
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    return offsets


def assert_valid_matching(
    cost: torch.Tensor,
    sizes: tuple[int, ...] | list[int],
    group_detr: int,
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[ProblemScore]:
    """Check one layer's per-image pairs form a full matching in every group; score each problem.

    Returns:
        The score of every ``(image, group)`` problem, image-major.
    """
    assert len(pairs) == len(sizes)
    group_width = cost.shape[0] // group_detr
    offsets = _offsets(sizes)
    cost64 = cost.detach().to(device="cpu", dtype=torch.float64)
    objectives = []
    for image_index, (rows, cols) in enumerate(pairs):
        assert rows.device.type == "cpu" and cols.device.type == "cpu"
        assert rows.dtype == torch.int64 and cols.dtype == torch.int64
        assert rows.shape == cols.shape and rows.ndim == 1
        size = sizes[image_index]
        assert rows.numel() == group_detr * min(group_width, size)
        for group_index in range(group_detr):
            in_group = (rows >= group_index * group_width) & (rows < (group_index + 1) * group_width)
            group_rows, group_cols = rows[in_group], cols[in_group]
            assert group_rows.numel() == min(group_width, size)
            assert group_rows.unique().numel() == group_rows.numel(), "a query is matched twice"
            assert group_cols.unique().numel() == group_cols.numel(), "a target is matched twice"
            assert bool(((group_cols >= 0) & (group_cols < size)).all()), "target index out of range"
            block = cost64[:, offsets[image_index] : offsets[image_index + 1]]
            matched = block[group_rows, group_cols]
            objectives.append(ProblemScore(float(matched.sum()), float(matched.abs().sum()), matched.numel()))
    return objectives


def assert_same_objectives(actual: list[ProblemScore], expected: list[ProblemScore]) -> None:
    """Require equally optimal answers per problem, up to float32 summation rounding.

    A float32 solver cannot tell apart two matchings whose objectives differ by less than the
    rounding of a float32 sum over the matched entries, ``matches * eps * sum(|entry|)``. Anything
    beyond that bound is a strictly worse answer, not a tie.
    """
    assert len(actual) == len(expected)
    epsilon = torch.finfo(torch.float32).eps
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        tolerance = max(got.matches, 1) * epsilon * max(got.magnitude, want.magnitude)
        assert abs(got.objective - want.objective) <= tolerance, (
            f"problem {index}: objective {got.objective!r} != oracle {want.objective!r} (tolerance {tolerance:.3g})"
        )


def assert_same_pairs(
    actual: list[tuple[torch.Tensor, torch.Tensor]],
    expected: list[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """Require identical per-image index pairs, in RF-DETR's group-concatenated order."""
    assert len(actual) == len(expected)
    for image_index, ((rows, cols), (want_rows, want_cols)) in enumerate(zip(actual, expected, strict=True)):
        assert torch.equal(rows, want_rows), f"image {image_index}: rows differ"
        assert torch.equal(cols, want_cols), f"image {image_index}: cols differ"


def detection_batch(
    sizes: tuple[int, ...], num_queries: int, num_classes: int, seed: int
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Build seeded CPU matcher inputs with valid normalized ``cxcywh`` boxes."""
    generator = torch.Generator().manual_seed(seed)

    def boxes(count: tuple[int, ...]) -> torch.Tensor:
        centers = torch.rand((*count, 2), generator=generator) * 0.6 + 0.2
        extents = torch.rand((*count, 2), generator=generator) * 0.25 + 0.05
        return torch.cat((centers, extents), dim=-1)

    outputs = {
        "pred_logits": torch.randn((len(sizes), num_queries, num_classes), generator=generator),
        "pred_boxes": boxes((len(sizes), num_queries)),
    }
    targets = [
        {"labels": torch.randint(0, num_classes, (size,), generator=generator), "boxes": boxes((size,))}
        for size in sizes
    ]
    return outputs, targets
