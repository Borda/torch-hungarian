"""Triton implementation of batched rectangular linear assignment.

Purpose:
    Provide the private CUDA implementation selected by ``assignment.py`` on
    supported Linux NVIDIA systems. The module owns no public dispatch policy;
    importing it is deliberately deferred until the caller has established that
    CUDA, the Torch version, and the GPU capability are eligible.

Scope:
    One Triton program solves one batch item. The program is intentionally
    correctness-first: outer augmenting-path steps remain sequential while each
    current set of candidate columns is processed as a masked lane vector. Its
    ordering reproduces the legacy CUDA and SciPy rectangular-LSAP algorithm.

Usage:
    ``batch_linear_assignment(cuda_cost)`` accepts a three-dimensional CUDA
    cost tensor and returns the stable public ``(B, W)`` long assignment shape.
    The optional validation setting is private benchmark instrumentation; normal
    dispatch always uses the default full validation.

Outputs:
    The kernel writes assignment and temporary Torch workspaces on the input
    device. The wrapper returns worker-to-task indices, using ``-1`` for an
    unmatched worker, and raises ordinary ``ValueError`` exceptions for invalid
    numeric values or infeasible matrices.

Failure:
    NaN and negative infinity are rejected by one host reduction before launch.
    The kernel records one infeasibility flag per batch item instead of asserting
    on device; the normal wrapper synchronizes once to turn any flag into an
    exception. Triton import, compilation, and GPU execution failures are not
    swallowed because they do not mean the optional dependency is merely absent.

Used by:
    ``torch_linear_assignment.assignment.batch_linear_assignment_cuda``. The
    module is private so its benchmark validation switch and workspace layout do
    not become part of the package API.
"""

from typing import Literal

import torch
import triton
import triton.language as tl


ValidationMode = Literal["full", "off", "nonfinite_only", "infeasibility_flag_only"]
_VALIDATION_MODES: frozenset[str] = frozenset(
    {"full", "off", "nonfinite_only", "infeasibility_flag_only"}
)


@triton.jit
def _linear_assignment_kernel(
    cost,
    u,
    v,
    shortest_path_costs,
    path,
    col4row,
    row4col,
    scanned_rows,
    scanned_columns,
    remaining,
    infeasible,
    NUM_ROWS: tl.constexpr,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Solve one rectangular assignment problem in a single Triton program."""
    batch_index = tl.program_id(0)
    lane = tl.arange(0, BLOCK_N)
    column_mask = lane < NUM_COLUMNS
    row_mask = lane < NUM_ROWS

    cost = cost + batch_index * NUM_ROWS * NUM_COLUMNS
    u = u + batch_index * NUM_ROWS
    v = v + batch_index * NUM_COLUMNS
    shortest_path_costs = shortest_path_costs + batch_index * NUM_COLUMNS
    path = path + batch_index * NUM_COLUMNS
    col4row = col4row + batch_index * NUM_ROWS
    row4col = row4col + batch_index * NUM_COLUMNS
    scanned_rows = scanned_rows + batch_index * NUM_ROWS
    scanned_columns = scanned_columns + batch_index * NUM_COLUMNS
    remaining = remaining + batch_index * NUM_COLUMNS

    for current_row in tl.range(0, NUM_ROWS, num_stages=1):
        tl.store(scanned_rows + lane, 0, mask=row_mask)
        tl.store(scanned_columns + lane, 0, mask=column_mask)
        tl.store(remaining + lane, NUM_COLUMNS - lane - 1, mask=column_mask)
        tl.store(shortest_path_costs + lane, float("inf"), mask=column_mask)

        sink = -1
        min_value = 0.0
        search_row = current_row
        num_remaining = NUM_COLUMNS

        for _ in tl.range(0, NUM_COLUMNS, num_stages=1):
            searching = sink == -1
            tl.store(scanned_rows + search_row, 1, mask=searching)
            candidate_mask = column_mask & (lane < num_remaining) & searching
            columns = tl.load(remaining + lane, mask=column_mask, other=0)
            previous_cost = tl.load(
                shortest_path_costs + columns,
                mask=candidate_mask,
                other=float("inf"),
            )
            reduced_cost = (
                min_value
                - tl.load(u + search_row)
                + tl.load(
                    cost + search_row * NUM_COLUMNS + columns,
                    mask=candidate_mask,
                    other=float("inf"),
                )
                - tl.load(v + columns, mask=candidate_mask, other=0.0)
            )
            improved = candidate_mask & (reduced_cost < previous_cost)
            candidate_cost = tl.where(improved, reduced_cost, previous_cost)
            tl.store(shortest_path_costs + columns, candidate_cost, mask=candidate_mask)
            tl.store(path + columns, search_row, mask=improved)

            # ``remaining`` is scan order, not column order. Among equal minima,
            # the original solver selects the last unmatched lane, otherwise the
            # first tied lane; swap-removal below preserves that evolving order.
            lowest = tl.min(
                tl.where(candidate_mask, candidate_cost, float("inf")),
                axis=0,
            )
            has_candidate = lowest != float("inf")
            step_valid = searching & has_candidate
            tied = candidate_mask & (candidate_cost == lowest)
            matched_rows = tl.load(row4col + columns, mask=candidate_mask, other=0)
            last_unmatched = tl.max(
                tl.where(tied & (matched_rows == -1), lane, -1),
                axis=0,
            )
            first_tied = tl.min(
                tl.where(tied, lane, NUM_COLUMNS),
                axis=0,
            )
            selected_index = tl.where(last_unmatched >= 0, last_unmatched, first_tied)
            safe_selected_index = tl.where(step_valid, selected_index, 0)
            selected_column = tl.load(remaining + safe_selected_index)
            selected_row = tl.load(row4col + selected_column)

            sink = tl.where(step_valid & (selected_row == -1), selected_column, sink)
            search_row = tl.where(
                step_valid & (selected_row != -1),
                selected_row,
                search_row,
            )
            tl.store(scanned_columns + selected_column, 1, mask=step_valid)
            last_index = num_remaining - 1
            last_column = tl.load(
                remaining + tl.where(step_valid, last_index, 0),
            )
            tl.store(remaining + safe_selected_index, last_column, mask=step_valid)
            num_remaining = tl.where(step_valid, num_remaining - 1, num_remaining)
            min_value = tl.where(step_valid, lowest, min_value)
            tl.store(infeasible + batch_index, 1, mask=searching & ~has_candidate)

        solved = sink != -1
        tl.store(infeasible + batch_index, 1, mask=~solved)

        current_potential = tl.load(u + current_row)
        tl.store(u + current_row, current_potential + min_value, mask=solved)
        visited_rows = tl.load(scanned_rows + lane, mask=row_mask, other=0) != 0
        assigned_columns = tl.load(col4row + lane, mask=row_mask, other=0)
        update_row = solved & row_mask & visited_rows & (lane != current_row)
        safe_assigned_column = tl.where(update_row, assigned_columns, 0)
        row_shortest = tl.load(shortest_path_costs + safe_assigned_column)
        tl.store(
            u + lane,
            tl.load(u + lane, mask=row_mask, other=0.0) + min_value - row_shortest,
            mask=update_row,
        )

        visited_columns = (
            tl.load(scanned_columns + lane, mask=column_mask, other=0) != 0
        )
        update_column = solved & column_mask & visited_columns
        column_shortest = tl.load(
            shortest_path_costs + lane, mask=column_mask, other=0.0
        )
        tl.store(
            v + lane,
            tl.load(v + lane, mask=column_mask, other=0.0)
            - min_value
            + column_shortest,
            mask=update_column,
        )

        augmenting = solved
        augmenting_column = sink
        for _ in tl.range(0, NUM_ROWS, num_stages=1):
            safe_column = tl.where(augmenting, augmenting_column, 0)
            augmenting_row = tl.load(path + safe_column)
            safe_row = tl.where(augmenting, augmenting_row, 0)
            previous_column = tl.load(col4row + safe_row)
            tl.store(row4col + safe_column, augmenting_row, mask=augmenting)
            tl.store(col4row + safe_row, augmenting_column, mask=augmenting)
            augmenting = augmenting & (augmenting_row != current_row)
            augmenting_column = tl.where(augmenting, previous_column, augmenting_column)


def _validate_mode(validation: ValidationMode) -> None:
    """Reject an unsupported private validation configuration."""
    if validation not in _VALIDATION_MODES:
        raise ValueError(f"Unknown Triton validation mode: {validation!r}.")


def _reject_invalid_numeric_entries(cost: torch.Tensor) -> None:
    """Raise the SciPy-compatible error for NaN or negative-infinite costs."""
    if not cost.is_floating_point():
        return
    invalid = torch.logical_or(torch.isnan(cost), torch.isneginf(cost)).any()
    if bool(invalid.item()):
        raise ValueError("matrix contains invalid numeric entries")


def _solver_cost_dtype(cost: torch.Tensor) -> torch.Tensor:
    """Preserve float64 costs and promote every other solver input to float32."""
    if cost.dtype == torch.float64:
        return cost
    return cost.to(torch.float32)


def _empty_assignment(cost: torch.Tensor) -> torch.Tensor:
    """Return the public assignment shape without launching a zero-size kernel."""
    batch_size, workers, _ = cost.shape
    return torch.full((batch_size, workers), -1, dtype=torch.long, device=cost.device)


def _solve(
    cost: torch.Tensor, validation: ValidationMode
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the one-program-per-batch kernel and return both assignment views."""
    batch_size, rows, columns = cost.shape
    block_n = triton.next_power_of_2(columns)
    workspace_options = {"device": cost.device}
    scalar_options = {**workspace_options, "dtype": cost.dtype}
    integer_options = {**workspace_options, "dtype": torch.int32}
    u = torch.zeros((batch_size, rows), **scalar_options)
    v = torch.zeros((batch_size, columns), **scalar_options)
    shortest_path_costs = torch.empty((batch_size, columns), **scalar_options)
    path = torch.full((batch_size, columns), -1, **integer_options)
    col4row = torch.full((batch_size, rows), -1, **integer_options)
    row4col = torch.full((batch_size, columns), -1, **integer_options)
    scanned_rows = torch.empty((batch_size, rows), **integer_options)
    scanned_columns = torch.empty((batch_size, columns), **integer_options)
    remaining = torch.empty((batch_size, columns), **integer_options)
    infeasible = torch.zeros((batch_size,), **integer_options)

    _linear_assignment_kernel[(batch_size,)](
        cost,
        u,
        v,
        shortest_path_costs,
        path,
        col4row,
        row4col,
        scanned_rows,
        scanned_columns,
        remaining,
        infeasible,
        NUM_ROWS=rows,
        NUM_COLUMNS=columns,
        BLOCK_N=block_n,
        num_warps=4,
    )
    if validation in {"full", "infeasibility_flag_only"} and bool(
        infeasible.any().item()
    ):
        raise ValueError("cost matrix is infeasible")
    return col4row, row4col


def batch_linear_assignment(
    cost: torch.Tensor,
    *,
    validation: ValidationMode = "full",
) -> torch.Tensor:
    """Solve a CUDA batch with full validation unless private benchmarks opt out."""
    _validate_mode(validation)
    if cost.ndim != 3:
        raise ValueError("Need 3-dimensional tensor with shape (B, W, T).")
    if not cost.is_cuda:
        raise ValueError("Triton linear assignment requires a CUDA tensor.")
    if validation in {"full", "nonfinite_only"}:
        _reject_invalid_numeric_entries(cost)

    cost = _solver_cost_dtype(cost)
    batch_size, workers, tasks = cost.shape
    if batch_size == 0 or workers == 0 or tasks == 0:
        return _empty_assignment(cost)

    with torch.amp.autocast("cuda", enabled=False):
        if tasks < workers:
            _, row4col = _solve(cost.transpose(1, 2).contiguous(), validation)
            return row4col.to(torch.long)
        col4row, _ = _solve(cost.contiguous(), validation)
        return col4row.to(torch.long)
