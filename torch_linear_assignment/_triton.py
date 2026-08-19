"""Triton implementation of batched rectangular linear assignment.

Purpose:
    Provide the private CUDA implementation selected by ``assignment.py`` on
    supported Linux NVIDIA systems. The module owns no public dispatch policy;
    importing it is deliberately deferred until the caller has established that
    CUDA, the Torch version, and the GPU capability are eligible.

Scope:
    One Triton program solves one batch item. The program is intentionally
    correctness-first: outer augmenting-path steps remain sequential while each
    current set of candidate columns is processed as a masked lane vector.
    Per-search state remains program-local. The correctness launch currently
    keeps its loop-carried vectors within one warp; wider launch configurations
    remain gated on exact GPU parity. Its ordering reproduces the legacy CUDA
    and SciPy rectangular-LSAP algorithm.

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
_VALIDATION_MODES: frozenset[str] = frozenset({"full", "off", "nonfinite_only", "infeasibility_flag_only"})


@triton.jit
def _linear_assignment_kernel(
    cost,
    u,
    v,
    col4row,
    row4col,
    infeasible,
    NUM_ROWS: tl.constexpr,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FP64: tl.constexpr,
):
    """Solve one rectangular assignment problem in a single Triton program."""
    batch_index = tl.program_id(0)
    lane = tl.arange(0, BLOCK_N)
    column_mask = lane < NUM_COLUMNS
    row_mask = lane < NUM_ROWS

    cost = cost + batch_index * NUM_ROWS * NUM_COLUMNS
    u = u + batch_index * NUM_ROWS
    v = v + batch_index * NUM_COLUMNS
    col4row = col4row + batch_index * NUM_ROWS
    row4col = row4col + batch_index * NUM_COLUMNS

    for current_row in tl.range(0, NUM_ROWS, num_stages=1):
        scanned_rows = tl.zeros((BLOCK_N,), tl.int1)
        scanned_columns = tl.zeros((BLOCK_N,), tl.int1)
        active_columns = column_mask
        scan_positions = NUM_COLUMNS - lane - 1
        path = tl.full((BLOCK_N,), -1, tl.int32)
        if IS_FP64:
            shortest_path_costs = tl.full((BLOCK_N,), float("inf"), tl.float64)
            scanned_row_costs = tl.zeros((BLOCK_N,), tl.float64)
        else:
            shortest_path_costs = tl.full((BLOCK_N,), float("inf"), tl.float32)
            scanned_row_costs = tl.zeros((BLOCK_N,), tl.float32)

        sink = -1
        current_potential = tl.load(u + current_row)
        # A workspace-derived zero keeps this loop-carried value FP32 or FP64.
        min_value = current_potential - current_potential
        search_row = current_row
        num_remaining = NUM_COLUMNS

        # Wide matrices usually reach an unmatched sink early; do not execute
        # the remaining masked column-search iterations after that point.
        searching = True
        while searching:
            scanned_rows = scanned_rows | (row_mask & (lane == search_row) & searching)
            candidate_mask = active_columns & searching
            previous_cost = shortest_path_costs
            reduced_cost = (
                min_value
                - tl.load(u + search_row)
                + tl.load(
                    cost + search_row * NUM_COLUMNS + lane,
                    mask=candidate_mask,
                    other=float("inf"),
                )
                - tl.load(v + lane, mask=candidate_mask, other=0.0)
            )
            improved = candidate_mask & (reduced_cost < previous_cost)
            candidate_cost = tl.where(improved, reduced_cost, previous_cost)
            shortest_path_costs = candidate_cost
            path = tl.where(improved, search_row, path)

            # The legacy solver scans a swap-removed ``remaining`` array. Track
            # each column's current scan position locally to preserve its exact
            # last-unmatched/first-matched tie rule without shared scratch writes.
            lowest = tl.min(
                tl.where(candidate_mask, candidate_cost, float("inf")),
                axis=0,
            )
            has_candidate = lowest != float("inf")
            step_valid = searching & has_candidate
            tied = candidate_mask & (candidate_cost == lowest)
            matched_rows = tl.load(row4col + lane, mask=candidate_mask, other=0)
            last_unmatched_position = tl.max(
                tl.where(tied & (matched_rows == -1), scan_positions, -1),
                axis=0,
            )
            first_tied_position = tl.min(
                tl.where(tied, scan_positions, NUM_COLUMNS),
                axis=0,
            )
            selected_position = tl.where(
                last_unmatched_position >= 0,
                last_unmatched_position,
                first_tied_position,
            )
            selected_column = tl.max(
                tl.where(candidate_mask & (scan_positions == selected_position), lane, -1),
                axis=0,
            )
            selected_column = tl.where(step_valid, selected_column, 0)
            selected_row = tl.load(row4col + selected_column)

            # A matched row enters the search tree through ``selected_column``.
            # Save that column's shortest cost by row now, avoiding a dynamic
            # gather from the loop-carried column vector after the search.
            matched_step = step_valid & (selected_row != -1)
            scanned_row_costs = tl.where(
                row_mask & (lane == selected_row) & matched_step,
                lowest,
                scanned_row_costs,
            )
            sink = tl.where(step_valid & (selected_row == -1), selected_column, sink)
            search_row = tl.where(
                matched_step,
                selected_row,
                search_row,
            )
            scanned_columns = scanned_columns | (column_mask & (lane == selected_column) & step_valid)
            last_index = num_remaining - 1
            last_column = tl.max(
                tl.where(active_columns & (scan_positions == last_index), lane, -1),
                axis=0,
            )
            scan_positions = tl.where(
                active_columns & (lane == last_column) & step_valid,
                selected_position,
                scan_positions,
            )
            active_columns = active_columns & ~((lane == selected_column) & step_valid)
            num_remaining = tl.where(step_valid, num_remaining - 1, num_remaining)
            min_value = tl.where(step_valid, lowest, min_value)
            tl.store(infeasible + batch_index, 1, mask=searching & ~has_candidate)
            searching = matched_step

        solved = sink != -1
        tl.store(infeasible + batch_index, 1, mask=~solved)

        tl.store(u + current_row, current_potential + min_value, mask=solved)
        visited_rows = scanned_rows
        update_row = solved & row_mask & visited_rows & (lane != current_row)
        tl.store(
            u + lane,
            tl.load(u + lane, mask=row_mask, other=0.0) + min_value - scanned_row_costs,
            mask=update_row,
        )

        update_column = solved & column_mask & scanned_columns
        tl.store(
            v + lane,
            tl.load(v + lane, mask=column_mask, other=0.0) - min_value + shortest_path_costs,
            mask=update_column,
        )

        augmenting = solved
        augmenting_column = sink
        augmentation_steps = 0
        # Recover only the actual path while retaining the row-count safety cap.
        while augmenting & (augmentation_steps < NUM_ROWS):
            safe_column = tl.where(augmenting, augmenting_column, 0)
            augmenting_row = tl.max(
                tl.where(column_mask & (lane == safe_column), path, -1),
                axis=0,
            )
            safe_row = tl.where(augmenting, augmenting_row, 0)
            previous_column = tl.load(col4row + safe_row)
            tl.store(row4col + safe_column, augmenting_row, mask=augmenting)
            tl.store(col4row + safe_row, augmenting_column, mask=augmenting)
            augmenting = augmenting & (augmenting_row != current_row)
            augmenting_column = tl.where(augmenting, previous_column, augmenting_column)
            augmentation_steps += 1

        # The next row consumes the assignments and potentials written above.
        tl.debug_barrier()


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


def _solve(cost: torch.Tensor, validation: ValidationMode) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the one-program-per-batch kernel and return both assignment views."""
    batch_size, rows, columns = cost.shape
    block_n = triton.next_power_of_2(columns)
    workspace_options = {"device": cost.device}
    scalar_options = {**workspace_options, "dtype": cost.dtype}
    integer_options = {**workspace_options, "dtype": torch.int32}
    u = torch.zeros((batch_size, rows), **scalar_options)
    v = torch.zeros((batch_size, columns), **scalar_options)
    col4row = torch.full((batch_size, rows), -1, **integer_options)
    row4col = torch.full((batch_size, columns), -1, **integer_options)
    infeasible = torch.zeros((batch_size,), **integer_options)

    _linear_assignment_kernel[(batch_size,)](
        cost,
        u,
        v,
        col4row,
        row4col,
        infeasible,
        NUM_ROWS=rows,
        NUM_COLUMNS=columns,
        BLOCK_N=block_n,
        IS_FP64=cost.dtype == torch.float64,
        # Multi-warp loop-carried state produces false infeasibility on the
        # saved 512-lane L4 case; keep one warp until a wider design is exact.
        num_warps=1,
    )
    if validation in {"full", "infeasibility_flag_only"} and bool(infeasible.any().item()):
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
