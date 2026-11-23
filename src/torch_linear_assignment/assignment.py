"""Public linear-assignment API with optional CUDA acceleration."""

import importlib
import re
import sys
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Any, Literal

import torch
from scipy.optimize import linear_sum_assignment

_CUDA_FALLBACK_WARNING_EMITTED = False
_TRITON_MINIMUM_TORCH_VERSION = (2, 4)

#: How much of the Triton solve's input checking ``batch_linear_assignment_indices`` performs.
ValidateMode = Literal["full", "off"]
_VALIDATE_MODES = ("full", "off")


def _prepare_solver_cost(cost: torch.Tensor) -> torch.Tensor:
    """Apply the documented solver dtype without discarding complex values."""
    if cost.dtype in {torch.float32, torch.float64} or cost.is_complex():
        return cost
    return cost.to(torch.float32)


def batch_linear_assignment_cpu(cost: torch.Tensor) -> torch.Tensor:
    """Solve CPU costs with SciPy without retaining their autograd graph."""
    cost = _prepare_solver_cost(cost.detach())
    batch_size, workers, _ = cost.shape
    matching = torch.full([batch_size, workers], -1, dtype=torch.long, device=cost.device)
    for batch_index in range(batch_size):
        row_indices, column_indices = linear_sum_assignment(cost[batch_index].numpy(), maximize=False)
        matching[batch_index].scatter_(
            0,
            torch.from_numpy(row_indices),
            torch.from_numpy(column_indices),
        )
    return matching


def _torch_supports_triton() -> bool:
    """Return whether the installed Torch version meets the Triton-path floor."""
    version = re.match(r"(\d+)\.(\d+)", torch.__version__)
    return version is not None and tuple(map(int, version.groups())) >= _TRITON_MINIMUM_TORCH_VERSION


def _load_triton_backend() -> ModuleType | None:
    """Load the optional Triton module without hiding nested import failures."""
    try:
        return importlib.import_module("torch_linear_assignment._triton")
    except ModuleNotFoundError as error:
        if error.name in {"torch_linear_assignment._triton", "triton"}:
            return None
        raise


def _cuda_uses_triton(cost: torch.Tensor) -> bool:
    """Return whether this CUDA cost tensor can use the supported Triton path."""
    if (
        sys.platform != "linux"
        or not _torch_supports_triton()
        or not torch.cuda.is_available()
        or torch.version.cuda is None
    ):
        return False
    return torch.cuda.get_device_capability(cost.device) >= (8, 0) and _load_triton_backend() is not None


def _warn_cuda_fallback() -> None:
    """Warn once when a CUDA input must use the SciPy CPU fallback."""
    global _CUDA_FALLBACK_WARNING_EMITTED
    if _CUDA_FALLBACK_WARNING_EMITTED:
        return
    _CUDA_FALLBACK_WARNING_EMITTED = True
    warnings.warn(
        "Triton linear-assignment support is unavailable for this CUDA input; using SciPy on CPU.",
        RuntimeWarning,
        stacklevel=4,
    )


def _run_on_cost_device(cost: torch.Tensor, solve: Callable[[], Any]) -> Any:
    """Call ``solve`` with the cost's GPU current; the Triton launch goes to the current device.

    The guard is skipped for the common case, where the device is already current.
    """
    if cost.device.index == torch.cuda.current_device():
        return solve()
    with torch.cuda.device(cost.device):
        return solve()


def _require_triton_backend() -> ModuleType:
    """Return the Triton module, or raise when the caller bypassed the eligibility check."""
    backend = _load_triton_backend()
    if backend is None:
        raise RuntimeError("Triton is unavailable for CUDA linear assignment.")
    return backend


def batch_linear_assignment_cuda(cost: torch.Tensor) -> torch.Tensor:
    """Solve a CUDA batch through the private Triton implementation."""
    backend = _require_triton_backend()
    return _run_on_cost_device(cost, lambda: backend.batch_linear_assignment(cost))


def _prefers_host(batch_size: int, workers: int, tasks: int) -> bool:
    """Return whether a batch this small is faster solved by SciPy than launched on the GPU.

    One problem, two problems up to 64 wide, and up to four problems up to 32 wide lose to the host
    solve even after the transfers; larger batches amortise the launch.
    """
    larger_side = max(workers, tasks)
    return batch_size == 1 or (batch_size == 2 and larger_side <= 64) or (batch_size <= 4 and larger_side <= 32)


def _check_cost(cost: torch.Tensor) -> None:
    """Raise the documented errors for a cost of the wrong rank or a complex dtype."""
    if cost.ndim != 3:
        raise ValueError("Need 3-dimensional tensor with shape (B, W, T).")
    if cost.is_complex():
        raise TypeError("Complex costs are not supported.")


def _uses_triton(cost: torch.Tensor) -> bool:
    """Return whether this cost goes to the Triton solve, warning once when a CUDA cost cannot.

    A batch small enough for SciPy to win is routed to the host on purpose, without a warning.
    """
    if not cost.is_cuda:
        return False
    if not _cuda_uses_triton(cost):
        _warn_cuda_fallback()
        return False
    return not _prefers_host(*cost.shape)


def _solve_on_host(cost: torch.Tensor) -> torch.Tensor:
    """Solve with SciPy on host memory, so any non-CPU device, such as CUDA, MPS or XPU, is copied first."""
    if cost.device.type != "cpu":
        cost = cost.cpu()
    return batch_linear_assignment_cpu(cost)


def batch_linear_assignment(cost: torch.Tensor) -> torch.Tensor:
    """Solve a batch of linear assignment problems.

    The method minimizes real-valued costs. Costs may require gradients;
    the returned discrete assignment is not differentiable.

    Args:
      cost: Cost matrix with shape (B, W, T), where W is the number of workers
            and T is the number of tasks.

    Returns:
      Matching tensor with shape (B, W), with assignments for each worker. If the
      task was not assigned, the corresponding index will be -1.

    Raises:
      TypeError: If costs have a complex dtype.
      ValueError: If costs have the wrong rank, invalid numeric entries, or no
        feasible matching.
    """
    _check_cost(cost)
    if _uses_triton(cost):
        return batch_linear_assignment_cuda(cost)

    # SciPy works on host memory, so any other device, such as MPS or XPU, is solved on the host
    # and the assignment is returned on the input's device.
    return _solve_on_host(cost).to(cost.device)


def batch_linear_assignment_indices(
    cost: torch.Tensor,
    *,
    validate: ValidateMode = "full",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve a batch and return the matched pairs as ``(rows, cols)`` of a fixed shape.

    The pairs follow SciPy's ``linear_sum_assignment``: ``rows`` is sorted within each problem and
    ``cols[b, i]`` is the task assigned to worker ``rows[b, i]``. Both tensors are ``torch.long``,
    live on the input's device, and have shape ``(B, min(W, T))``, so no output size depends on the
    data. On the Triton path, converting the solver's maps to pairs adds no host reads.

    Args:
      cost: Cost matrix with shape (B, W, T), where W is the number of workers
            and T is the number of tasks.
      validate: On the Triton path, ``"full"`` checks floating-point input for NaN and ``-inf``
            before launch and checks for infeasibility after a solve. Each check reads back to
            the host. Integer inputs skip the first check, and empty inputs skip the solve.
            ``"off"`` skips both checks and permits CUDA graph capture after warm-up when the
            input routes to Triton. Invalid numeric entries or infeasible costs then produce
            unspecified indices. SciPy-backed calls always validate, and CUDA calls routed
            to SciPy require a host transfer.

    Returns:
      The pair ``(rows, cols)``, each with shape (B, min(W, T)).

    Raises:
      TypeError: If costs have a complex dtype.
      ValueError: If ``validate`` is unknown or costs have the wrong rank. On checked paths, also
        raised for invalid numeric entries or an infeasible matching.

    Examples:
      >>> cost = torch.tensor([[[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]]])
      >>> rows, cols = batch_linear_assignment_indices(cost)
      >>> rows.tolist(), cols.tolist()
      ([[0, 1, 2]], [[1, 0, 2]])
    """
    if not isinstance(validate, str) or validate not in _VALIDATE_MODES:
        raise ValueError(f"Unknown validate mode: {validate!r}.")
    _check_cost(cost)
    if _uses_triton(cost):
        backend = _require_triton_backend()
        return _run_on_cost_device(cost, lambda: backend.batch_linear_assignment_indices(cost, validation=validate))

    batch_size, workers, tasks = cost.shape
    assignment = _solve_on_host(cost)
    if batch_size == 0 or workers == 0 or tasks == 0:
        rows = torch.zeros((batch_size, min(workers, tasks)), dtype=torch.long, device=cost.device)
        return rows, rows.clone()
    rows, cols = assignment_to_indices(assignment)
    return rows.to(cost.device), cols.to(cost.device)


def assignment_to_indices(
    assignment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert assignment to the SciPy format.

    Args:
        assignment: The assignment with shape (B, W).

    Returns:
        row_ind, col_ind: An array of row indices and one of corresponding column indices
            giving the optimal assignment, each with shape (B, K).

    Raises:
        ValueError if batch assignments have different sizes.
    """
    batch_size = assignment.shape[0]
    if batch_size == 0:
        indices = torch.zeros(0, 0, dtype=torch.long, device=assignment.device)
        return indices, indices
    mask = assignment >= 0
    n_matches = mask.sum(1)
    if (n_matches[1:] != n_matches[0]).any():
        raise ValueError("Inconsistent matching sizes.")
    n_matches = n_matches[0].item()
    row_ind = mask.nonzero()[:, 1].reshape(batch_size, n_matches)
    col_ind = assignment.masked_select(mask).reshape(batch_size, n_matches)
    return row_ind, col_ind
