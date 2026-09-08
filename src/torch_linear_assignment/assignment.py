"""Public linear-assignment API with optional CUDA acceleration."""

import importlib
import re
import sys
import warnings
from types import ModuleType

import torch
from scipy.optimize import linear_sum_assignment

_CUDA_FALLBACK_WARNING_EMITTED = False
_TRITON_MINIMUM_TORCH_VERSION = (2, 4)


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
        stacklevel=3,
    )


def batch_linear_assignment_cuda(cost: torch.Tensor) -> torch.Tensor:
    """Solve a CUDA batch through the private Triton implementation."""
    backend = _load_triton_backend()
    if backend is None:
        raise RuntimeError("Triton is unavailable for CUDA linear assignment.")
    return backend.batch_linear_assignment(cost)


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
    if cost.ndim != 3:
        raise ValueError("Need 3-dimensional tensor with shape (B, W, T).")
    if cost.is_complex():
        raise TypeError("Complex costs are not supported.")
    if cost.is_cuda and _cuda_uses_triton(cost):
        return batch_linear_assignment_cuda(cost)

    device = cost.device
    if cost.is_cuda:
        _warn_cuda_fallback()
        cost = cost.cpu()
    return batch_linear_assignment_cpu(cost).to(device)


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
