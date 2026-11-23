"""Public exports for batched linear assignment."""

from .assignment import assignment_to_indices, batch_linear_assignment, batch_linear_assignment_indices

#: Single version declaration; pyproject.toml reads it through setuptools.
__version__ = "0.1.0"

__all__ = ["__version__", "assignment_to_indices", "batch_linear_assignment", "batch_linear_assignment_indices"]
