"""Benchmark SciPy, legacy CUDA, and Triton with exact-parity gates.

The script emits JSON Lines so GPU evidence can be attached to a pull request
without merging measurements across devices. A no-CUDA host emits one
``gpu_skipped`` record and exits successfully; that record is diagnostic and
never performance-acceptance evidence.
"""

import argparse
import importlib.metadata
import json
import math
import platform
import statistics
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch

from torch_linear_assignment.assignment import (
    _batch_linear_assignment_cuda_legacy,
    _load_legacy_backend,
    _load_triton_backend,
    batch_linear_assignment_cpu,
)


_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def _csv_values(value: str) -> list[str]:
    """Split a comma-separated CLI value and reject empty selections."""
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _integer_csv(value: str) -> list[int]:
    """Parse positive comma-separated integers for workload dimensions."""
    try:
        values = [int(item) for item in _csv_values(value)]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("workload dimensions must be positive")
    return values


def _arguments() -> argparse.Namespace:
    """Parse the reproducible Issue 32 benchmark configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        type=_csv_values,
        default=["legacy_cuda", "triton"],
        help="comma-separated: scipy,legacy_cuda,triton",
    )
    parser.add_argument("--workers", type=int, default=300)
    parser.add_argument("--tasks", type=_integer_csv, default=[100, 300, 600])
    parser.add_argument("--batches", type=_integer_csv, default=[1, 16, 50, 208, 624])
    parser.add_argument(
        "--dtypes",
        type=_csv_values,
        default=["float32", "float64"],
    )
    parser.add_argument(
        "--validation-modes",
        type=_csv_values,
        default=["full", "off", "nonfinite_only", "infeasibility_flag_only"],
        help="private Triton validation modes; other backends use full only",
    )
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    unknown_backends = set(arguments.backends) - {
        "scipy",
        "legacy_cuda",
        "triton",
    }
    unknown_dtypes = set(arguments.dtypes) - _DTYPES.keys()
    if unknown_backends:
        parser.error(f"unknown backends: {sorted(unknown_backends)}")
    if unknown_dtypes:
        parser.error(f"unknown dtypes: {sorted(unknown_dtypes)}")
    if arguments.workers <= 0 or arguments.warmup < 0 or arguments.repetitions <= 0:
        parser.error("workers/repetitions must be positive and warmup non-negative")
    return arguments


def _versions() -> dict[str, str | None]:
    """Return environment versions without importing optional Triton eagerly."""
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton_version = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton_version,
    }


def _device_metadata() -> dict[str, str]:
    """Describe the CUDA device used by every record in this process."""
    properties = torch.cuda.get_device_properties(0)
    return {
        "type": "cuda",
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
    }


def _solver_dtype_name(input_dtype: torch.dtype) -> str:
    """Name the documented dtype used by SciPy and Triton parity checks."""
    return "float64" if input_dtype == torch.float64 else "float32"


def _scipy_oracle(cost: torch.Tensor) -> torch.Tensor:
    """Solve promoted CPU costs with the public SciPy-defined semantics."""
    solver_dtype = torch.float64 if cost.dtype == torch.float64 else torch.float32
    return batch_linear_assignment_cpu(cost.to(solver_dtype))


def _backend_callable(
    backend: str,
    validation: str,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Resolve one private comparison adapter without changing public dispatch."""
    if backend == "scipy":
        return lambda cost: _scipy_oracle(cost.cpu())
    if backend == "legacy_cuda":
        return (
            _batch_linear_assignment_cuda_legacy
            if _load_legacy_backend() is not None
            else None
        )

    triton_backend = _load_triton_backend()
    if triton_backend is None:
        return None
    return lambda cost: triton_backend.batch_linear_assignment(
        cost,
        validation=validation,
    )


def _time_call(
    operation: Callable[[torch.Tensor], torch.Tensor],
    cost: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """Measure one call with synchronization only at the sample boundary."""
    if cost.is_cuda:
        torch.cuda.synchronize(cost.device)
    started = time.perf_counter_ns()
    result = operation(cost)
    if cost.is_cuda:
        torch.cuda.synchronize(cost.device)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return elapsed_ms, result


def _nearest_rank_p95(samples: list[float]) -> float:
    """Return the deterministic nearest-rank 95th percentile."""
    ordered = sorted(samples)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _emit(record: dict[str, Any], output: Path | None) -> None:
    """Write one lossless JSONL record to stdout and an optional artifact."""
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if output is not None:
        with output.open("a", encoding="utf-8") as output_file:
            output_file.write(f"{line}\n")


def _status_record(status: str, run_id: str) -> dict[str, Any]:
    """Build a non-measurement record with explicit acceptance exclusion."""
    return {
        "schema_version": 1,
        "record_type": "run",
        "status": status,
        "run_id": run_id,
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "versions": _versions(),
        "acceptance_eligible": False,
    }


def _case_record(
    *,
    run_id: str,
    backend: str,
    validation: str,
    batch: int,
    workers: int,
    tasks: int,
    dtype_name: str,
    seed: int,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    """Measure one backend/workload pair after enforcing exact SciPy parity."""
    dtype = _DTYPES[dtype_name]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cpu_cost = torch.randn((batch, workers, tasks), generator=generator).to(dtype)
    oracle = _scipy_oracle(cpu_cost)
    operation = _backend_callable(backend, validation)
    base = {
        "schema_version": 1,
        "record_type": "case",
        "run_id": run_id,
        "backend": backend,
        "validation_mode": validation,
        "device": _device_metadata(),
        "versions": _versions(),
        "shape": {
            "batch": batch,
            "workers": workers,
            "tasks": tasks,
            "orientation": (
                "transpose"
                if tasks < workers
                else "square"
                if tasks == workers
                else "direct"
            ),
        },
        "input_dtype": dtype_name,
        "solver_dtype": _solver_dtype_name(dtype),
        "seed": seed,
    }
    if operation is None:
        return {
            **base,
            "status": "backend_unavailable",
            "acceptance_eligible": False,
        }

    benchmark_cost = cpu_cost if backend == "scipy" else cpu_cost.cuda()
    torch.cuda.reset_peak_memory_stats() if benchmark_cost.is_cuda else None
    memory_before = (
        torch.cuda.memory_allocated(benchmark_cost.device)
        if benchmark_cost.is_cuda
        else 0
    )
    cold_ms, result = _time_call(operation, benchmark_cost)
    if not torch.equal(result.cpu(), oracle):
        return {
            **base,
            "status": "parity_failed",
            "parity": {"oracle": "scipy_promoted", "exact": False},
            "acceptance_eligible": False,
        }

    for _ in range(warmup):
        operation(benchmark_cost)
    if benchmark_cost.is_cuda:
        torch.cuda.synchronize(benchmark_cost.device)

    samples = [_time_call(operation, benchmark_cost)[0] for _ in range(repetitions)]
    peak_delta = (
        torch.cuda.max_memory_allocated(benchmark_cost.device) - memory_before
        if benchmark_cost.is_cuda
        else None
    )
    return {
        **base,
        "status": "measured",
        "timing": {
            "cold_first_call_ms": cold_ms,
            "cold_method": "process_first_solver_call_includes_jit_load_execution_excludes_import",
            "warm_ms": {
                "n": repetitions,
                "warmup": warmup,
                "median": statistics.median(samples),
                "p95": _nearest_rank_p95(samples),
                "quantile": "nearest_rank",
                "timer": "wall_clock_plus_cuda_sync",
            },
        },
        "memory": {
            "peak_allocated_delta_bytes": peak_delta,
            "logical_workspace_bytes": None,
            "workspace_source": "unavailable",
        },
        "parity": {"oracle": "scipy_promoted", "exact": True},
        "acceptance_eligible": backend in {"legacy_cuda", "triton"},
        "exclusions": [
            "cold timing is not isolated from process or driver caches",
            "logical workspace accounting is unavailable",
        ],
    }


def _validation_modes(backend: str, requested: Iterable[str]) -> list[str]:
    """Return private validation variants only for the Triton backend."""
    return list(requested) if backend == "triton" else ["full"]


def main() -> None:
    """Run the configured matrix and emit structured evidence or skip rows."""
    arguments = _arguments()
    run_id = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    if arguments.output is not None:
        arguments.output.unlink(missing_ok=True)
    if not torch.cuda.is_available():
        _emit(_status_record("gpu_skipped", run_id), arguments.output)
        return

    for backend in arguments.backends:
        for validation in _validation_modes(backend, arguments.validation_modes):
            for batch in arguments.batches:
                for tasks in arguments.tasks:
                    for dtype_name in arguments.dtypes:
                        try:
                            record = _case_record(
                                run_id=run_id,
                                backend=backend,
                                validation=validation,
                                batch=batch,
                                workers=arguments.workers,
                                tasks=tasks,
                                dtype_name=dtype_name,
                                seed=arguments.seed,
                                warmup=arguments.warmup,
                                repetitions=arguments.repetitions,
                            )
                        except (torch.OutOfMemoryError, MemoryError):
                            torch.cuda.empty_cache()
                            record = {
                                **_status_record("oom", run_id),
                                "backend": backend,
                                "validation_mode": validation,
                                "shape": {
                                    "batch": batch,
                                    "workers": arguments.workers,
                                    "tasks": tasks,
                                },
                                "input_dtype": dtype_name,
                            }
                        _emit(record, arguments.output)


if __name__ == "__main__":
    main()
