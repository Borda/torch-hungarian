"""Benchmark SciPy CPU and installed CUDA backends with exact-parity gates.

The script saves lossless JSON Lines and prints a compact human-readable table.
This keeps GPU evidence attachable to a pull request without merging
measurements across devices. A no-CUDA host emits one ``gpu_skipped`` record
and exits successfully; that record is diagnostic and never
performance-acceptance evidence. Any solver mismatch emits replayable
diagnostics and makes the process exit unsuccessfully.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import statistics
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from torch_linear_assignment import batch_linear_assignment

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
        default=["scipy", "public_cuda"],
        help="comma-separated: scipy,public_cuda,legacy_cuda,triton",
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
    parser.add_argument(
        "--label",
        help="optional label identifying the installed package or comparison run",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        help="JSONL artifact path (default: benchmark-results-<run-id>.jsonl)",
    )
    arguments = parser.parse_args()

    unknown_backends = set(arguments.backends) - {
        "scipy",
        "public_cuda",
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
    """Return package and environment versions without importing optional Triton eagerly."""
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton_version = None
    try:
        package_version = importlib.metadata.version("torch-linear-assignment")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    return {
        "package": package_version,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton_version,
    }


def _device_metadata(device: torch.device) -> dict[str, str]:
    """Describe the CPU or CUDA device that executed one benchmark record."""
    if device.type != "cuda":
        return {"type": device.type}
    properties = torch.cuda.get_device_properties(device)
    return {
        "type": "cuda",
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
    }


def _backend_implementation(backend: str, device: torch.device) -> str:
    """Name the implementation selected by a benchmark backend on this device."""
    if backend != "public_cuda":
        return backend

    has_triton = importlib.util.find_spec("torch_linear_assignment._triton") is not None
    if has_triton:
        return "triton" if torch.cuda.get_device_capability(device) >= (8, 0) else "scipy_fallback"
    if importlib.util.find_spec("torch_linear_assignment._backend") is not None:
        return "legacy_cuda"
    return "scipy_fallback"


def _solver_dtype_name(input_dtype: torch.dtype) -> str:
    """Name the documented dtype used by SciPy and Triton parity checks."""
    return "float64" if input_dtype == torch.float64 else "float32"


def _scipy_oracle(cost: torch.Tensor) -> torch.Tensor:
    """Solve promoted CPU costs with the public SciPy-defined semantics."""
    solver_dtype = torch.float64 if cost.dtype == torch.float64 else torch.float32
    cpu_cost = cost.detach().to(device="cpu", dtype=solver_dtype).contiguous()
    batch_size, workers, _ = cpu_cost.shape
    assignment = torch.full((batch_size, workers), -1, dtype=torch.long)
    for batch_index, matrix in enumerate(cpu_cost):
        row_indices, column_indices = linear_sum_assignment(matrix.numpy())
        assignment[batch_index, torch.from_numpy(row_indices)] = torch.from_numpy(column_indices)
    return assignment


def _backend_callable(
    backend: str,
    validation: str,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Resolve a public backend or an explicitly requested private comparison adapter."""
    if backend == "scipy":
        return _scipy_oracle
    if backend == "public_cuda":
        return batch_linear_assignment

    assignment_module = importlib.import_module("torch_linear_assignment.assignment")
    if backend == "legacy_cuda":
        legacy_backend = getattr(assignment_module, "_batch_linear_assignment_cuda_legacy", None)
        legacy_loader = getattr(assignment_module, "_load_legacy_backend", None)
        return legacy_backend if legacy_loader is not None and legacy_loader() is not None else None

    triton_loader = getattr(assignment_module, "_load_triton_backend", None)
    if triton_loader is None:
        return None
    triton_backend = triton_loader()
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


def _assignment_objectives(cost: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    """Return one promoted objective value for each batch assignment."""
    solver_dtype = torch.float64 if cost.dtype == torch.float64 else torch.float32
    promoted_cost = cost.to(solver_dtype)
    batch_indices = torch.arange(assignment.shape[0]).unsqueeze(1).expand_as(assignment)
    worker_indices = torch.arange(assignment.shape[1]).unsqueeze(0).expand_as(assignment)
    matched = assignment >= 0
    assigned_costs = promoted_cost[batch_indices, worker_indices, assignment.clamp_min(0)]
    return torch.where(matched, assigned_costs, 0).sum(dim=1)


def _output_contract(
    *,
    actual: torch.Tensor,
    expected: torch.Tensor,
    expected_device: torch.device,
    tasks: int,
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    """Validate assignment shape, dtype, device, and task-index range for diagnostics."""
    cpu_actual = actual.detach().cpu()
    violations = []
    if actual.shape != expected.shape:
        violations.append("shape")
    if actual.dtype != torch.long:
        violations.append("dtype")
    if actual.device != expected_device:
        violations.append("device")

    invalid_assignments = (cpu_actual < -1) | (cpu_actual >= tasks)
    details = {
        "expected_shape": list(expected.shape),
        "actual_shape": list(actual.shape),
        "expected_dtype": str(torch.long),
        "actual_dtype": str(actual.dtype),
        "expected_device": str(expected_device),
        "actual_device": str(actual.device),
        "assignment_range": [-1, tasks - 1],
    }
    if invalid_assignments.any():
        violations.append("assignment_range")
        first_invalid = invalid_assignments.nonzero()[0]
        details["invalid_assignment_count"] = int(invalid_assignments.sum())
        details["first_invalid_assignment"] = {
            "index": first_invalid.tolist(),
            "value": cpu_actual[tuple(first_invalid)].item(),
        }
    return cpu_actual, violations, details


def _parity_diagnostics(
    *,
    cost: torch.Tensor,
    expected: torch.Tensor,
    actual: torch.Tensor,
    result_call: str,
    expected_device: torch.device,
) -> dict[str, Any] | None:
    """Describe the first exact-parity failure with enough evidence to replay it."""
    cpu_actual, contract_violations, output_contract = _output_contract(
        actual=actual,
        expected=expected,
        expected_device=expected_device,
        tasks=cost.shape[2],
    )
    input_bytes = cost.contiguous().view(torch.uint8).numpy().tobytes()
    diagnostics = {
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "result_call": result_call,
        "contract_violations": contract_violations,
        "output_contract": output_contract,
    }
    if contract_violations:
        return diagnostics

    mismatched = (cpu_actual != expected).any(dim=1)
    mismatched_indices = mismatched.nonzero().flatten().tolist()
    if not mismatched_indices:
        return None

    expected_objectives = _assignment_objectives(cost, expected)
    actual_objectives = _assignment_objectives(cost, cpu_actual)
    first = mismatched_indices[0]
    return {
        **diagnostics,
        "mismatch_count": len(mismatched_indices),
        "mismatched_batch_indices": mismatched_indices,
        "expected_objectives": expected_objectives[mismatched].tolist(),
        "actual_objectives": actual_objectives[mismatched].tolist(),
        "first_mismatch": {
            "batch_index": first,
            "expected_assignment": expected[first].tolist(),
            "actual_assignment": cpu_actual[first].tolist(),
        },
    }


def _parity_failure_record(
    *,
    base: dict[str, Any],
    cost: torch.Tensor,
    expected: torch.Tensor,
    actual: torch.Tensor,
    result_call: str,
    expected_device: torch.device,
) -> dict[str, Any] | None:
    """Build a non-eligible case record when one solver invocation diverges."""
    diagnostics = _parity_diagnostics(
        cost=cost,
        expected=expected,
        actual=actual,
        result_call=result_call,
        expected_device=expected_device,
    )
    if diagnostics is None:
        return None
    return {
        **base,
        "status": "parity_failed",
        "parity": {"oracle": "scipy_promoted", "exact": False},
        "diagnostics": diagnostics,
        "acceptance_eligible": False,
    }


def _output_path(requested: Path | None, run_id: str) -> Path:
    """Resolve the explicit artifact path or a run-specific default."""
    return requested if requested is not None else Path(f"benchmark-results-{run_id}.jsonl")


def _emit(record: dict[str, Any], output: Path) -> None:
    """Append one lossless record to the JSONL artifact."""
    line = json.dumps(record, sort_keys=True)
    with output.open("a", encoding="utf-8") as output_file:
        output_file.write(f"{line}\n")


def _format_table(records: list[dict[str, Any]]) -> str:
    """Format status, cold/warm timing, and CPU speedup without discarding JSON."""
    headers = [
        "run",
        "backend",
        "validation",
        "dtype",
        "batch",
        "workers x tasks",
        "status",
        "warm median ms",
        "warm p95 ms",
        "cold ms",
        "CPU/GPU warm",
        "exact parity",
    ]
    rows = []
    for record in records:
        shape = record.get("shape", {})
        warm = record.get("timing", {}).get("warm_ms", {})
        cold = record.get("timing", {}).get("cold_first_call_ms")
        speedup = record.get("speedup_vs_scipy_cpu", {}).get("warm_median")
        parity = record.get("parity", {}).get("exact")
        rows.append(
            [
                str(record.get("run_label") or record.get("versions", {}).get("package") or "-"),
                str(record.get("implementation", record.get("backend", "-"))),
                str(record.get("validation_mode", "-")),
                str(record.get("input_dtype", "-")),
                str(shape.get("batch", "-")),
                (f"{shape['workers']} x {shape['tasks']}" if "workers" in shape and "tasks" in shape else "-"),
                str(record["status"]),
                f"{warm['median']:.1f}" if "median" in warm else "-",
                f"{warm['p95']:.1f}" if "p95" in warm else "-",
                f"{cold:.1f}" if cold is not None else "-",
                f"{speedup:.1f}x" if speedup is not None else "-",
                "yes" if parity is True else "no" if parity is False else "-",
            ]
        )

    return pd.DataFrame(rows, columns=headers).to_string(index=False, justify="left")


def _print_summary(records: list[dict[str, Any]], output: Path) -> None:
    """Print the compact table and location of the complete JSONL evidence."""
    print(_format_table(records))
    print(f"\nJSONL results: {output.resolve()}", flush=True)


def _status_record(
    status: str,
    run_id: str,
    run_label: str | None = None,
) -> dict[str, Any]:
    """Build a non-measurement record with explicit acceptance exclusion."""
    return {
        "schema_version": 1,
        "record_type": "run",
        "status": status,
        "run_id": run_id,
        "run_label": run_label,
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "versions": _versions(),
        "acceptance_eligible": False,
    }


def _case_key(record: dict[str, Any]) -> tuple[int, int, int, str, int] | None:
    """Return the deterministic input identity used to pair CPU and CUDA measurements."""
    shape = record.get("shape", {})
    required = ("batch", "workers", "tasks")
    if any(name not in shape for name in required):
        return None
    if "input_dtype" not in record or "seed" not in record:
        return None
    return (
        shape["batch"],
        shape["workers"],
        shape["tasks"],
        record["input_dtype"],
        record["seed"],
    )


def _attach_cpu_speedups(records: list[dict[str, Any]]) -> None:
    """Attach cold and warm CUDA speedups for measured records sharing a SciPy input."""
    cpu_records = {
        key: record
        for record in records
        if record.get("backend") == "scipy"
        and record.get("status") == "measured"
        and (key := _case_key(record)) is not None
    }
    for record in records:
        if record.get("backend") == "scipy" or record.get("status") != "measured":
            continue
        cpu_record = cpu_records.get(_case_key(record))
        if cpu_record is None:
            continue
        cpu_timing = cpu_record["timing"]
        cuda_timing = record["timing"]
        record["speedup_vs_scipy_cpu"] = {
            "cold_first_call": cpu_timing["cold_first_call_ms"] / cuda_timing["cold_first_call_ms"],
            "warm_median": cpu_timing["warm_ms"]["median"] / cuda_timing["warm_ms"]["median"],
            "warm_p95": cpu_timing["warm_ms"]["p95"] / cuda_timing["warm_ms"]["p95"],
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
    run_label: str | None = None,
) -> dict[str, Any]:
    """Measure one backend/workload pair after enforcing exact SciPy parity."""
    dtype = _DTYPES[dtype_name]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cpu_cost = torch.randn((batch, workers, tasks), generator=generator).to(dtype)
    oracle = _scipy_oracle(cpu_cost)
    operation = _backend_callable(backend, validation)
    benchmark_cost = cpu_cost if backend == "scipy" else cpu_cost.cuda()
    base = {
        "schema_version": 1,
        "record_type": "case",
        "run_id": run_id,
        "run_label": run_label,
        "backend": backend,
        "implementation": _backend_implementation(backend, benchmark_cost.device),
        "validation_mode": validation,
        "device": _device_metadata(benchmark_cost.device),
        "versions": _versions(),
        "shape": {
            "batch": batch,
            "workers": workers,
            "tasks": tasks,
            "orientation": ("transpose" if tasks < workers else "square" if tasks == workers else "direct"),
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

    torch.cuda.reset_peak_memory_stats() if benchmark_cost.is_cuda else None
    memory_before = torch.cuda.memory_allocated(benchmark_cost.device) if benchmark_cost.is_cuda else 0
    cold_ms, result = _time_call(operation, benchmark_cost)
    failure = _parity_failure_record(
        base=base,
        cost=cpu_cost,
        expected=oracle,
        actual=result,
        result_call="cold",
        expected_device=benchmark_cost.device,
    )
    if failure is not None:
        return failure

    for index in range(warmup):
        result = operation(benchmark_cost)
        failure = _parity_failure_record(
            base=base,
            cost=cpu_cost,
            expected=oracle,
            actual=result,
            result_call=f"warmup:{index}",
            expected_device=benchmark_cost.device,
        )
        if failure is not None:
            return failure

    samples = []
    for index in range(repetitions):
        elapsed_ms, result = _time_call(operation, benchmark_cost)
        failure = _parity_failure_record(
            base=base,
            cost=cpu_cost,
            expected=oracle,
            actual=result,
            result_call=f"timed:{index}",
            expected_device=benchmark_cost.device,
        )
        if failure is not None:
            return failure
        samples.append(elapsed_ms)
    peak_delta = (
        torch.cuda.max_memory_allocated(benchmark_cost.device) - memory_before if benchmark_cost.is_cuda else None
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
        "acceptance_eligible": backend in {"public_cuda", "legacy_cuda", "triton"},
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
    output = _output_path(arguments.output, run_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    records = []
    if not torch.cuda.is_available():
        record = _status_record("gpu_skipped", run_id, getattr(arguments, "label", None))
        records.append(record)
        _emit(record, output)
        _print_summary(records, output)
        return

    cases = [
        (backend, validation, batch, tasks, dtype_name)
        for backend in arguments.backends
        for validation in _validation_modes(backend, arguments.validation_modes)
        for batch in arguments.batches
        for tasks in arguments.tasks
        for dtype_name in arguments.dtypes
    ]
    parity_failed = False
    for backend, validation, batch, tasks, dtype_name in tqdm(cases, desc="Benchmark cases", unit="case"):
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
                run_label=getattr(arguments, "label", None),
            )
        except (torch.OutOfMemoryError, MemoryError):
            torch.cuda.empty_cache()
            record = {
                **_status_record("oom", run_id, getattr(arguments, "label", None)),
                "backend": backend,
                "validation_mode": validation,
                "shape": {
                    "batch": batch,
                    "workers": arguments.workers,
                    "tasks": tasks,
                },
                "input_dtype": dtype_name,
            }
        records.append(record)
        parity_failed |= record["status"] == "parity_failed"

    _attach_cpu_speedups(records)
    for record in records:
        _emit(record, output)
    _print_summary(records, output)
    if parity_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
