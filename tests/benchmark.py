"""Benchmark SciPy CPU and installed CUDA backends with exact-parity gates.

The script saves lossless JSON Lines and prints a compact human-readable table.
Each requested case runs in fresh processes so cold compilation, warm latency,
and allocation boundaries remain auditable without merging devices. Missing
GPU evidence, incomplete process rounds, and solver mismatches fail closed;
``--allow-incomplete`` is an explicit diagnostic-only escape hatch.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from torch_linear_assignment import batch_linear_assignment

_WORKER_RECORD_PREFIX = "__TLA_BENCHMARK_RECORD__="
_DEFAULT_WORKER_TIMEOUT_SECONDS = 900.0

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}

_VALIDATION_MODES = (
    "off",
    "nonfinite_only",
    "infeasibility_flag_only",
    "full",
)

#: Default workload, a cross product of one worker count with tasks and batches.
_DEFAULT_WORKERS = 300
_DEFAULT_TASKS = [100, 300, 600]
# One problem is routed to SciPy by the public call, so a default case of 1 would not measure the GPU.
_DEFAULT_BATCHES = [16, 50, 208, 624]

#: Named workloads as explicit ``(batch, workers, tasks)`` triples, for shapes a cross product
#: over a single worker count cannot express.
_PRESETS: dict[str, tuple[tuple[int, int, int], ...]] = {
    # RF-DETR training: 300 queries per group against 1-30 targets per image. The batches are
    # 4 and 16 images x 13 query groups, and 16 images x 13 groups x 3 decoder layers.
    "rfdetr": tuple((batch, 300, tasks) for batch in (52, 208, 624) for tasks in (1, 2, 5, 10, 20, 30)),
    # Wide, square, and many-small-problem shapes, where kernel-side changes cost the most.
    "wide": ((208, 300, 300), (208, 300, 600), (208, 512, 512), (832, 300, 60)),
    # The three batch-208 orientations of the README's representative performance table, so a GPU
    # class can be added to it from one run. Run it with five process rounds, three warmup calls and
    # thirty repetitions, which `make benchmark-readme` and the Modal lane both do.
    "readme": ((208, 300, 100), (208, 300, 300), (208, 300, 600)),
}

#: Lowest NVIDIA compute capability the package runs Triton on; below it the public call uses SciPy.
_MINIMUM_TRITON_COMPUTE_CAPABILITY = (8, 0)


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


def _positive_finite_seconds(value: str) -> float:
    """Parse a positive, finite subprocess timeout in seconds."""
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive finite number of seconds") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("expected a positive finite number of seconds")
    return seconds


def _arguments() -> argparse.Namespace:
    """Parse the reproducible Issue 32 benchmark configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        type=_csv_values,
        default=["scipy", "public_cuda"],
        help="comma-separated: scipy,public_cuda,triton",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(_PRESETS),
        help="named (batch, workers, tasks) shapes; replaces --workers, --tasks and --batches",
    )
    parser.add_argument("--workers", type=int, help=f"default: {_DEFAULT_WORKERS}")
    parser.add_argument("--tasks", type=_integer_csv, help=f"default: {_DEFAULT_TASKS}")
    parser.add_argument("--batches", type=_integer_csv, help=f"default: {_DEFAULT_BATCHES}")
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
    parser.add_argument("--seed", type=int, default=320)
    parser.add_argument(
        "--label",
        help="optional label identifying the installed package or comparison run",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--process-rounds",
        type=int,
        default=3,
        help="fresh-process repetitions; alternate rounds reverse backend order",
    )
    parser.add_argument(
        "--worker-timeout-seconds",
        type=_positive_finite_seconds,
        default=_DEFAULT_WORKER_TIMEOUT_SECONDS,
        help="maximum fresh-worker runtime in seconds; default accommodates cold GPU compilation",
    )
    parser.add_argument(
        "--backend-order",
        choices=("alternate", "forward", "reverse"),
        default="alternate",
        help="case-local backend order; alternate reverses every process round",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="exit successfully with unavailable/OOM/skip rows for diagnostic runs only",
    )
    parser.add_argument(
        "--expect-package-version",
        help="fail unless every case reports this installed distribution version",
    )
    parser.add_argument(
        "--expect-implementation",
        choices=("triton", "legacy_cuda", "scipy_fallback"),
        help="fail unless every measured non-SciPy row executes this implementation",
    )
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    parser.add_argument(
        "--output",
        type=Path,
        help="JSONL artifact path (default: benchmark-results-<run-id>.jsonl)",
    )
    arguments = parser.parse_args()

    explicit_dimensions = [arguments.workers, arguments.tasks, arguments.batches]
    if arguments.preset is not None and any(value is not None for value in explicit_dimensions):
        parser.error("--preset sets every shape; do not combine it with --workers, --tasks or --batches")
    arguments.workers = _DEFAULT_WORKERS if arguments.workers is None else arguments.workers
    arguments.tasks = _DEFAULT_TASKS if arguments.tasks is None else arguments.tasks
    arguments.batches = _DEFAULT_BATCHES if arguments.batches is None else arguments.batches

    unknown_backends = set(arguments.backends) - {
        "scipy",
        "public_cuda",
        "triton",
    }
    unknown_dtypes = set(arguments.dtypes) - _DTYPES.keys()
    unknown_validation_modes = set(arguments.validation_modes) - set(_VALIDATION_MODES)
    if unknown_backends:
        parser.error(f"unknown backends: {sorted(unknown_backends)}")
    if unknown_dtypes:
        parser.error(f"unknown dtypes: {sorted(unknown_dtypes)}")
    if unknown_validation_modes:
        parser.error(f"unknown validation modes: {sorted(unknown_validation_modes)}")
    if arguments.workers <= 0 or arguments.warmup < 0 or arguments.repetitions <= 0 or arguments.process_rounds <= 0:
        parser.error("workers/repetitions/process-rounds must be positive and warmup non-negative")
    return arguments


def _versions() -> dict[str, str | None]:
    """Return package and environment versions without importing optional Triton eagerly."""
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton_version = None
    # The active fork publishes as torch-hungarian; the frozen 0.0.x baseline is still
    # distributed upstream as torch-linear-assignment. Both install torch_linear_assignment.
    package_version = None
    for distribution_name in ("torch-hungarian", "torch-linear-assignment"):
        try:
            package_version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        break
    return {
        "package": package_version,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton_version,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
    }


def _cpu_model() -> str:
    """Return the most specific locally available CPU model string."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _nvidia_smi_metadata() -> dict[str, Any]:
    """Capture best-effort driver, clock, power, and temperature state."""
    query = "driver_version,name,pci.bus_id,clocks.sm,clocks.mem,power.draw,temperature.gpu"
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return {"status": "unavailable", "reason": "nvidia-smi executable not found"}
    except subprocess.TimeoutExpired:
        return {"status": "unavailable", "reason": "nvidia-smi timed out"}
    if completed.returncode != 0:
        return {
            "status": "unavailable",
            "reason": completed.stderr.strip() or f"nvidia-smi exited {completed.returncode}",
        }
    fields = ("driver_version", "name", "pci_bus_id", "sm_clock_mhz", "memory_clock_mhz", "power_w", "temperature_c")
    devices = [
        dict(zip(fields, (value.strip() for value in line.split(",")), strict=True))
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    return {"status": "measured", "devices": devices}


def _source_metadata() -> dict[str, str | None]:
    """Bind evidence to the benchmark source and installed package origin."""
    execution_path = Path(__file__).resolve()
    script_path = Path(os.environ.get("TLA_BENCHMARK_SOURCE_PATH", execution_path)).resolve()
    package_module = importlib.import_module("torch_linear_assignment")
    package_origin = getattr(package_module, "__file__", None)
    package_path = Path(package_origin).resolve() if package_origin else None
    assignment_module = importlib.import_module("torch_linear_assignment.assignment")
    assignment_origin = getattr(assignment_module, "__file__", None)
    assignment_path = Path(assignment_origin).resolve() if assignment_origin else None
    revision = os.environ.get("TLA_BENCHMARK_GIT_REVISION")
    dirty = "unavailable"
    if revision is None:
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                check=True,
                cwd=script_path.parent.parent,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            revision = None
    try:
        dirty = (
            "dirty"
            if subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                check=True,
                cwd=script_path.parent.parent,
                text=True,
                timeout=10,
            ).stdout.strip()
            else "clean"
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    return {
        "benchmark_path": str(script_path),
        "benchmark_execution_path": str(execution_path),
        "benchmark_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "benchmark_git_revision": revision,
        "benchmark_git_dirty": dirty,
        "package_module_origin": str(package_path) if package_path else None,
        "package_module_sha256": (
            hashlib.sha256(package_path.read_bytes()).hexdigest()
            if package_path is not None and package_path.is_file()
            else None
        ),
        "assignment_module_origin": str(assignment_path) if assignment_path else None,
        "assignment_module_sha256": (
            hashlib.sha256(assignment_path.read_bytes()).hexdigest()
            if assignment_path is not None and assignment_path.is_file()
            else None
        ),
    }


def _run_metadata() -> dict[str, Any]:
    """Capture run-level provenance once so telemetry does not perturb every case."""
    return {
        "source": _source_metadata(),
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
        },
        "versions": _versions(),
        "gpu_state_start": _nvidia_smi_metadata(),
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


def _backend_implementation(backend: str, cost: torch.Tensor) -> str:
    """Name the public route for the measured cost after the call loads its backend."""
    if backend != "public_cuda":
        return backend

    if "torch_linear_assignment._triton" in sys.modules:
        # Older releases have no shape-dependent dispatch and are still benchmarked by this script.
        uses_triton = getattr(sys.modules.get("torch_linear_assignment.assignment"), "_uses_triton", None)
        if uses_triton is not None and not uses_triton(cost):
            return "scipy_fallback"
        return "triton"
    if "torch_linear_assignment._backend" in sys.modules:
        return "legacy_cuda"
    return "scipy_fallback"


def _implementation_metadata(implementation: str) -> dict[str, str | None]:
    """Identify and hash the loaded module that executed the GPU implementation."""
    module_name = {
        "triton": "torch_linear_assignment._triton",
        "legacy_cuda": "torch_linear_assignment._backend",
    }.get(implementation)
    module = sys.modules.get(module_name) if module_name else None
    origin_value = getattr(module, "__file__", None)
    origin = Path(origin_value).resolve() if origin_value else None
    return {
        "module": module_name,
        "origin": str(origin) if origin else None,
        "sha256": (
            hashlib.sha256(origin.read_bytes()).hexdigest() if origin is not None and origin.is_file() else None
        ),
    }


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
    """Resolve the public entry point, the SciPy oracle, or the private Triton adapter."""
    if backend == "scipy":
        return _scipy_oracle
    if backend == "public_cuda":
        return batch_linear_assignment

    assignment_module = importlib.import_module("torch_linear_assignment.assignment")
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


def _kernel_event_timing(
    cost: torch.Tensor, implementation: str, repetitions: int, expected: torch.Tensor
) -> dict[str, Any]:
    """Time the bare Triton launch with CUDA events, without the public wrapper's host-side work.

    The wrapper's dtype promotion and transpose run once, outside the event window, and the launch
    uses validation ``off`` so no host read sits inside it. Rows from SciPy or the legacy CUDA
    extension have no such entry point and report ``unavailable``, which does not make a case
    incomplete. The private entry point is looked up only after the measured call has loaded it.
    """
    solve = getattr(sys.modules.get("torch_linear_assignment._triton"), "_solve", None)
    if not cost.is_cuda or implementation != "triton" or solve is None:
        return {"status": "unavailable", "reason": f"no Triton kernel entry point for {implementation!r}"}
    _, workers, tasks = cost.shape
    if cost.numel() == 0:
        return {"status": "unavailable", "reason": "empty problems launch no kernel"}
    solver_cost = cost if cost.dtype == torch.float64 else cost.to(torch.float32)
    solver_cost = (solver_cost.transpose(1, 2) if tasks < workers else solver_cost).contiguous()
    samples = []
    with torch.cuda.device(cost.device):
        # One untimed launch so no lazy setup can land inside the first event window.
        outputs = solve(solver_cost, "off")
        result = outputs[1 if tasks < workers else 0].to(torch.long)
        if not torch.equal(result.cpu(), expected):
            return {"status": "parity_failed", "result_call": "kernel_warmup"}
        for _ in range(repetitions):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(cost.device)
            start.record()
            outputs = solve(solver_cost, "off")
            end.record()
            end.synchronize()
            result = outputs[1 if tasks < workers else 0].to(torch.long)
            if not torch.equal(result.cpu(), expected):
                return {"status": "parity_failed", "result_call": f"kernel_timed:{len(samples)}"}
            samples.append(start.elapsed_time(end))
    return {
        "status": "measured",
        "n": repetitions,
        "median": statistics.median(samples),
        "p95": _nearest_rank_p95(samples),
        "samples": samples,
        "quantile": "nearest_rank",
        "timer": "cuda_events",
        # The device idles until the host launches, so launch overhead lands inside the window.
        "included": ["solver workspace allocation", "Triton launch host overhead", "kernel execution"],
        "excluded": ["input validation", "dtype promotion", "transpose", "result conversion"],
    }


def _start_memory_boundary(cost: torch.Tensor) -> dict[str, int | str | None]:
    """Reset CUDA peak statistics immediately before one measured allocation boundary."""
    if not cost.is_cuda:
        return {
            "status": "not_cuda",
            "allocated_before_bytes": None,
            "allocated_after_bytes": None,
            "peak_allocated_bytes": None,
            "peak_allocated_delta_bytes": None,
        }
    torch.cuda.reset_peak_memory_stats(cost.device)
    return {
        "status": "measured",
        "allocated_before_bytes": torch.cuda.memory_allocated(cost.device),
        "allocated_after_bytes": None,
        "peak_allocated_bytes": None,
        "peak_allocated_delta_bytes": None,
    }


def _finish_memory_boundary(
    cost: torch.Tensor,
    boundary: dict[str, int | str | None],
) -> dict[str, int | str | None]:
    """Complete one CUDA allocation boundary after synchronized measured work."""
    if not cost.is_cuda:
        return boundary
    allocated_after = torch.cuda.memory_allocated(cost.device)
    peak_allocated = torch.cuda.max_memory_allocated(cost.device)
    allocated_before = boundary["allocated_before_bytes"]
    assert isinstance(allocated_before, int)
    return {
        **boundary,
        "allocated_after_bytes": allocated_after,
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_delta_bytes": peak_allocated - allocated_before,
    }


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
        "solver median ms",
        "cold ms",
        "CPU/GPU warm",
        "exact parity",
        "rounds",
    ]
    rows = []
    for record in records:
        shape = record.get("shape", {})
        warm = record.get("timing", {}).get("warm_ms", {})
        kernel = record.get("timing", {}).get("kernel_ms", {})
        cold = record.get("timing", {}).get("cold_first_call_ms")
        speedup = record.get("speedup_vs_scipy_cpu", {}).get("warm_median")
        parity = record.get("parity", {}).get("exact")
        rows.append(
            [
                str(record.get("run_label") or record.get("versions", {}).get("package") or "-"),
                _backend_label(record),
                str(record.get("validation_mode", "-")),
                str(record.get("input_dtype", "-")),
                str(shape.get("batch", "-")),
                (f"{shape['workers']} x {shape['tasks']}" if "workers" in shape and "tasks" in shape else "-"),
                str(record["status"]),
                f"{warm['median']:.1f}" if "median" in warm else "-",
                f"{warm['p95']:.1f}" if "p95" in warm else "-",
                f"{kernel['median']:.2f}" if "median" in kernel else "-",
                f"{cold:.1f}" if cold is not None else "-",
                f"{speedup:.1f}x" if speedup is not None else "-",
                "yes" if parity is True else "no" if parity is False else "-",
                str(record.get("process_rounds", "-")),
            ]
        )

    return pd.DataFrame(rows, columns=headers).to_string(index=False, justify="left")


def _backend_label(record: dict[str, Any]) -> str:
    """Name the measured lane, and the implementation it ran when that differs from the lane.

    The public call and the private Triton lane both execute Triton on a supported GPU, so the
    label keeps them apart as ``public_cuda/triton`` and ``triton``.
    """
    backend = record.get("backend")
    implementation = record.get("implementation")
    if backend is None or implementation in (None, backend):
        return str(implementation or backend or "-")
    return f"{backend}/{implementation}"


def _median_present(values: list[float | None]) -> float | None:
    """Return the median of the values that were recorded, or ``None`` when none were."""
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def _summarize_case_rounds(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse one case's fresh-process rounds into a single row of medians.

    Warm and kernel medians and cold time are medians across rounds, warm p95 is the worst
    round's p95, and parity is exact only when every round was exact.
    """
    timings = [record.get("timing", {}) for record in group]
    warm = {
        "median": _median_present([timing.get("warm_ms", {}).get("median") for timing in timings]),
        "p95": max(
            (timing["warm_ms"]["p95"] for timing in timings if "p95" in timing.get("warm_ms", {})), default=None
        ),
    }
    timing: dict[str, Any] = {
        "cold_first_call_ms": _median_present([timing.get("cold_first_call_ms") for timing in timings]),
        "warm_ms": {name: value for name, value in warm.items() if value is not None},
    }
    kernel_median = _median_present([timing.get("kernel_ms", {}).get("median") for timing in timings])
    if kernel_median is not None:
        timing["kernel_ms"] = {"median": kernel_median}
    exact = [record.get("parity", {}).get("exact") for record in group]
    identity = (
        "run_label",
        "versions",
        "backend",
        "implementation",
        "status",
        "validation_mode",
        "input_dtype",
        "shape",
        "seed",
    )
    return {
        **{name: group[0][name] for name in identity if name in group[0]},
        "timing": timing,
        "parity": {"exact": False if False in exact else True if all(value is True for value in exact) else None},
        "process_rounds": len(group),
        # The default matches `_case_key`, so a record without a round never adds a `None` id.
        "round_ids": [record.get("process_round", 0) for record in group],
    }


def _summarize_rounds(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce measured records to one row per case for the console table.

    Every process round of a case is folded into one row of medians, and each GPU row's
    speedup is recomputed from those medians. Records that are not measurements, such as
    skips and failures, are kept as they are, so none of them is hidden.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    order: list[tuple[Any, ...] | dict[str, Any]] = []
    for record in records:
        if record.get("status") != "measured":
            order.append(record)
            continue
        shape = record.get("shape", {})
        key = (
            record.get("run_label"),
            record.get("backend"),
            record.get("implementation"),
            record.get("validation_mode"),
            shape.get("batch"),
            shape.get("workers"),
            shape.get("tasks"),
            record.get("input_dtype"),
            record.get("seed"),
        )
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)

    rows = [entry if isinstance(entry, dict) else _summarize_case_rounds(groups[entry]) for entry in order]

    def _pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
        shape = row.get("shape") or {}
        return (
            row.get("run_label"),
            shape.get("batch"),
            shape.get("workers"),
            shape.get("tasks"),
            row.get("input_dtype"),
            row.get("seed"),
        )

    scipy_warm = {
        _pair_key(row): row["timing"]["warm_ms"].get("median")
        for row in rows
        if row.get("backend") == "scipy" and row.get("status") == "measured"
    }
    for row in rows:
        if row.get("backend") == "scipy" or row.get("status") != "measured":
            continue
        failed = [
            record
            for record in records
            if record.get("status") != "measured"
            and record.get("backend") == row.get("backend")
            and _pair_key(record) == _pair_key(row)
            and record.get("validation_mode") == row.get("validation_mode")
        ]
        if any(record.get("parity", {}).get("exact") is False for record in failed):
            row["parity"]["exact"] = False
        elif failed:
            row["parity"]["exact"] = None
        cpu_row = next(
            (
                item
                for item in rows
                if item.get("backend") == "scipy"
                and item.get("status") == "measured"
                and _pair_key(item) == _pair_key(row)
            ),
            None,
        )
        cpu_failed = any(
            record.get("status") != "measured"
            and record.get("backend") == "scipy"
            and _pair_key(record) == _pair_key(row)
            for record in records
        )
        if failed or cpu_failed or cpu_row is None or set(cpu_row["round_ids"]) != set(row["round_ids"]):
            continue
        cpu, gpu = scipy_warm.get(_pair_key(row)), row["timing"]["warm_ms"].get("median")
        if cpu is not None and gpu:
            row["speedup_vs_scipy_cpu"] = {"warm_median": cpu / gpu}
    return rows


def _print_summary(records: list[dict[str, Any]], output: Path) -> None:
    """Print one table row per case, its limitations, and the location of the complete JSONL evidence."""
    print(_format_table(_summarize_rounds(records)))
    limitations = [detail for record in records for detail in record.get("details", [])]
    if limitations:
        print("\nLimitations:\n" + "\n".join(f"- {detail}" for detail in limitations))
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


def _case_key(record: dict[str, Any]) -> tuple[int, int, int, str, tuple[int, int]] | None:
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
        (record["seed"], record.get("process_round", 0)),
    )


def _attach_process_statistics(records: list[dict[str, Any]], required_rounds: int) -> None:
    """Aggregate cold and warm timing across the required fresh-process rounds."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        shape = record.get("shape", {})
        required_shape = ("batch", "workers", "tasks")
        if record.get("status") != "measured" or any(name not in shape for name in required_shape):
            continue
        key = (
            record.get("backend"),
            record.get("implementation"),
            record.get("validation_mode"),
            shape["batch"],
            shape["workers"],
            shape["tasks"],
            record.get("input_dtype"),
            record.get("seed"),
            record.get("input_sha256"),
        )
        groups.setdefault(key, []).append(record)

    for group in groups.values():
        implementation = group[0].get("implementation")
        rounds = sorted(record["process_round"] for record in group)
        input_hashes = {record.get("input_sha256") for record in group}
        cold_samples = [record["timing"]["cold_first_call_ms"] for record in group]
        warm_samples = [record["timing"]["warm_ms"]["median"] for record in group]
        isolated = all(record.get("process_isolated", False) for record in group)
        complete = rounds == list(range(required_rounds)) and len(input_hashes) == 1 and isolated
        process_statistics = {
            "status": "measured" if complete else "incomplete",
            "rounds": rounds,
            "required_rounds": required_rounds,
            "input_sha256": next(iter(input_hashes)) if len(input_hashes) == 1 else None,
            "cold_first_call_ms": {
                "samples": cold_samples,
                "median": statistics.median(cold_samples),
                "p95": _nearest_rank_p95(cold_samples),
            },
            "warm_median_ms": {
                "samples": warm_samples,
                "median": statistics.median(warm_samples),
                "p95": _nearest_rank_p95(warm_samples),
            },
        }
        cold_compilation = {
            "status": process_statistics["status"],
            "definition": (
                "fresh_process_first_scipy_solver_call_after_package_import_and_input_setup"
                if group[0].get("backend") == "scipy"
                else "fresh_process_first_public_solver_call_after_package_import_cuda_context_and_device_input_setup"
            ),
            "samples_required": required_rounds,
            "n": len(cold_samples),
            "samples_ms": cold_samples,
            "median_ms": statistics.median(cold_samples),
            "p95_ms": _nearest_rank_p95(cold_samples),
            "quantile": "nearest_rank",
            "fresh_process": isolated,
            "cache_policy": ("empty_unique_triton_cache" if implementation == "triton" else "not_applicable"),
        }
        for record in group:
            record["process_statistics"] = process_statistics
            record["cold_compilation"] = cold_compilation
            if not complete:
                record["acceptance_eligible"] = False


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


def _validation_group_key(record: dict[str, Any]) -> tuple[Any, ...] | None:
    """Return the identity shared by validation-mode measurements of one input."""
    case_key = _case_key(record)
    if case_key is None or record.get("backend") != "triton":
        return None
    return (*case_key, record.get("input_sha256"))


def _attach_validation_overheads(records: list[dict[str, Any]]) -> None:
    """Attach full four-mode validation overheads without mixing inputs or rounds."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = _validation_group_key(record)
        if key is not None:
            groups.setdefault(key, []).append(record)

    required = set(_VALIDATION_MODES)
    for group in groups.values():
        measured = {record["validation_mode"]: record for record in group if record.get("status") == "measured"}
        observed = set(measured)
        digests = {record.get("input_sha256") for record in group}
        if observed != required or len(digests) != 1:
            overhead: dict[str, Any] = {
                "status": "incomplete",
                "required_modes": list(_VALIDATION_MODES),
                "observed_modes": sorted(observed),
                "input_sha256": next(iter(digests)) if len(digests) == 1 else None,
            }
            for record in group:
                record["validation_overhead"] = overhead
                record["acceptance_eligible"] = False
            continue

        medians = {mode: measured[mode]["timing"]["warm_ms"]["median"] for mode in _VALIDATION_MODES}
        baseline = medians["off"]
        absolute = {mode: medians[mode] - baseline for mode in _VALIDATION_MODES[1:]}
        percent = {mode: value / baseline * 100.0 if baseline else None for mode, value in absolute.items()}
        interaction = medians["full"] - medians["nonfinite_only"] - medians["infeasibility_flag_only"] + baseline
        overhead = {
            "status": "measured",
            "baseline_mode": "off",
            "input_sha256": next(iter(digests)),
            "warm_median_ms": medians,
            "absolute_vs_off_ms": absolute,
            "percent_vs_off": percent,
            "interaction_ms": interaction,
            "interaction_percent_vs_off": interaction / baseline * 100.0 if baseline else None,
        }
        for record in group:
            record["validation_overhead"] = overhead


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
    process_round: int = 0,
    process_isolated: bool = False,
    cold_cache_policy: str | None = None,
    cold_samples_required: int = 1,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure one backend/workload pair and validate every solver result."""
    dtype = _DTYPES[dtype_name]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cpu_cost = torch.randn((batch, workers, tasks), generator=generator, dtype=dtype)
    operation = _backend_callable(backend, validation)
    benchmark_cost = cpu_cost if backend == "scipy" else cpu_cost.cuda()
    included_work = [
        "backend adapter invocation",
        "solver workspace allocation",
        "solver execution",
        "result construction and conversion",
    ]
    excluded_work = [
        "benchmark and backend imports",
        "input generation",
        "SciPy oracle construction",
        "post-timing parity diagnostics",
    ]
    if benchmark_cost.is_cuda:
        included_work.append("CUDA completion synchronization")
        excluded_work.extend(["CUDA context creation", "host-to-device input transfer"])
    provenance = run_metadata or {
        "source": _source_metadata(),
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
        },
        "versions": _versions(),
    }
    base = {
        "schema_version": 1,
        "record_type": "case",
        "run_id": run_id,
        "run_label": run_label,
        "process_round": process_round,
        "process_isolated": process_isolated,
        "backend": backend,
        "validation_mode": validation,
        "device": _device_metadata(benchmark_cost.device),
        "versions": _versions(),
        "run_metadata": provenance,
        "shape": {
            "batch": batch,
            "workers": workers,
            "tasks": tasks,
            "orientation": ("transpose" if tasks < workers else "square" if tasks == workers else "direct"),
        },
        "input_dtype": dtype_name,
        "solver_dtype": _solver_dtype_name(dtype),
        "seed": seed,
        "input_sha256": hashlib.sha256(memoryview(cpu_cost.contiguous().view(torch.uint8).numpy())).hexdigest(),
        "measurement_scope": {
            "kind": "cpu_solver_latency" if backend == "scipy" else "device_resident_solver_latency",
            "included": included_work,
            "excluded": excluded_work,
        },
    }
    if operation is None:
        implementation = _backend_implementation(backend, benchmark_cost)
        return {
            **base,
            "implementation": implementation,
            "implementation_source": _implementation_metadata(implementation),
            "status": "backend_unavailable",
            "acceptance_eligible": False,
        }

    cold_memory = _start_memory_boundary(benchmark_cost)
    cold_ms, result = _time_call(operation, benchmark_cost)
    cold_memory = _finish_memory_boundary(benchmark_cost, cold_memory)
    implementation = _backend_implementation(backend, benchmark_cost)
    base["implementation"] = implementation
    base["implementation_source"] = _implementation_metadata(implementation)
    if implementation == "triton":
        base["measurement_scope"]["included"].append("solver validation")

    # Construct the oracle only after the first timed public solver call. This
    # preserves a symmetric first-call boundary while still rejecting its result.
    oracle = _scipy_oracle(cpu_cost)
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

    # Release the prior output before measuring a steady-state call. Otherwise
    # replacing ``result`` briefly counts two output tensors in the warm peak.
    del result
    warm_memory = _start_memory_boundary(benchmark_cost)
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
        del result
    warm_memory = _finish_memory_boundary(benchmark_cost, warm_memory)
    # After the memory boundary closes, so the extra launches do not count in the warm peak.
    kernel_timing = _kernel_event_timing(benchmark_cost, implementation, repetitions, oracle)
    if kernel_timing.get("status") == "parity_failed":
        return {
            **base,
            "status": "parity_failed",
            "parity": {"oracle": "scipy_promoted", "exact": False},
            "diagnostics": kernel_timing,
            "acceptance_eligible": False,
        }
    cold_definition = (
        "fresh_process_first_scipy_solver_call_after_package_import_and_input_setup"
        if backend == "scipy"
        else "fresh_process_first_public_solver_call_after_package_import_cuda_context_and_device_input_setup"
    )
    effective_cache_policy = cold_cache_policy if implementation == "triton" else "not_applicable"
    return {
        **base,
        "status": "measured",
        "timing": {
            "cold_first_call_ms": cold_ms,
            "cold_method": (cold_definition if process_isolated else "current_process_first_timed_public_solver_call"),
            "warm_ms": {
                "n": repetitions,
                "warmup": warmup,
                "median": statistics.median(samples),
                "p95": _nearest_rank_p95(samples),
                "samples": samples,
                "quantile": "nearest_rank",
                "timer": "wall_clock_plus_cuda_sync",
            },
            "kernel_ms": kernel_timing,
        },
        "memory": {
            "cold": cold_memory,
            "warm": warm_memory,
            "logical_workspace_bytes": None,
            "workspace_source": "unavailable",
        },
        "cold_compilation": {
            "status": "measured" if process_isolated else "not_isolated",
            "definition": cold_definition,
            "cache_policy": effective_cache_policy,
            "cache_directory": (
                os.environ.get("TRITON_CACHE_DIR") if process_isolated and implementation == "triton" else None
            ),
            "samples_required": cold_samples_required,
        },
        "parity": {"oracle": "scipy_promoted", "exact": True},
        "acceptance_eligible": backend in {"public_cuda", "triton"} and implementation in {"triton", "legacy_cuda"},
        "exclusions": [
            "GPU driver and hardware state are observed but not reset between case processes",
            "logical workspace accounting is unavailable",
        ],
    }


def _validation_modes(backend: str, requested: Iterable[str]) -> list[str]:
    """Return private validation variants only for the Triton backend."""
    return list(requested) if backend == "triton" else ["full"]


def _case_shapes(arguments: argparse.Namespace) -> list[tuple[int, int, int]]:
    """Return the requested ``(batch, workers, tasks)`` shapes, from a preset or the cross product."""
    preset = getattr(arguments, "preset", None)
    if preset is not None:
        return list(_PRESETS[preset])
    return [(batch, arguments.workers, tasks) for batch in arguments.batches for tasks in arguments.tasks]


def _case_matrix(arguments: argparse.Namespace) -> list[tuple[int, str, str, int, int, int, str]]:
    """Interleave comparable backends and reverse order across process rounds."""
    cases = []
    for process_round in range(arguments.process_rounds):
        backends = list(arguments.backends)
        if arguments.backend_order == "reverse" or (arguments.backend_order == "alternate" and process_round % 2):
            backends.reverse()
        for batch, workers, tasks in _case_shapes(arguments):
            for dtype_name in arguments.dtypes:
                for backend in backends:
                    for validation in _validation_modes(backend, arguments.validation_modes):
                        cases.append((process_round, backend, validation, batch, workers, tasks, dtype_name))
    return cases


def _case_status_record(specification: dict[str, Any], status: str, detail: str | None = None) -> dict[str, Any]:
    """Build an auditable non-measurement row for one requested case."""
    record = {
        **_status_record(status, specification["run_id"], specification.get("run_label")),
        "record_type": "case",
        "process_round": specification.get("process_round", 0),
        "backend": specification["backend"],
        "validation_mode": specification["validation"],
        "shape": {
            "batch": specification["batch"],
            "workers": specification["workers"],
            "tasks": specification["tasks"],
        },
        "input_dtype": specification["dtype_name"],
        "seed": specification["seed"],
        "run_metadata": specification.get("run_metadata"),
    }
    if detail:
        record["detail"] = detail
    return record


def _worker_case_record(specification: dict[str, Any]) -> dict[str, Any]:
    """Execute one isolated case while converting expected memory failures to evidence."""
    run_metadata = specification.get("run_metadata")
    if run_metadata is not None:
        specification = {
            **specification,
            "run_metadata": {**run_metadata, "source": _source_metadata()},
        }
    try:
        return _case_record(**specification)
    except (torch.OutOfMemoryError, MemoryError) as error:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _case_status_record(specification, "oom", str(error))


def _isolated_case_record(specification: dict[str, Any]) -> dict[str, Any]:
    """Run one backend/case in a fresh interpreter and recover its JSON record."""
    worker_timeout_seconds = specification.get("worker_timeout_seconds", _DEFAULT_WORKER_TIMEOUT_SECONDS)
    worker_specification = {
        **{key: value for key, value in specification.items() if key != "worker_timeout_seconds"},
        "process_isolated": True,
        "cold_cache_policy": "empty_unique_triton_cache",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="tla-triton-cache-") as cache_directory:
            environment = os.environ.copy()
            environment["TRITON_CACHE_DIR"] = cache_directory
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-case",
                    json.dumps(worker_specification),
                ],
                capture_output=True,
                check=False,
                env=environment,
                text=True,
                timeout=worker_timeout_seconds,
            )
    except subprocess.TimeoutExpired:
        record = _case_status_record(
            specification,
            "worker_timeout",
            f"worker exceeded configured {worker_timeout_seconds:g}-second timeout",
        )
        record["worker_timeout_seconds"] = worker_timeout_seconds
        return record
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"worker exited {completed.returncode}"
        return _case_status_record(specification, "worker_failed", detail[-4000:])
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(_WORKER_RECORD_PREFIX):
            return json.loads(line.removeprefix(_WORKER_RECORD_PREFIX))
    detail = completed.stderr.strip() or completed.stdout.strip() or f"worker exited {completed.returncode}"
    return _case_status_record(specification, "worker_failed", detail[-4000:])


def _identity_failures(records: list[dict[str, Any]], arguments: argparse.Namespace) -> list[str]:
    """Return explicit package or backend identity expectation failures."""
    failures = []
    expected_package = getattr(arguments, "expect_package_version", None)
    if expected_package:
        observed = sorted(
            {record.get("versions", {}).get("package") for record in records if record.get("record_type") == "case"},
            key=str,
        )
        if observed != [expected_package]:
            failures.append(f"expected package {expected_package}, observed {observed}")
    expected_implementation = getattr(arguments, "expect_implementation", None)
    if expected_implementation:
        observed = sorted(
            {
                record.get("implementation")
                for record in records
                if record.get("backend") != "scipy" and record.get("status") == "measured"
            },
            key=str,
        )
        if observed != [expected_implementation]:
            failures.append(f"expected implementation {expected_implementation}, observed {observed}")
    return failures


def _case_is_complete(record: dict[str, Any]) -> bool:
    """Return whether a requested case produced publishable measurement evidence."""
    if record.get("status") != "measured":
        return False
    validation_overhead = record.get("validation_overhead")
    if validation_overhead is not None and validation_overhead.get("status") != "measured":
        return False
    process_statistics = record.get("process_statistics")
    if process_statistics is not None and process_statistics.get("status") != "measured":
        return False
    cold_compilation = record.get("cold_compilation")
    if cold_compilation is not None and cold_compilation.get("status") != "measured":
        return False
    return record.get("backend") == "scipy" or record.get("acceptance_eligible") is True


def _triton_ineligibility(arguments: argparse.Namespace) -> str | None:
    """Explain why the current CUDA device cannot give Triton evidence, or return ``None``.

    Only a run that measures the Triton backend or expects the public call to use it is
    affected; a ``public_cuda``-only run still measures a legacy build or the SciPy fallback.
    """
    wants_triton = "triton" in getattr(arguments, "backends", ()) or (
        getattr(arguments, "expect_implementation", None) == "triton"
    )
    if not wants_triton:
        return None
    device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    if capability >= _MINIMUM_TRITON_COMPUTE_CAPABILITY:
        return None
    minimum = ".".join(map(str, _MINIMUM_TRITON_COMPUTE_CAPABILITY))
    return (
        f"{torch.cuda.get_device_name(device_index)} (cuda:{device_index}) has compute capability "
        f"{capability[0]}.{capability[1]}; the Triton backend requires {minimum} or newer, "
        "so the GPU cases were skipped"
    )


def _skip_run(
    status: str,
    details: list[str],
    run_id: str,
    arguments: argparse.Namespace,
    output: Path,
    run_metadata: dict[str, Any],
) -> None:
    """Record why no case ran, and fail closed unless the run is an explicit diagnostic."""
    record = _status_record(status, run_id, getattr(arguments, "label", None))
    if details:
        record["details"] = details
    record["run_metadata"] = {**run_metadata, "gpu_state_end": _nvidia_smi_metadata()}
    _emit(record, output)
    _print_summary([record], output)
    if not getattr(arguments, "allow_incomplete", False):
        raise SystemExit(1)


def _benchmark_progress(
    cases: list[tuple[int, str, str, int, int, int, str]], *, interactive: bool
) -> Iterable[tuple[int, str, str, int, int, int, str]]:
    """Yield cases with a live terminal bar or up to twenty completed-case log lines."""
    if interactive:
        yield from tqdm(cases, desc="Benchmark cases", unit="case")
        return

    total = len(cases)
    interval = max(1, (total + 19) // 20)
    for completed, case in enumerate(cases, start=1):
        yield case
        if completed % interval == 0 or completed == total:
            print(f"Benchmark cases: {completed}/{total}", file=sys.stderr, flush=True)


def main() -> None:
    """Run the configured matrix and emit structured evidence or skip rows."""
    arguments = _arguments()
    worker_case = getattr(arguments, "worker_case", None)
    if worker_case:
        record = _worker_case_record(json.loads(worker_case))
        print(f"{_WORKER_RECORD_PREFIX}{json.dumps(record, sort_keys=True)}", flush=True)
        return

    run_id = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    output = _output_path(arguments.output, run_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    records: list[dict[str, Any]] = []
    run_metadata = _run_metadata()
    requires_cuda = any(backend != "scipy" for backend in getattr(arguments, "backends", ("public_cuda",)))
    if requires_cuda and not torch.cuda.is_available():
        _skip_run("gpu_skipped", [], run_id, arguments, output, run_metadata)
        return
    ineligibility = _triton_ineligibility(arguments) if requires_cuda else None
    if ineligibility is not None:
        _skip_run("gpu_ineligible", [ineligibility], run_id, arguments, output, run_metadata)
        return

    cases = _case_matrix(arguments)
    for process_round, backend, validation, batch, workers, tasks, dtype_name in _benchmark_progress(
        cases, interactive=sys.stderr.isatty()
    ):
        specification = {
            "run_id": run_id,
            "backend": backend,
            "validation": validation,
            "batch": batch,
            "workers": workers,
            "tasks": tasks,
            "dtype_name": dtype_name,
            "seed": arguments.seed,
            "warmup": arguments.warmup,
            "repetitions": arguments.repetitions,
            "run_label": getattr(arguments, "label", None),
            "process_round": process_round,
            "cold_samples_required": arguments.process_rounds,
            "run_metadata": run_metadata,
            "worker_timeout_seconds": getattr(arguments, "worker_timeout_seconds", _DEFAULT_WORKER_TIMEOUT_SECONDS),
        }
        record = _isolated_case_record(specification)
        records.append(record)

    gpu_state_end = _nvidia_smi_metadata()
    for record in records:
        record.setdefault("run_metadata", run_metadata)["gpu_state_end"] = gpu_state_end
    _attach_cpu_speedups(records)
    _attach_validation_overheads(records)
    _attach_process_statistics(records, arguments.process_rounds)
    case_records = list(records)
    incomplete = any(not _case_is_complete(record) for record in case_records)
    if arguments.allow_incomplete:
        for record in case_records:
            record["acceptance_eligible"] = False
            record["acceptance_exclusion"] = "diagnostic_allow_incomplete"
    identity_failures = _identity_failures(records, arguments)
    if identity_failures:
        records.append(
            {
                **_status_record("identity_failed", run_id, getattr(arguments, "label", None)),
                "details": identity_failures,
                "run_metadata": {**run_metadata, "gpu_state_end": gpu_state_end},
            }
        )
    for record in records:
        _emit(record, output)
    _print_summary(records, output)
    if identity_failures or (incomplete and not getattr(arguments, "allow_incomplete", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
