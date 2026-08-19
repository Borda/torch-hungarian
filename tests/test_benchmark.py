"""Regression tests for benchmark correctness gates and replay diagnostics."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_benchmark_module(module_name: str = "torch_linear_assignment_benchmark") -> ModuleType:
    """Load the standalone benchmark script without making tests a package."""
    path = Path(__file__).with_name("benchmark.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark_module()


def test_benchmark_import_requires_only_public_package_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent the copied benchmark from requiring implementation-private helpers on PyPI 0.0.6."""
    import torch_linear_assignment.assignment as assignment_module

    for name in (
        "_batch_linear_assignment_cuda_legacy",
        "_load_legacy_backend",
        "_load_triton_backend",
        "batch_linear_assignment_cpu",
    ):
        monkeypatch.delattr(assignment_module, name, raising=False)

    module = _load_benchmark_module("torch_linear_assignment_benchmark_public_only")

    assert callable(module.batch_linear_assignment)
    assert callable(module._scipy_oracle)


def test_default_backends_compare_scipy_cpu_and_public_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent default evidence from benchmarking only private implementation variants."""
    monkeypatch.setattr(sys, "argv", ["benchmark.py"])

    arguments = benchmark._arguments()

    assert arguments.backends == ["scipy", "public_cuda"]


def test_arguments_preserve_an_optional_package_run_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent copied-package benchmark evidence from losing its user-provided identity."""
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--label", "pypi-0.0.6"])

    arguments = benchmark._arguments()

    assert arguments.label == "pypi-0.0.6"


def test_cpu_device_metadata_and_versions_identify_the_installed_package() -> None:
    """Prevent SciPy rows from claiming CUDA or hiding the package version under test."""
    assert benchmark._device_metadata(torch.device("cpu")) == {"type": "cpu"}
    assert "package" in benchmark._versions()


def test_public_cuda_backend_uses_the_installed_public_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent the portable CUDA lane from silently selecting a current private backend."""
    expected = torch.tensor([[1, 0]], dtype=torch.long)
    calls: list[torch.Tensor] = []

    def public_solver(cost: torch.Tensor) -> torch.Tensor:
        calls.append(cost)
        return expected

    monkeypatch.setattr(benchmark, "batch_linear_assignment", public_solver)
    operation = benchmark._backend_callable("public_cuda", "full")
    cost = torch.tensor([[[4.0, 1.0], [2.0, 3.0]]])

    assert operation is not None
    assert torch.equal(operation(cost), expected)
    assert calls == [cost]


@pytest.mark.parametrize(
    ("has_triton", "has_legacy", "capability", "expected"),
    [
        pytest.param(True, True, (8, 0), "triton", id="current-ampere"),
        pytest.param(True, True, (7, 5), "scipy_fallback", id="current-pre-ampere"),
        pytest.param(False, True, (7, 5), "legacy_cuda", id="pypi-legacy"),
        pytest.param(False, False, (8, 0), "scipy_fallback", id="no-gpu-backend"),
    ],
)
def test_public_cuda_implementation_identifies_the_installed_backend(
    monkeypatch: pytest.MonkeyPatch,
    has_triton: bool,
    has_legacy: bool,
    capability: tuple[int, int],
    expected: str,
) -> None:
    """Prevent public CUDA timing rows from obscuring legacy, Triton, or fallback execution."""

    def find_spec(name: str) -> object | None:
        return (
            object()
            if (name.endswith("._triton") and has_triton) or (name.endswith("._backend") and has_legacy)
            else None
        )

    monkeypatch.setattr(benchmark.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)

    assert benchmark._backend_implementation("public_cuda", torch.device("cuda:0")) == expected


def test_attach_cpu_speedups_pairs_measured_public_cuda_with_scipy() -> None:
    """Prevent GPU timings from omitting their required comparable CPU speedups."""
    records = [
        {
            "backend": "scipy",
            "status": "measured",
            "shape": {"batch": 16, "workers": 300, "tasks": 600},
            "input_dtype": "float32",
            "seed": 32,
            "timing": {"cold_first_call_ms": 20.0, "warm_ms": {"median": 10.0, "p95": 12.0}},
        },
        {
            "backend": "public_cuda",
            "status": "measured",
            "shape": {"batch": 16, "workers": 300, "tasks": 600},
            "input_dtype": "float32",
            "seed": 32,
            "timing": {"cold_first_call_ms": 5.0, "warm_ms": {"median": 2.0, "p95": 3.0}},
        },
    ]

    benchmark._attach_cpu_speedups(records)

    assert records[1]["speedup_vs_scipy_cpu"] == {
        "cold_first_call": 4.0,
        "warm_median": 5.0,
        "warm_p95": 4.0,
    }


def test_format_table_keeps_cold_time_and_paired_warm_speedup_visible() -> None:
    """Prevent compact output from hiding the required cold latency or CPU/GPU comparison."""
    table = benchmark._format_table(
        [
            {
                "run_label": "development-triton",
                "backend": "public_cuda",
                "implementation": "triton",
                "validation_mode": "full",
                "input_dtype": "float32",
                "shape": {"batch": 16, "workers": 300, "tasks": 600},
                "status": "measured",
                "timing": {
                    "cold_first_call_ms": 50.0,
                    "warm_ms": {"median": 2.0, "p95": 3.0},
                },
                "speedup_vs_scipy_cpu": {"warm_median": 5.0},
                "parity": {"exact": True},
            }
        ]
    )

    assert "cold ms" in table
    assert "CPU/GPU warm" in table
    assert "development-triton" in table
    assert "50.0" in table
    assert "5.0x" in table


def test_format_table_delegates_alignment_to_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent the benchmark from regaining custom table-width machinery."""
    calls: list[tuple[list[list[str]], list[str]]] = []

    class Frame:
        def to_string(self, *, index: bool, justify: str) -> str:
            assert index is False
            assert justify == "left"
            return "pandas table"

    def data_frame(rows: list[list[str]], *, columns: list[str]) -> Frame:
        calls.append((rows, columns))
        return Frame()

    monkeypatch.setattr(benchmark.pd, "DataFrame", data_frame)

    table = benchmark._format_table([{"status": "gpu_skipped"}])

    assert table == "pandas table"
    assert calls[0][0][0][6] == "gpu_skipped"
    assert calls[0][1][0:3] == ["run", "backend", "validation"]


def test_parity_diagnostics_report_replayable_objective_mismatch() -> None:
    """Prevent parity failures from omitting the matrix and objective evidence needed for replay."""
    cost = torch.tensor(
        [
            [[1.0, 8.0], [7.0, 2.0]],
            [[4.0, 3.0], [2.0, 5.0]],
        ],
        dtype=torch.bfloat16,
    )
    expected = torch.tensor([[0, 1], [1, 0]])
    actual = torch.tensor([[0, 1], [0, 1]])

    diagnostics = benchmark._parity_diagnostics(
        cost=cost,
        expected=expected,
        actual=actual,
        result_call="timed:2",
        expected_device=cost.device,
    )

    assert diagnostics == {
        "input_sha256": hashlib.sha256(cost.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "result_call": "timed:2",
        "contract_violations": [],
        "output_contract": {
            "expected_shape": [2, 2],
            "actual_shape": [2, 2],
            "expected_dtype": "torch.int64",
            "actual_dtype": "torch.int64",
            "expected_device": "cpu",
            "actual_device": "cpu",
            "assignment_range": [-1, 1],
        },
        "mismatch_count": 1,
        "mismatched_batch_indices": [1],
        "expected_objectives": [5.0],
        "actual_objectives": [9.0],
        "first_mismatch": {
            "batch_index": 1,
            "expected_assignment": [1, 0],
            "actual_assignment": [0, 1],
        },
    }
    assert (
        benchmark._parity_diagnostics(
            cost=cost,
            expected=expected,
            actual=expected,
            result_call="cold",
            expected_device=cost.device,
        )
        is None
    )


@pytest.mark.parametrize(
    ("actual", "violation"),
    [
        pytest.param(torch.tensor([[0, 1]], dtype=torch.int32), "dtype", id="wrong-dtype"),
        pytest.param(torch.tensor([[0]]), "shape", id="wrong-shape"),
        pytest.param(torch.tensor([[0, 2]]), "assignment_range", id="out-of-range"),
    ],
)
def test_parity_diagnostics_report_output_contract_violations(
    actual: torch.Tensor,
    violation: str,
) -> None:
    """Prevent malformed backend outputs from crashing or bypassing benchmark evidence."""
    cost = torch.tensor([[[1.0, 8.0], [7.0, 2.0]]])
    expected = torch.tensor([[0, 1]])

    diagnostics = benchmark._parity_diagnostics(
        cost=cost,
        expected=expected,
        actual=actual,
        result_call="cold",
        expected_device=cost.device,
    )

    assert diagnostics is not None
    assert diagnostics["contract_violations"] == [violation]
    assert diagnostics["output_contract"]["expected_shape"] == [1, 2]
    assert diagnostics["output_contract"]["actual_shape"] == list(actual.shape)


@pytest.mark.parametrize(
    ("failure_call", "expected_result_call"),
    [
        pytest.param(2, "warmup:0", id="warmup"),
        pytest.param(3, "timed:0", id="timed"),
    ],
)
def test_case_record_rejects_mismatch_after_cold_call(
    monkeypatch: pytest.MonkeyPatch,
    failure_call: int,
    expected_result_call: str,
) -> None:
    """Prevent a correct cold result from hiding a later nondeterministic solver failure."""
    calls = 0

    def operation(cost: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        result = benchmark._scipy_oracle(cost)
        if calls == failure_call:
            result = result.clone()
            result[0, 0] = (result[0, 0] + 1) % cost.shape[2]
        return result

    monkeypatch.setattr(benchmark, "_backend_callable", lambda _backend, _validation: operation)
    monkeypatch.setattr(benchmark, "_device_metadata", lambda *_args: {"type": "cpu"})

    record = benchmark._case_record(
        run_id="test-run",
        backend="scipy",
        validation="full",
        batch=1,
        workers=3,
        tasks=5,
        dtype_name="float32",
        seed=32,
        warmup=1,
        repetitions=1,
    )

    assert record["status"] == "parity_failed"
    assert record["acceptance_eligible"] is False
    assert record["parity"] == {"oracle": "scipy_promoted", "exact": False}
    assert record["diagnostics"]["result_call"] == expected_result_call
    assert record["diagnostics"]["mismatched_batch_indices"] == [0]
    assert record["diagnostics"]["expected_objectives"] != record["diagnostics"]["actual_objectives"]


def test_case_record_rejects_correct_values_with_wrong_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent exact-valued int32 indices from satisfying the int64 assignment contract."""

    def operation(cost: torch.Tensor) -> torch.Tensor:
        return benchmark._scipy_oracle(cost).to(torch.int32)

    monkeypatch.setattr(benchmark, "_backend_callable", lambda _backend, _validation: operation)
    monkeypatch.setattr(benchmark, "_device_metadata", lambda *_args: {"type": "cpu"})

    record = benchmark._case_record(
        run_id="test-run",
        backend="scipy",
        validation="full",
        batch=1,
        workers=3,
        tasks=5,
        dtype_name="float32",
        seed=32,
        warmup=0,
        repetitions=1,
    )

    assert record["status"] == "parity_failed"
    assert record["diagnostics"]["contract_violations"] == ["dtype"]


def test_emit_saves_jsonl_without_printing_raw_records(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Prevent machine-readable evidence from replacing the human-readable console table."""
    output = tmp_path / "nested" / "results.jsonl"
    output.parent.mkdir()
    records = [
        {"status": "backend_unavailable", "backend": "legacy_cuda"},
        {"status": "measured", "backend": "triton"},
    ]

    for record in records:
        benchmark._emit(record, output)

    assert capsys.readouterr().out == ""
    assert [json.loads(line) for line in output.read_text().splitlines()] == records


def test_format_table_labels_timings_and_parity() -> None:
    """Prevent compact output from presenting latency without backend or parity context."""
    records = [
        {
            "run_label": "development-triton",
            "backend": "triton",
            "validation_mode": "full",
            "input_dtype": "float32",
            "shape": {"batch": 16, "workers": 300, "tasks": 600},
            "status": "measured",
            "timing": {"warm_ms": {"median": 12.3456, "p95": 13.4567}},
            "parity": {"exact": True},
        },
        {
            "run_label": "development-triton",
            "backend": "triton",
            "validation_mode": "full",
            "input_dtype": "float32",
            "shape": {"batch": 624, "workers": 300, "tasks": 600},
            "status": "parity_failed",
            "parity": {"exact": False},
        },
    ]

    table = benchmark._format_table(records)

    header, measured, failed = table.splitlines()
    assert all(label in header for label in ("run", "backend", "validation", "dtype"))
    assert measured.split()[0:5] == ["development-triton", "triton", "full", "float32", "16"]
    assert "300 x 600" in measured
    assert "measured" in measured
    assert "12.3" in measured
    assert "624" in table
    assert "parity_failed" in failed
    assert table.count("yes") == 1
    assert table.count("no") == 1


def test_default_output_path_includes_run_id() -> None:
    """Prevent default benchmark runs from overwriting evidence from another run."""
    assert benchmark._output_path(None, "2026-08-19T11-24-00Z") == Path("benchmark-results-2026-08-19T11-24-00Z.jsonl")
    assert benchmark._output_path(Path("custom.jsonl"), "ignored") == Path("custom.jsonl")


def test_main_writes_default_jsonl_for_gpu_skip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Prevent a default invocation from losing its structured no-GPU evidence."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark, "_arguments", lambda: argparse.Namespace(output=None))
    monkeypatch.setattr(benchmark.time, "strftime", lambda *_args: "2026-08-19T11-24-00Z")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    benchmark.main()

    output = tmp_path / "benchmark-results-2026-08-19T11-24-00Z.jsonl"
    record = json.loads(output.read_text())
    assert record["status"] == "gpu_skipped"
    assert record["acceptance_eligible"] is False
    assert f"JSONL results: {output}" in capsys.readouterr().out


def test_main_exits_nonzero_after_parity_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Prevent parity-failed JSONL evidence from producing a successful validation command."""
    output = tmp_path / "results.jsonl"
    arguments = argparse.Namespace(
        backends=["triton"],
        workers=2,
        tasks=[2],
        batches=[1],
        dtypes=["float32"],
        validation_modes=["full"],
        seed=32,
        warmup=0,
        repetitions=1,
        output=output,
    )
    monkeypatch.setattr(benchmark, "_arguments", lambda: arguments)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        benchmark,
        "_case_record",
        lambda **_kwargs: {"status": "parity_failed", "acceptance_eligible": False},
    )

    with pytest.raises(SystemExit, match="1"):
        benchmark.main()

    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"status": "parity_failed", "acceptance_eligible": False}
    ]
    stdout = capsys.readouterr().out
    assert "status        " in stdout
    assert "parity_failed" in stdout
    assert "exact parity" in stdout
    assert f"JSONL results: {output.resolve()}" in stdout


def test_main_tracks_the_complete_case_matrix_with_one_progress_bar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prevent progress reporting from omitting cases or nesting noisy bars."""
    arguments = argparse.Namespace(
        backends=["scipy", "triton"],
        workers=2,
        tasks=[2, 3],
        batches=[1, 4],
        dtypes=["float32", "float64"],
        validation_modes=["full", "off"],
        seed=32,
        warmup=0,
        repetitions=1,
        output=tmp_path / "results.jsonl",
    )
    progress_calls: list[tuple[list[tuple[str, str, int, int, str]], dict[str, str]]] = []

    def track_progress(
        cases: list[tuple[str, str, int, int, str]],
        **options: str,
    ) -> list[tuple[str, str, int, int, str]]:
        materialized = list(cases)
        progress_calls.append((materialized, options))
        return materialized

    monkeypatch.setattr(benchmark, "tqdm", track_progress)
    monkeypatch.setattr(benchmark, "_arguments", lambda: arguments)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        benchmark,
        "_case_record",
        lambda **_kwargs: {"status": "measured", "acceptance_eligible": True},
    )
    monkeypatch.setattr(benchmark, "_attach_cpu_speedups", lambda _records: None)
    monkeypatch.setattr(benchmark, "_emit", lambda _record, _output: None)
    monkeypatch.setattr(benchmark, "_print_summary", lambda _records, _output: None)

    benchmark.main()

    assert len(progress_calls) == 1
    cases, options = progress_calls[0]
    assert len(cases) == 24
    assert cases[0] == ("scipy", "full", 1, 2, "float32")
    assert cases[-1] == ("triton", "off", 4, 3, "float64")
    assert options == {"desc": "Benchmark cases", "unit": "case"}
