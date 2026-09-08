"""Regression tests for benchmark correctness gates and replay diagnostics."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    assert arguments.seed == 320
    assert arguments.process_rounds == 3
    assert arguments.backend_order == "alternate"
    assert arguments.allow_incomplete is False


def test_arguments_preserve_an_optional_package_run_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent copied-package benchmark evidence from losing its user-provided identity."""
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--label", "pypi-0.0.6"])

    arguments = benchmark._arguments()

    assert arguments.label == "pypi-0.0.6"


def test_cpu_device_metadata_and_versions_identify_the_installed_package() -> None:
    """Prevent SciPy rows from claiming CUDA or hiding the package version under test."""
    assert benchmark._device_metadata(torch.device("cpu")) == {"type": "cpu"}
    assert "package" in benchmark._versions()


def test_source_metadata_hashes_benchmark_and_imported_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent development artifacts from losing source and installed-module identity."""
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="abc123\n"),
    )

    source = benchmark._source_metadata()

    assert source["benchmark_git_revision"] == "abc123"
    assert len(source["benchmark_sha256"]) == 64
    assert source["package_module_origin"]
    assert len(source["package_module_sha256"]) == 64


def test_source_metadata_accepts_original_path_and_revision_from_copied_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent import-isolated benchmark copies from losing repository provenance."""
    original = Path(benchmark.__file__).with_name("benchmark.py").resolve()
    monkeypatch.setenv("TLA_BENCHMARK_SOURCE_PATH", str(original))
    monkeypatch.setenv("TLA_BENCHMARK_GIT_REVISION", "copied-run-revision")
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("environment provenance must avoid git subprocess"),
    )

    source = benchmark._source_metadata()

    assert source["benchmark_path"] == str(original)
    assert source["benchmark_git_revision"] == "copied-run-revision"


def test_public_cuda_backend_uses_the_installed_public_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent the portable CUDA lane from silently selecting a current private backend."""
    expected = torch.tensor([[1, 0]], dtype=torch.long)
    calls: list[torch.Tensor] = []

    def public_solver(cost: torch.Tensor) -> torch.Tensor:
        """Record the tensor routed through the public CUDA adapter."""
        calls.append(cost)
        return expected

    monkeypatch.setattr(benchmark, "batch_linear_assignment", public_solver)
    operation = benchmark._backend_callable("public_cuda", "full")
    cost = torch.tensor([[[4.0, 1.0], [2.0, 3.0]]])

    assert operation is not None
    assert torch.equal(operation(cost), expected)
    assert calls == [cost]


@pytest.mark.parametrize(
    ("loaded_module", "expected"),
    [
        pytest.param("torch_linear_assignment._triton", "triton", id="current-triton"),
        pytest.param("torch_linear_assignment._backend", "legacy_cuda", id="pypi-legacy"),
        pytest.param(None, "scipy_fallback", id="no-loaded-gpu-backend"),
    ],
)
def test_public_cuda_implementation_identifies_the_executed_backend(
    monkeypatch: pytest.MonkeyPatch,
    loaded_module: str | None,
    expected: str,
) -> None:
    """Prevent a discoverable module from standing in for the backend actually loaded."""
    monkeypatch.delitem(sys.modules, "torch_linear_assignment._triton", raising=False)
    monkeypatch.delitem(sys.modules, "torch_linear_assignment._backend", raising=False)
    if loaded_module is not None:
        monkeypatch.setitem(sys.modules, loaded_module, ModuleType(loaded_module))

    assert benchmark._backend_implementation("public_cuda", torch.device("cuda:0")) == expected


def test_discoverable_triton_module_does_not_mislabel_scipy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent source discoverability from standing in for a successful public dispatch."""
    monkeypatch.setattr(benchmark.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 0))
    monkeypatch.delitem(sys.modules, "torch_linear_assignment._triton", raising=False)

    assert benchmark._backend_implementation("public_cuda", torch.device("cuda:0")) == "scipy_fallback"


def test_loaded_implementation_metadata_hashes_the_executed_module(tmp_path: Path) -> None:
    """Prevent a backend label from lacking the loaded implementation artifact identity."""
    module_path = tmp_path / "_triton.py"
    module_path.write_text("VALUE = 1\n", encoding="utf-8")
    module = ModuleType("torch_linear_assignment._triton")
    module.__file__ = str(module_path)
    sys.modules[module.__name__] = module
    try:
        metadata = benchmark._implementation_metadata("triton")
    finally:
        sys.modules.pop(module.__name__, None)

    assert metadata == {
        "module": "torch_linear_assignment._triton",
        "origin": str(module_path.resolve()),
        "sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
    }


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
        """Stand in for the small DataFrame surface used by the formatter."""

        def to_string(self, *, index: bool, justify: str) -> str:
            """Validate formatting options and return a stable sentinel table."""
            assert index is False
            assert justify == "left"
            return "pandas table"

    def data_frame(rows: list[list[str]], *, columns: list[str]) -> Frame:
        """Capture table rows and columns passed to pandas."""
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


def test_parity_diagnostics_report_wrong_output_device() -> None:
    """Prevent exact CPU values from satisfying a CUDA-result device contract."""
    cost = torch.tensor([[[1.0, 8.0], [7.0, 2.0]]])
    expected = torch.tensor([[0, 1]])

    diagnostics = benchmark._parity_diagnostics(
        cost=cost,
        expected=expected,
        actual=expected.clone(),
        result_call="cold",
        expected_device=torch.device("meta"),
    )

    assert diagnostics is not None
    assert diagnostics["contract_violations"] == ["device"]


def test_time_call_synchronizes_immediately_before_and_after_cuda_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent asynchronous CUDA work from escaping either sample boundary."""
    events: list[str] = []
    cost = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: events.append("synchronize"))

    elapsed_ms, result = benchmark._time_call(lambda _cost: events.append("operation") or "result", cost)

    assert elapsed_ms >= 0
    assert result == "result"
    assert events == ["synchronize", "operation", "synchronize"]


def test_case_record_times_cold_before_constructing_the_scipy_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent correctness setup from warming the backend's claimed first solver call."""
    events: list[str] = []
    expected = torch.tensor([[0, 1]])

    def operation(_cost: torch.Tensor) -> torch.Tensor:
        """Record solver execution order and return the expected assignment."""
        events.append("operation")
        return expected

    def oracle(_cost: torch.Tensor) -> torch.Tensor:
        """Record oracle construction order and return the expected assignment."""
        events.append("oracle")
        return expected

    def time_call(callable_: object, cost: torch.Tensor) -> tuple[float, torch.Tensor]:
        """Record timer entry before invoking the supplied solver."""
        events.append("timer")
        return 1.0, callable_(cost)  # type: ignore[operator]

    monkeypatch.setattr(benchmark, "_backend_callable", lambda _backend, _validation: operation)
    monkeypatch.setattr(benchmark, "_scipy_oracle", oracle)
    monkeypatch.setattr(benchmark, "_time_call", time_call)

    benchmark._case_record(
        run_id="test-run",
        backend="scipy",
        validation="full",
        batch=1,
        workers=2,
        tasks=2,
        dtype_name="float32",
        seed=320,
        warmup=0,
        repetitions=1,
    )

    assert events[:3] == ["timer", "operation", "oracle"]


def test_case_record_generates_float64_inputs_natively(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent float64 evidence from containing only converted float32 random values."""
    original_randn = torch.randn
    generated_dtypes: list[torch.dtype | None] = []

    def randn(*args: object, **kwargs: object) -> torch.Tensor:
        """Capture the requested random dtype before delegating to Torch."""
        generated_dtypes.append(kwargs.get("dtype"))  # type: ignore[arg-type]
        return original_randn(*args, **kwargs)

    monkeypatch.setattr(torch, "randn", randn)

    record = benchmark._case_record(
        run_id="test-run",
        backend="scipy",
        validation="full",
        batch=1,
        workers=2,
        tasks=2,
        dtype_name="float64",
        seed=320,
        warmup=0,
        repetitions=1,
    )

    assert record["status"] == "measured"
    assert generated_dtypes == [torch.float64]
    assert record["input_dtype"] == "float64"
    assert len(record["input_sha256"]) == 64
    assert record["timing"]["warm_ms"]["samples"]
    assert record["measurement_scope"]["kind"] == "cpu_solver_latency"
    assert "CUDA completion synchronization" not in record["measurement_scope"]["included"]
    assert "host-to-device input transfer" not in record["measurement_scope"]["excluded"]


def test_case_matrix_interleaves_backends_and_reverses_each_process_round() -> None:
    """Prevent backend-major ordering from systematically biasing one implementation."""
    arguments = argparse.Namespace(
        backends=["scipy", "public_cuda"],
        workers=2,
        tasks=[2],
        batches=[1],
        dtypes=["float32"],
        validation_modes=["full"],
        process_rounds=2,
        backend_order="alternate",
    )

    cases = benchmark._case_matrix(arguments)

    assert [case[1] for case in cases] == ["scipy", "public_cuda", "public_cuda", "scipy"]
    assert [case[0] for case in cases] == [0, 0, 1, 1]


def test_process_statistics_require_every_fresh_process_round() -> None:
    """Prevent one cold sample from being presented as repeated process-cold evidence."""
    records = [
        {
            "backend": "public_cuda",
            "implementation": "triton",
            "status": "measured",
            "process_round": process_round,
            "process_isolated": True,
            "validation_mode": "full",
            "shape": {"batch": 1, "workers": 2, "tasks": 2},
            "input_dtype": "float32",
            "seed": 320,
            "input_sha256": "same-input",
            "timing": {
                "cold_first_call_ms": cold,
                "warm_ms": {"median": warm, "p95": warm, "samples": [warm]},
            },
            "acceptance_eligible": True,
        }
        for process_round, cold, warm in ((0, 30.0, 10.0), (1, 24.0, 8.0), (2, 27.0, 9.0))
    ]

    benchmark._attach_process_statistics(records, required_rounds=3)

    statistics = records[0]["process_statistics"]
    assert statistics["status"] == "measured"
    assert statistics["rounds"] == [0, 1, 2]
    assert statistics["cold_first_call_ms"] == {
        "samples": [30.0, 24.0, 27.0],
        "median": 27.0,
        "p95": 30.0,
    }
    assert statistics["warm_median_ms"]["median"] == 9.0
    assert records[0]["cold_compilation"]["cache_policy"] == "empty_unique_triton_cache"


def test_process_statistics_do_not_claim_triton_cache_isolation_for_scipy() -> None:
    """Prevent CPU cold rows from inheriting Triton-specific cache claims."""
    record = {
        "backend": "scipy",
        "implementation": "scipy_cpu",
        "status": "measured",
        "process_round": 0,
        "process_isolated": True,
        "validation_mode": "full",
        "shape": {"batch": 1, "workers": 2, "tasks": 2},
        "input_dtype": "float32",
        "seed": 320,
        "input_sha256": "same-input",
        "timing": {
            "cold_first_call_ms": 1.0,
            "warm_ms": {"median": 0.5, "p95": 0.5, "samples": [0.5]},
        },
        "acceptance_eligible": False,
    }

    benchmark._attach_process_statistics([record], required_rounds=1)

    cold = record["cold_compilation"]
    assert cold["cache_policy"] == "not_applicable"
    assert cold["definition"] == ("fresh_process_first_scipy_solver_call_after_package_import_and_input_setup")


def test_process_statistics_reject_missing_rounds() -> None:
    """Prevent incomplete fresh-process sampling from remaining acceptance eligible."""
    record = {
        "backend": "public_cuda",
        "implementation": "triton",
        "status": "measured",
        "process_round": 0,
        "process_isolated": True,
        "validation_mode": "full",
        "shape": {"batch": 1, "workers": 2, "tasks": 2},
        "input_dtype": "float32",
        "seed": 320,
        "input_sha256": "same-input",
        "timing": {
            "cold_first_call_ms": 30.0,
            "warm_ms": {"median": 10.0, "p95": 10.0, "samples": [10.0]},
        },
        "acceptance_eligible": True,
    }

    benchmark._attach_process_statistics([record], required_rounds=3)

    assert record["process_statistics"]["status"] == "incomplete"
    assert record["acceptance_eligible"] is False
    assert benchmark._case_is_complete(record) is False


def test_validation_overhead_uses_all_modes_with_one_identical_input() -> None:
    """Prevent validation timing deltas from comparing different generated costs."""
    records = [
        {
            "backend": "triton",
            "status": "measured",
            "process_round": 0,
            "validation_mode": mode,
            "shape": {"batch": 1, "workers": 2, "tasks": 2},
            "input_dtype": "float32",
            "seed": 320,
            "input_sha256": "same-input",
            "timing": {"warm_ms": {"median": median, "samples": [median]}},
        }
        for mode, median in {
            "off": 10.0,
            "nonfinite_only": 11.0,
            "infeasibility_flag_only": 12.0,
            "full": 14.0,
        }.items()
    ]

    benchmark._attach_validation_overheads(records)

    overhead = records[-1]["validation_overhead"]
    assert overhead["status"] == "measured"
    assert overhead["input_sha256"] == "same-input"
    assert overhead["absolute_vs_off_ms"] == {
        "nonfinite_only": 1.0,
        "infeasibility_flag_only": 2.0,
        "full": 4.0,
    }
    assert overhead["percent_vs_off"] == {
        "nonfinite_only": 10.0,
        "infeasibility_flag_only": 20.0,
        "full": 40.0,
    }
    assert overhead["interaction_ms"] == 1.0
    assert overhead["interaction_percent_vs_off"] == 10.0


def test_incomplete_validation_overhead_is_not_acceptance_evidence() -> None:
    """Prevent a missing validation mode from silently closing the overhead check."""
    records = [
        {
            "backend": "triton",
            "status": "measured",
            "process_round": 0,
            "validation_mode": "off",
            "shape": {"batch": 1, "workers": 2, "tasks": 2},
            "input_dtype": "float32",
            "seed": 320,
            "input_sha256": "same-input",
            "timing": {"warm_ms": {"median": 10.0, "samples": [10.0]}},
            "acceptance_eligible": True,
        }
    ]

    benchmark._attach_validation_overheads(records)

    assert records[0]["validation_overhead"]["status"] == "incomplete"
    assert benchmark._case_is_complete(records[0]) is False


def test_case_record_separates_cold_and_warm_memory_boundaries() -> None:
    """Prevent a single peak allocation value from conflating compilation and warm calls."""
    record = benchmark._case_record(
        run_id="test-run",
        backend="scipy",
        validation="full",
        batch=1,
        workers=2,
        tasks=2,
        dtype_name="float32",
        seed=320,
        warmup=0,
        repetitions=1,
        process_isolated=True,
        cold_cache_policy="empty_unique_triton_cache",
    )

    assert record["memory"]["cold"]["status"] == "not_cuda"
    assert record["memory"]["warm"]["status"] == "not_cuda"
    assert set(record["memory"]["cold"]) >= {
        "allocated_before_bytes",
        "allocated_after_bytes",
        "peak_allocated_bytes",
        "peak_allocated_delta_bytes",
    }
    assert record["cold_compilation"]["cache_policy"] == "not_applicable"
    assert record["cold_compilation"]["cache_directory"] is None
    assert record["timing"]["cold_method"] == (
        "fresh_process_first_scipy_solver_call_after_package_import_and_input_setup"
    )


def test_isolated_case_uses_a_unique_triton_cache_and_records_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent cold evidence from inheriting a prior worker's Triton disk cache."""
    observed: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        """Capture the isolated-worker command and its cache environment."""
        observed["command"] = command
        observed["cache_dir"] = kwargs["env"]["TRITON_CACHE_DIR"]  # type: ignore[index]
        return SimpleNamespace(
            returncode=0,
            stdout='__TLA_BENCHMARK_RECORD__={"status": "measured"}\n',
            stderr="",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    specification = {
        "run_id": "test-run",
        "backend": "triton",
        "validation": "full",
        "batch": 1,
        "workers": 2,
        "tasks": 2,
        "dtype_name": "float32",
        "seed": 320,
        "warmup": 0,
        "repetitions": 1,
    }

    record = benchmark._isolated_case_record(specification)

    assert record == {"status": "measured"}
    assert "--worker-case" in observed["command"]  # type: ignore[operator]
    assert Path(str(observed["cache_dir"])).exists() is False


def test_isolated_case_rejects_measured_payload_from_failed_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent a nonzero worker exit from publishing its last partial payload."""

    def run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        """Return a contradictory subprocess result for the fail-closed check."""
        return SimpleNamespace(
            returncode=7,
            stdout='__TLA_BENCHMARK_RECORD__={"status": "measured", "acceptance_eligible": true}\n',
            stderr="worker crashed after writing its payload",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    specification = {
        "run_id": "test-run",
        "backend": "triton",
        "validation": "full",
        "batch": 1,
        "workers": 2,
        "tasks": 2,
        "dtype_name": "float32",
        "seed": 320,
        "warmup": 0,
        "repetitions": 1,
    }

    record = benchmark._isolated_case_record(specification)

    assert record["status"] == "worker_failed"
    assert record["acceptance_eligible"] is False
    assert record["detail"] == "worker crashed after writing its payload"


def test_isolated_case_marks_the_fresh_cache_policy_in_worker_specification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent JSONL cold rows from omitting their cache-isolation definition."""
    captured: dict[str, object] = {}

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        """Capture the serialized worker specification."""
        captured["specification"] = json.loads(command[-1])
        return SimpleNamespace(
            returncode=0,
            stdout='__TLA_BENCHMARK_RECORD__={"status": "measured"}\n',
            stderr="",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    benchmark._isolated_case_record(
        {
            "run_id": "test-run",
            "backend": "triton",
            "validation": "full",
            "batch": 1,
            "workers": 2,
            "tasks": 2,
            "dtype_name": "float32",
            "seed": 320,
            "warmup": 0,
            "repetitions": 1,
        }
    )

    specification = captured["specification"]
    assert specification["process_isolated"] is True  # type: ignore[index]
    assert specification["cold_cache_policy"] == "empty_unique_triton_cache"  # type: ignore[index]


def test_nvidia_smi_metadata_is_explicit_when_command_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent absent clock/power telemetry from disappearing silently."""
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError))

    assert benchmark._nvidia_smi_metadata() == {
        "status": "unavailable",
        "reason": "nvidia-smi executable not found",
    }


def test_nvidia_smi_metadata_parses_all_requested_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent captured GPU operating-state columns from shifting silently."""
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="550.54, NVIDIA L4, 00000000:00:04.0, 2040, 6251, 68.4, 55\n",
            stderr="",
        ),
    )

    assert benchmark._nvidia_smi_metadata() == {
        "status": "measured",
        "devices": [
            {
                "driver_version": "550.54",
                "name": "NVIDIA L4",
                "pci_bus_id": "00000000:00:04.0",
                "sm_clock_mhz": "2040",
                "memory_clock_mhz": "6251",
                "power_w": "68.4",
                "temperature_c": "55",
            }
        ],
    }


def test_isolated_case_record_marks_the_worker_as_process_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent a same-process case call from being published as a process-cold result."""
    observed_specifications: list[dict[str, object]] = []

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        """Echo the worker's process-isolation marker in its result."""
        specification = json.loads(command[-1])
        observed_specifications.append(specification)
        record = {"status": "measured", "process_isolated": specification["process_isolated"]}
        return SimpleNamespace(
            returncode=0, stdout=f"noise\n{benchmark._WORKER_RECORD_PREFIX}{json.dumps(record)}\n", stderr=""
        )

    monkeypatch.setattr(benchmark.subprocess, "run", run)

    record = benchmark._isolated_case_record(
        {
            "run_id": "run",
            "backend": "scipy",
            "validation": "full",
            "batch": 1,
            "workers": 2,
            "tasks": 2,
            "dtype_name": "float32",
            "seed": 320,
            "warmup": 0,
            "repetitions": 1,
        }
    )

    assert record == {"status": "measured", "process_isolated": True}
    assert observed_specifications[0]["process_isolated"] is True


def test_identity_expectations_validate_package_and_executed_backend() -> None:
    """Prevent a correctly timed row from being accepted under the wrong installation."""
    records = [
        {
            "record_type": "case",
            "backend": "public_cuda",
            "status": "measured",
            "implementation": "legacy_cuda",
            "versions": {"package": "0.0.6"},
        }
    ]
    matching = argparse.Namespace(expect_package_version="0.0.6", expect_implementation="legacy_cuda")
    mismatching = argparse.Namespace(expect_package_version="0.1.0.dev0", expect_implementation="triton")

    assert benchmark._identity_failures(records, matching) == []
    assert benchmark._identity_failures(records, mismatching) == [
        "expected package 0.1.0.dev0, observed ['0.0.6']",
        "expected implementation triton, observed ['legacy_cuda']",
    ]


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
        """Return one deliberately corrupted assignment at the selected call."""
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
        """Return correct indices with the contractually wrong integer dtype."""
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
    monkeypatch.setattr(
        benchmark,
        "_arguments",
        lambda: argparse.Namespace(
            output=None,
            backends=["scipy", "public_cuda"],
            label=None,
            allow_incomplete=False,
            worker_case=None,
        ),
    )
    monkeypatch.setattr(benchmark.time, "strftime", lambda *_args: "2026-08-19T11-24-00Z")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit, match="1"):
        benchmark.main()

    output = tmp_path / "benchmark-results-2026-08-19T11-24-00Z.jsonl"
    record = json.loads(output.read_text())
    assert record["status"] == "gpu_skipped"
    assert record["acceptance_eligible"] is False
    assert f"JSONL results: {output}" in capsys.readouterr().out


def test_main_allows_an_explicit_diagnostic_gpu_skip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep CPU-only CI diagnostic while requiring an explicit non-acceptance opt-out."""
    output = tmp_path / "skip.jsonl"
    monkeypatch.setattr(
        benchmark,
        "_arguments",
        lambda: argparse.Namespace(
            output=output,
            backends=["scipy", "public_cuda"],
            label=None,
            allow_incomplete=True,
            worker_case=None,
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    benchmark.main()

    assert json.loads(output.read_text())["status"] == "gpu_skipped"


def test_main_invalidates_every_case_in_an_allow_incomplete_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prevent one successful row from escaping a diagnostic-only mixed run."""
    output = tmp_path / "diagnostic.jsonl"
    arguments = argparse.Namespace(
        backends=["public_cuda"],
        workers=2,
        tasks=[2, 3],
        batches=[1],
        dtypes=["float32"],
        validation_modes=["full"],
        seed=320,
        warmup=0,
        repetitions=1,
        output=output,
        process_rounds=1,
        backend_order="alternate",
        allow_incomplete=True,
        expect_package_version=None,
        expect_implementation=None,
        label=None,
        worker_case=None,
    )
    records = iter(
        [
            {
                "record_type": "case",
                "status": "measured",
                "backend": "public_cuda",
                "acceptance_eligible": True,
            },
            {
                "record_type": "case",
                "status": "worker_failed",
                "backend": "public_cuda",
                "acceptance_eligible": False,
            },
        ]
    )
    monkeypatch.setattr(benchmark, "_arguments", lambda: arguments)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark, "_run_metadata", dict)
    monkeypatch.setattr(benchmark, "_nvidia_smi_metadata", dict)
    monkeypatch.setattr(benchmark, "_isolated_case_record", lambda _specification: next(records))

    benchmark.main()

    written = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(written) == 2
    assert {record["acceptance_eligible"] for record in written} == {False}
    assert {record["acceptance_exclusion"] for record in written} == {"diagnostic_allow_incomplete"}


@pytest.mark.parametrize("status", ["backend_unavailable", "oom", "worker_failed"])
def test_main_exits_nonzero_for_every_incomplete_requested_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: str,
) -> None:
    """Prevent missing requested GPU work from producing a successful benchmark command."""
    output = tmp_path / f"{status}.jsonl"
    arguments = argparse.Namespace(
        backends=["public_cuda"],
        workers=2,
        tasks=[2],
        batches=[1],
        dtypes=["float32"],
        validation_modes=["full"],
        seed=320,
        warmup=0,
        repetitions=1,
        output=output,
        process_rounds=1,
        backend_order="alternate",
        allow_incomplete=False,
        expect_package_version=None,
        expect_implementation=None,
        label=None,
        worker_case=None,
    )
    monkeypatch.setattr(benchmark, "_arguments", lambda: arguments)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark, "_run_metadata", dict)
    monkeypatch.setattr(benchmark, "_nvidia_smi_metadata", dict)
    monkeypatch.setattr(
        benchmark,
        "_isolated_case_record",
        lambda _specification: {"record_type": "case", "status": status, "acceptance_eligible": False},
    )

    with pytest.raises(SystemExit, match="1"):
        benchmark.main()

    assert json.loads(output.read_text())["status"] == status


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
        process_rounds=1,
        backend_order="alternate",
        allow_incomplete=False,
        expect_package_version=None,
        expect_implementation=None,
        label=None,
        worker_case=None,
    )
    monkeypatch.setattr(benchmark, "_arguments", lambda: arguments)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        benchmark,
        "_isolated_case_record",
        lambda _specification: {"status": "parity_failed", "acceptance_eligible": False},
    )
    monkeypatch.setattr(benchmark, "_run_metadata", dict)
    monkeypatch.setattr(benchmark, "_nvidia_smi_metadata", dict)

    with pytest.raises(SystemExit, match="1"):
        benchmark.main()

    written = [json.loads(line) for line in output.read_text().splitlines()]
    assert written[0]["status"] == "parity_failed"
    assert written[0]["acceptance_eligible"] is False
    assert written[0]["run_metadata"] == {"gpu_state_end": {}}
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
        process_rounds=1,
        backend_order="alternate",
        allow_incomplete=False,
        expect_package_version=None,
        expect_implementation=None,
        label=None,
        worker_case=None,
    )
    progress_calls: list[tuple[list[tuple[int, str, str, int, int, str]], dict[str, str]]] = []

    def track_progress(
        cases: list[tuple[int, str, str, int, int, str]],
        **options: str,
    ) -> list[tuple[int, str, str, int, int, str]]:
        """Capture one materialized case matrix passed to the progress wrapper."""
        materialized = list(cases)
        progress_calls.append((materialized, options))
        return materialized

    monkeypatch.setattr(benchmark, "tqdm", track_progress)
    monkeypatch.setattr(benchmark, "_arguments", lambda: arguments)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        benchmark,
        "_isolated_case_record",
        lambda _specification: {"status": "measured", "acceptance_eligible": True},
    )
    monkeypatch.setattr(benchmark, "_run_metadata", dict)
    monkeypatch.setattr(benchmark, "_nvidia_smi_metadata", dict)
    monkeypatch.setattr(benchmark, "_attach_cpu_speedups", lambda _records: None)
    monkeypatch.setattr(benchmark, "_emit", lambda _record, _output: None)
    monkeypatch.setattr(benchmark, "_print_summary", lambda _records, _output: None)

    benchmark.main()

    assert len(progress_calls) == 1
    cases, options = progress_calls[0]
    assert len(cases) == 24
    assert cases[0] == (0, "scipy", "full", 1, 2, "float32")
    assert cases[-1] == (0, "triton", "off", 4, 3, "float64")
    assert options == {"desc": "Benchmark cases", "unit": "case"}
