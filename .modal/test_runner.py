"""Run the GPU test lane on Modal's serverless NVIDIA hardware.

Ported from https://github.com/Borda/affordable-GPU-CI and adapted to this
project: the reproducible ``develop`` dependency group replaces the upstream
``tests`` group, the container starts from a slim Debian image instead of the
NGC PyTorch image because the pinned PyPI PyTorch wheel already carries CUDA,
and the run is fail-closed on GPU eligibility.

The fail-closed gate matters more here than anywhere else in the suite: every
compiled-backend test in ``tests/unittests/test_triton.py`` skips itself when the host is
ineligible, so a run on an unsupported GPU would report success while proving
nothing about the Triton kernels. The gate refuses the run instead.

Usage:
    modal run .modal/test_runner.py
    modal run .modal/test_runner.py --test-path tests/unittests/test_triton.py
    modal run .modal/test_runner.py --test-path tests/integrations/
    modal run .modal/test_runner.py --pytest-args "-v -k compiled"
    MODAL_GPU=L4:2 modal run .modal/test_runner.py --min-devices 2
    modal run .modal/test_runner.py --benchmark-preset rfdetr
    modal run .modal/test_runner.py --benchmark-preset wide --benchmark-args "--process-rounds 5"

The default ``tests/`` path runs both ``tests/unittests`` and
``tests/integrations``. The image installs the ``rfdetr`` dependency group, so
the RF-DETR integration tests run on this lane instead of skipping. Benchmark presets write
their JSONL records to ``test-outputs/benchmark-<preset>.jsonl``.

Environment Variables:
    MODAL_GPU: GPU request passed to Modal, such as ``L4`` (the default),
        ``A100``, or ``L4:2`` for two devices. Compute capability 8.0 or newer
        is required, so ``T4`` (``sm_75``) is rejected by the gate rather than
        silently skipped.
    MODAL_GPU_FALLBACKS: Comma-separated types Modal may use instead when the
        test lane asks for ``L4``; defaults to ``A10,L40S`` then and to none
        for any other ``MODAL_GPU``. An empty value turns them off. The
        benchmark lane never uses fallbacks, because timings are only
        comparable on one type.
    MODAL_PYTHON_VERSION: Python version of the container image.
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

import modal

#: Modal application name; stable so runs group together in the Modal dashboard.
APP_NAME = "torch-hungarian-gpu-tests"


def _gpu_request(primary: str, fallbacks: str = "") -> str | list[str]:
    """Build Modal's ``gpu`` argument from a primary request and optional fallback types.

    A comma-separated ``primary`` is already a list. A device count on the first type, as in
    ``L4:2``, is copied to each fallback so a fallback never offers fewer devices.
    """
    types = [item.strip() for item in primary.split(",") if item.strip()]
    if not types:
        raise ValueError(f"MODAL_GPU names no GPU type: {primary!r}")
    count = types[0].partition(":")[2]
    for fallback in (item.strip() for item in fallbacks.split(",") if item.strip()):
        candidate = f"{fallback}:{count}" if count and ":" not in fallback else fallback
        if candidate.upper() not in {item.upper() for item in types}:
            types.append(candidate)
    return types if len(types) > 1 else types[0]


_PRIMARY_GPU = os.environ.get("MODAL_GPU") or "L4"

#: Fallback types tried after an ``L4`` request: A10 and L40S are also compute capability 8.0 or
#: newer, so any of them can run the suite. Any other primary type gets no fallback unless
#: ``MODAL_GPU_FALLBACKS`` names some; an empty value turns them off.
_DEFAULT_FALLBACKS = "A10,L40S" if _PRIMARY_GPU.split(":")[0].strip().upper() == "L4" else ""

#: GPU request for the tests: ``MODAL_GPU`` (default ``L4``) first, then its fallbacks.
TEST_GPU = _gpu_request(_PRIMARY_GPU, os.environ.get("MODAL_GPU_FALLBACKS", _DEFAULT_FALLBACKS))

#: GPU request for benchmarks: one type, so timings from different runs stay comparable.
BENCHMARK_GPU = _gpu_request(_PRIMARY_GPU)

#: Python version of the container image.
PYTHON_VERSION = os.environ.get("MODAL_PYTHON_VERSION", "3.12")

#: Lowest compute capability the Triton backend supports.
MINIMUM_COMPUTE_CAPABILITY = (8, 0)

#: Lowest PyTorch version the Triton backend supports.
MINIMUM_TORCH_VERSION = (2, 4)

#: Where the checkout is copied inside the container.
REMOTE_PROJECT_DIR = "/root/project"

#: Captured pytest log, relative to the project directory on both sides.
OUTPUT_LOG_PATH = Path("test-outputs") / "pytest-output.log"

#: Hard safety limit; the suite runs in minutes, so this only catches a hang.
RUN_TIMEOUT_SECONDS = 3600

#: Captured benchmark records, relative to the project directory on both sides.
BENCHMARK_OUTPUT_DIR = Path("test-outputs")

#: Benchmark presets the lane may run; each records validation-off solver time beside the public call.
BENCHMARK_PRESETS = ("rfdetr", "wide")

_SEPARATOR = "=" * 80

#: Run outside the checkout, so only the installed package can satisfy the import.
_CHECKOUT_IMPORT_CHECK = (
    "from pathlib import Path; import rfdetr.models.matcher, rfdetr.models._assignment; "
    "import torch_linear_assignment as package; "
    f"origin = Path(package.__file__).resolve(); checkout = Path({REMOTE_PROJECT_DIR!r}).resolve(); "
    "expected = checkout / 'src/torch_linear_assignment/__init__.py'; "
    "assert origin == expected, f'torch_linear_assignment is not imported from the checkout: {origin}'; "
    "print(f'torch_linear_assignment imported from {origin}')"
)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git")
    .pip_install("uv")
    # The dependency layer comes first and sees only pyproject.toml, so Modal reuses it until a
    # dependency pin changes; a source change only rebuilds the small layers after it. copy=True
    # is required because these files are used during the build.
    .add_local_file("pyproject.toml", remote_path=f"{REMOTE_PROJECT_DIR}/pyproject.toml", copy=True)
    .workdir(REMOTE_PROJECT_DIR)
    # The pinned develop group installs the CUDA PyTorch wheel and Triton, so
    # the container needs no CUDA base image of its own. The rfdetr group adds
    # the RF-DETR integration tests.
    .run_commands("uv pip install --system --group develop --group rfdetr")
    .add_local_dir(
        ".",
        remote_path=REMOTE_PROJECT_DIR,
        copy=True,
        ignore=[
            ".git",
            ".pytest_cache",
            ".venv",
            "__pycache__",
            "*.pyc",
            "*.egg-info",
            "*.log",
            "build",
            "dist",
            "test-outputs",
            ".modal",
        ],
    )
    # Installing the checkout last replaces any released torch-hungarian that an rfdetr dependency
    # pulled in. The check fails the image build unless the package is imported from the checkout,
    # so no rfdetr dependency can turn those tests into a no-op.
    .run_commands(
        "uv pip install --system -e .",
        f"cd /tmp && python -c {shlex.quote(_CHECKOUT_IMPORT_CHECK)}",
    )
)

app = modal.App(APP_NAME, image=image)


class RunResult(TypedDict, total=False):
    """Payload returned across the Modal boundary, so it stays a plain dict."""

    returncode: int
    success: bool
    gate_failure: str
    pytest_output: str
    summary: str


def _gpu_gate_failure(min_devices: int) -> str | None:
    """Report why the GPU lane cannot honestly run, or ``None`` when it can.

    Args:
        min_devices: How many eligible CUDA devices the run requires.

    Returns:
        A human-readable reason, or ``None`` when the host is eligible.
    """
    import importlib.util
    import re

    import torch

    if not torch.cuda.is_available():
        return "no CUDA device is available"
    if torch.version.cuda is None:
        return "the installed PyTorch build has no CUDA support"
    version = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if version is None or tuple(map(int, version.groups())) < MINIMUM_TORCH_VERSION:
        return f"PyTorch {torch.__version__} is older than the required 2.4"
    if importlib.util.find_spec("triton") is None:
        return "the triton package is not installed"
    eligible = [
        index
        for index in range(torch.cuda.device_count())
        if torch.cuda.get_device_capability(index) >= MINIMUM_COMPUTE_CAPABILITY
    ]
    if len(eligible) < min_devices:
        return (
            f"{min_devices} device(s) with compute capability >= 8.0 are required; "
            f"{len(eligible)} of {torch.cuda.device_count()} visible device(s) qualify"
        )
    return None


def _describe_environment() -> None:
    """Print the versions and devices that the run is about to be judged on."""
    import importlib.util
    from importlib import metadata

    import torch

    triton = metadata.version("triton") if importlib.util.find_spec("triton") else "unavailable"
    lines = [
        _SEPARATOR,
        "GPU ENVIRONMENT",
        _SEPARATOR,
        f"Torch: {torch.__version__} | Torch CUDA: {torch.version.cuda} | Triton: {triton}",
        f"CUDA available: {torch.cuda.is_available()} | device count: {torch.cuda.device_count()}",
    ]
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        lines.append(
            f"GPU {index}: {properties.name} | sm_{capability[0]}{capability[1]} | "
            f"{properties.total_memory / 1e9:.2f} GB | {properties.multi_processor_count} SMs"
        )
    lines.append(_SEPARATOR)
    print("\n".join(lines))


def _run_pytest(test_path: str, pytest_args: str) -> tuple[int, str]:
    """Run pytest, streaming its output while collecting it for the caller.

    Args:
        test_path: Path handed to pytest.
        pytest_args: Extra pytest arguments, parsed with shell quoting rules.

    Returns:
        The pytest exit code and its combined output.
    """
    command = ["pytest", test_path, *shlex.split(pytest_args), "--color=no"]
    print(f"{_SEPARATOR}\nRUNNING: {' '.join(command)}\n{_SEPARATOR}")

    output_file = Path(REMOTE_PROJECT_DIR) / OUTPUT_LOG_PATH
    output_file.parent.mkdir(parents=True, exist_ok=True)

    collected: list[str] = []
    with open(output_file, "w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REMOTE_PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("failed to capture the pytest output stream")
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            collected.append(line)
        process.wait()

    return process.returncode, "".join(collected)


@app.function(gpu=TEST_GPU, timeout=RUN_TIMEOUT_SECONDS)
def run_tests(test_path: str = "tests/", pytest_args: str = "-v", min_devices: int = 1) -> RunResult:
    """Run the test suite on a Modal GPU, refusing an ineligible host.

    Args:
        test_path: Path handed to pytest.
        pytest_args: Extra pytest arguments.
        min_devices: How many eligible CUDA devices the run requires.

    Returns:
        The exit code, the success flag, the captured output, and the gate
        failure reason when the host was rejected.

    Examples:
        >>> # Invoked remotely by the local entrypoint:
        >>> # modal run .modal/test_runner.py --test-path tests/
    """
    os.chdir(REMOTE_PROJECT_DIR)
    _describe_environment()

    gate_failure = _gpu_gate_failure(min_devices)
    if gate_failure is not None:
        print(f"{_SEPARATOR}\nGPU GATE FAILED: {gate_failure}\n{_SEPARATOR}")
        return {"returncode": 1, "success": False, "gate_failure": gate_failure, "pytest_output": ""}

    returncode, output = _run_pytest(test_path, pytest_args)
    print(f"{_SEPARATOR}\nEXIT CODE: {returncode}\n{_SEPARATOR}")
    return {"returncode": returncode, "success": returncode == 0, "pytest_output": output}


@app.function(gpu=BENCHMARK_GPU, timeout=RUN_TIMEOUT_SECONDS)
def run_benchmark(preset: str, benchmark_args: str = "", git_revision: str = "") -> RunResult:
    """Run one benchmark preset on a Modal GPU and return its JSONL records.

    The benchmark compares SciPy, the public call and the private Triton backend
    (all four validation modes, which the validation-overhead gate requires) in
    fresh processes, and fails closed on any
    parity mismatch or incomplete case, so a zero return code is the evidence.

    Args:
        preset: One of :data:`BENCHMARK_PRESETS`.
        benchmark_args: Extra benchmark arguments, parsed with shell quoting rules.
        git_revision: Revision of the local checkout, recorded as provenance
            because the image carries no ``.git`` directory.

    Returns:
        The exit code, the success flag, the JSONL text as ``pytest_output``, the
        printed results table as ``summary``, and the gate failure reason when the
        host was rejected.

    Examples:
        >>> # modal run .modal/test_runner.py --benchmark-preset rfdetr
    """
    os.chdir(REMOTE_PROJECT_DIR)
    _describe_environment()

    gate_failure = _gpu_gate_failure(1)
    if gate_failure is not None:
        print(f"{_SEPARATOR}\nGPU GATE FAILED: {gate_failure}\n{_SEPARATOR}")
        record = json.dumps({"status": "gpu_ineligible", "details": [gate_failure]}) + "\n"
        return {"returncode": 1, "success": False, "gate_failure": gate_failure, "pytest_output": record}

    output = Path(REMOTE_PROJECT_DIR) / BENCHMARK_OUTPUT_DIR / f"benchmark-{preset}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    extra_args = shlex.split(benchmark_args)
    allowed = {"--process-rounds", "--warmup", "--repetitions", "--worker-timeout-seconds"}
    if any(arg.split("=", 1)[0] not in allowed for arg in extra_args if arg.startswith("--")):
        raise ValueError("benchmark arguments may only tune rounds, warmup, repetitions, and timeout")
    output.unlink(missing_ok=True)
    command = [
        sys.executable,
        "tests/benchmark.py",
        "--preset",
        preset,
        "--backends",
        "scipy,public_cuda,triton",
        "--dtypes",
        "float32",
        "--expect-implementation",
        "triton",
        "--output",
        str(output),
        *extra_args,
    ]
    print(f"{_SEPARATOR}\nRUNNING: {' '.join(command)}\n{_SEPARATOR}")
    environment = {**os.environ, "TLA_BENCHMARK_GIT_REVISION": git_revision} if git_revision else None
    # Capture only stdout: it carries the results table, while the progress bar stays on stderr.
    process = subprocess.run(
        command, cwd=REMOTE_PROJECT_DIR, env=environment, check=False, stdout=subprocess.PIPE, text=True
    )
    print(process.stdout, end="")
    records = output.read_text() if output.exists() else ""
    print(f"{_SEPARATOR}\nEXIT CODE: {process.returncode}\n{_SEPARATOR}")
    return {
        "returncode": process.returncode,
        # Fail closed: without --allow-incomplete a zero exit always wrote records, so an empty
        # file means the run left no evidence, and a clean skip cannot pass for a measured run.
        "success": process.returncode == 0 and bool(records.strip()),
        "pytest_output": records,
        "summary": process.stdout,
    }


def _dispatch_benchmark(preset: str, benchmark_args: str) -> None:
    """Run one benchmark preset remotely and save its JSONL and table next to the local checkout.

    Args:
        preset: One of :data:`BENCHMARK_PRESETS`.
        benchmark_args: Extra benchmark arguments, such as ``--process-rounds 5``.
    """
    if preset not in BENCHMARK_PRESETS:
        sys.exit(f"unknown preset {preset!r}; choose from {', '.join(BENCHMARK_PRESETS)}")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    print(f"{_SEPARATOR}\nGPU BENCHMARK\n{_SEPARATOR}\nGPU: {BENCHMARK_GPU} | preset: {preset} | revision: {revision}")

    result = run_benchmark.remote(preset=preset, benchmark_args=benchmark_args, git_revision=revision)

    output = BENCHMARK_OUTPUT_DIR / f"benchmark-{preset}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.get("pytest_output") or "")
    summary = output.with_suffix(".txt")
    if result.get("gate_failure"):
        summary.write_text(f"GPU gate failed: {result['gate_failure']}\n")
    else:
        summary.write_text(result.get("summary") or "The benchmark printed no results table; see the run log.\n")
    print(f"Saved benchmark records to: {output} and the table to: {summary}")
    if not result["success"]:
        sys.exit(result.get("gate_failure") or result["returncode"] or "Benchmark produced no records")
    print("Benchmark passed its parity and completeness gates.")


@app.local_entrypoint()
def main(
    test_path: str = "tests/",
    pytest_args: str = "-v",
    min_devices: int = 1,
    benchmark_preset: str = "",
    benchmark_args: str = "",
) -> None:
    """Dispatch the GPU run and mirror its log next to the local checkout.

    Args:
        test_path: Path handed to pytest.
        pytest_args: Extra pytest arguments.
        min_devices: How many eligible CUDA devices the run requires.
        benchmark_preset: Run this benchmark preset instead of pytest; one of
            :data:`BENCHMARK_PRESETS`.
        benchmark_args: Extra benchmark arguments, used with ``benchmark_preset``.

    Examples:
        >>> # modal run .modal/test_runner.py --pytest-args "-v -k compiled"
        >>> # modal run .modal/test_runner.py --benchmark-preset rfdetr
    """
    if benchmark_preset:
        _dispatch_benchmark(benchmark_preset, benchmark_args)
        return
    print(
        f"{_SEPARATOR}\nGPU TEST RUNNER\n{_SEPARATOR}\n"
        f"GPU: {TEST_GPU} | Python: {PYTHON_VERSION} | minimum eligible devices: {min_devices}\n"
        f"Test path: {test_path} | pytest args: {pytest_args}\n{_SEPARATOR}"
    )

    result = run_tests.remote(test_path=test_path, pytest_args=pytest_args, min_devices=min_devices)

    OUTPUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_text = result.get("pytest_output") or f"GPU gate failed: {result.get('gate_failure', 'unknown reason')}\n"
    OUTPUT_LOG_PATH.write_text(log_text)
    print(f"Saved pytest output to: {OUTPUT_LOG_PATH}")

    if not result["success"]:
        sys.exit(result["returncode"])
    print("All GPU tests passed.")
