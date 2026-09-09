"""Run the GPU test lane on Modal's serverless NVIDIA hardware.

Ported from https://github.com/Borda/affordable-GPU-CI and adapted to this
project: the reproducible ``develop`` dependency group replaces the upstream
``tests`` group, the container starts from a slim Debian image instead of the
NGC PyTorch image because the pinned PyPI PyTorch wheel already carries CUDA,
and the run is fail-closed on GPU eligibility.

The fail-closed gate matters more here than anywhere else in the suite: every
compiled-backend test in ``tests/test_triton.py`` skips itself when the host is
ineligible, so a run on an unsupported GPU would report success while proving
nothing about the Triton kernels. The gate refuses the run instead.

Usage:
    modal run .modal/test_runner.py
    modal run .modal/test_runner.py --test-path tests/test_triton.py
    modal run .modal/test_runner.py --pytest-args "-v -k compiled"
    MODAL_GPU=L4:2 modal run .modal/test_runner.py --min-devices 2

Environment Variables:
    MODAL_GPU: GPU request passed to Modal, such as ``L4``, ``A100``, or
        ``L4:2`` for two devices. Compute capability 8.0 or newer is required,
        so ``T4`` (``sm_75``) is rejected by the gate rather than silently
        skipped.
    MODAL_PYTHON_VERSION: Python version of the container image.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

import modal

#: Modal application name; stable so runs group together in the Modal dashboard.
APP_NAME = "torch-hungarian-gpu-tests"

#: GPU request forwarded to Modal. ``L4`` is the cheapest ``sm_89`` device.
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")

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

_SEPARATOR = "=" * 80

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git")
    .pip_install("uv")
    # copy=True is required because dependencies are installed during the build.
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
    .workdir(REMOTE_PROJECT_DIR)
    # The pinned develop group installs the CUDA PyTorch wheel and Triton, so
    # the container needs no CUDA base image of its own.
    .run_commands("uv pip install --system -e . --group develop")
)

app = modal.App(APP_NAME, image=image)


class RunResult(TypedDict, total=False):
    """Payload returned across the Modal boundary, so it stays a plain dict."""

    returncode: int
    success: bool
    gate_failure: str
    pytest_output: str


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


@app.function(gpu=GPU_TYPE, timeout=RUN_TIMEOUT_SECONDS)
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


@app.local_entrypoint()
def main(test_path: str = "tests/", pytest_args: str = "-v", min_devices: int = 1) -> None:
    """Dispatch the GPU run and mirror its log next to the local checkout.

    Args:
        test_path: Path handed to pytest.
        pytest_args: Extra pytest arguments.
        min_devices: How many eligible CUDA devices the run requires.

    Examples:
        >>> # modal run .modal/test_runner.py --pytest-args "-v -k compiled"
    """
    print(
        f"{_SEPARATOR}\nGPU TEST RUNNER\n{_SEPARATOR}\n"
        f"GPU: {GPU_TYPE} | Python: {PYTHON_VERSION} | minimum eligible devices: {min_devices}\n"
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
