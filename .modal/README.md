# Modal GPU test lane

Hosted GitHub runners have no GPU, so the compiled Triton tests never execute in `cpu-tests.yml`. This directory runs them on real NVIDIA hardware through [Modal](https://modal.com)'s serverless GPUs, following the pattern published in [Borda/affordable-GPU-CI](https://github.com/Borda/affordable-GPU-CI).

## Why it is fail-closed

Every test in `tests/test_triton.py` skips itself when the host cannot honestly run a compiled kernel, and the CUDA tests in `tests/test_assignment.py` do the same. A GPU run that quietly skipped everything would report success while proving nothing, so `test_runner.py` checks eligibility before pytest starts and exits nonzero when the host does not qualify:

- an available CUDA device on a CUDA-enabled PyTorch build,
- PyTorch 2.4 or newer,
- the `triton` package,
- at least `--min-devices` devices with compute capability 8.0 or newer.

T4 (`sm_75`) therefore fails the gate rather than passing with a fully skipped suite. Use L4, A10G, A100, or H100.

## Local use

```bash
python -m pip install "modal==1.5.5"
export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...

modal run .modal/test_runner.py
modal run .modal/test_runner.py --test-path tests/test_triton.py --pytest-args "-v -k compiled"
MODAL_GPU=A100 modal run .modal/test_runner.py
MODAL_GPU=L4:2 modal run .modal/test_runner.py --min-devices 2
```

The container installs the project with the pinned `develop` dependency group, so the GPU environment matches the hosted CPU snapshot except for the CUDA PyTorch wheel. The pytest log is mirrored to `test-outputs/pytest-output.log` next to the checkout.

## CI use

| Workflow                                 | Trigger                                                          |
| ---------------------------------------- | ---------------------------------------------------------------- |
| `.github/workflows/_modal-gpu-tests.yml` | reusable; called by the two below                                |
| `.github/workflows/gpu-tests.yml`        | push to `main`, plus manual dispatch with a GPU and device count |
| `.github/workflows/gpu-tests-label.yml`  | the `gpu-tests` label on a pull request                          |

Required repository secrets: `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. The label lane also needs a `gpu-tests` label to exist; it is removed again once the run reports back.

The label lane triggers on `pull_request` and never on `pull_request_target`. GitHub withholds secrets from fork pull requests, so fork code cannot reach the Modal credentials even if the label is applied by mistake.

## Cost

Modal's free tier covers $30 of compute per month. This suite runs in a few minutes, so an L4 run costs roughly $0.01 and the free tier absorbs far more runs than this project produces. Cold runs also pay a one-time image build; later runs reuse the cached layers.
