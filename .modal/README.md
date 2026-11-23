# Modal GPU test lane

Hosted GitHub runners have no GPU, so the compiled Triton tests never execute in `ci_cpu-tests.yml`. This directory runs them on real NVIDIA hardware through [Modal](https://modal.com)'s serverless GPUs, following the pattern published in [Borda/affordable-GPU-CI](https://github.com/Borda/affordable-GPU-CI).

## Why it is fail-closed

Every test in `tests/unittests/test_triton.py` and `tests/integrations/test_rfdetr_gpu.py` skips itself when the host cannot honestly run a compiled kernel, and the CUDA tests in `tests/unittests/test_assignment.py` do the same. A GPU run that quietly skipped everything would report success while proving nothing, so `test_runner.py` checks eligibility before pytest starts and exits nonzero when the host does not qualify:

- an available CUDA device on a CUDA-enabled PyTorch build,
- PyTorch 2.4 or newer,
- the `triton` package,
- at least `--min-devices` devices with compute capability 8.0 or newer.

T4 (`sm_75`) therefore fails the gate rather than passing with a fully skipped suite. Use L4, A10G, A100, or H100.

## Local use

```bash
python -m pip install --group modal
export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...

modal run .modal/test_runner.py
modal run .modal/test_runner.py --test-path tests/unittests/test_triton.py --pytest-args "-v -k compiled"
modal run .modal/test_runner.py --test-path tests/integrations/
MODAL_GPU=A100 modal run .modal/test_runner.py
MODAL_GPU=L4:2 modal run .modal/test_runner.py --min-devices 2
modal run .modal/test_runner.py --benchmark-preset rfdetr --benchmark-args "--process-rounds 1"
```

The container installs the project with the pinned `develop` and `rfdetr` dependency groups, so the GPU environment matches the hosted CPU snapshot except for the CUDA PyTorch wheel, and the image build fails unless the package is imported from the checkout. The default `tests/` path runs both `tests/unittests` and `tests/integrations`. The pytest log is mirrored to `test-outputs/pytest-output.log` next to the checkout; a `--benchmark-preset` run saves its records to `test-outputs/benchmark-<preset>.jsonl` and its printed table to `test-outputs/benchmark-<preset>.txt` instead.

## CI use

| Workflow                                           | Trigger                                                                                                    |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `.github/workflows/_modal-gpu-tests.yml`           | reusable; called by the two test workflows below                                                           |
| `.github/workflows/_modal-gpu-benchmarks.yml`      | reusable; runs the presets and reports them, called by the two workflows below                             |
| `../.github/workflows/dispatch_gpu-tests.yml`      | push to `main`, plus manual dispatch with a GPU and device count                                           |
| `.github/workflows/trigger-gpu-labels.yml`         | the `gpu-tests` label runs the tests and the `gpu-benchmarks` label runs the benchmarks, on a pull request |
| `../.github/workflows/dispatch_gpu-benchmarks.yml` | manual dispatch with a GPU, preset and arguments                                                           |

Required repository secrets: `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. The label lanes also need the `gpu-tests` and `gpu-benchmarks` labels to exist; each is removed again once its run reports back.

The benchmark lane runs every preset in parallel, or the one chosen on dispatch, and fails when any case misses exact parity, any case is incomplete, or the public call did not execute the Triton backend. It reports the results table, one row of medians across process rounds per case, and, as limitations, every record that was not measured to the run summary and, for the label, as a pull request comment; the JSONL records are uploaded as `gpu-benchmark-<preset>` artifacts. Every Triton shape runs all four validation modes, because the run fails closed without a complete validation-overhead measurement. The default three process rounds run each of the 126 preset cases in three fresh processes, and every Triton case pays a cold compile in each; pass `--process-rounds 1` on dispatch for a quicker look.

Both label lanes trigger on `pull_request` and never on `pull_request_target`. GitHub withholds secrets from fork pull requests, so fork code cannot reach the Modal credentials even if the label is applied by mistake.

## Cost

Modal's free tier covers $30 of compute per month. This suite runs in a few minutes, so an L4 run costs roughly $0.01 and the free tier absorbs far more runs than this project produces. Cold runs also pay a one-time image build; later runs reuse the cached layers.
