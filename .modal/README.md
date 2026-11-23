# Modal GPU test lane

Hosted GitHub runners have no GPU, so the compiled Triton tests never execute in `cpu-tests.yml`. This directory runs them on real NVIDIA hardware through [Modal](https://modal.com)'s serverless GPUs, following the pattern published in [Borda/affordable-GPU-CI](https://github.com/Borda/affordable-GPU-CI).

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

| Workflow                                 | Trigger                                                            |
| ---------------------------------------- | ------------------------------------------------------------------ |
| `.github/workflows/_modal-gpu-tests.yml` | reusable; called by the two test workflows below                   |
| `.github/workflows/gpu-tests.yml`        | push to `main`, plus manual dispatch with a GPU and device count   |
| `.github/workflows/gpu-tests-label.yml`  | the `gpu-tests` label on a pull request                            |
| `.github/workflows/gpu-benchmarks.yml`   | manual dispatch, plus the `gpu-benchmarks` label on a pull request |

Required repository secrets: `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. The label lanes also need the `gpu-tests` and `gpu-benchmarks` labels to exist; each is removed again once its run reports back.

The benchmark lane runs every preset in parallel, or the one chosen on dispatch, and fails when any case misses exact parity, any case is incomplete, or the public call did not execute the Triton backend. It reports the results table, one row of medians across process rounds per case, and, as limitations, every record that was not measured to the run summary and, for the label, as a pull request comment; the JSONL records are uploaded as `gpu-benchmark-<preset>` artifacts. Every Triton shape runs all four validation modes, because the run fails closed without a complete validation-overhead measurement. The default three process rounds run each case in three fresh processes, and every Triton case pays a cold compile in each; pass `--process-rounds 1` on dispatch for a quicker look.

For a kernel speedup claim, run the same workloads against both the baseline and candidate installed from their exact revisions on the same GPU and software stack. Use `make benchmark-rfdetr` for the matcher. For batch-208 large shapes, run `make benchmark` with `--batches 208 --dtypes float32 --backends scipy,public_cuda,triton` and use `--workers 300 --tasks 300,600` and `--workers 512 --tasks 512` in separate runs. Pass a distinct `--output` path for each revision and workload. Match the seed, warmup, repetitions and process rounds, and retain the JSONL with exact parity and input digests. Compare warm medians for the same backend, validation mode and shape across revisions. The displayed speedup column uses the current run's SciPy CPU time. Set a matcher regression tolerance before comparing the samples.

Both label lanes trigger on `pull_request` and never on `pull_request_target`. GitHub withholds secrets from fork pull requests, so fork code cannot reach the Modal credentials even if the label is applied by mistake.

## Cost

Modal's free tier covers $30 of compute per month. This suite runs in a few minutes, so an L4 run costs roughly $0.01 and the free tier absorbs far more runs than this project produces. Cold runs also pay a one-time image build; later runs reuse the cached layers.
