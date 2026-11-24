# Modal GPU test lane

Hosted GitHub runners have no GPU, so the compiled Triton tests never execute in `ci_cpu-tests.yml`. This directory runs them on real NVIDIA hardware through [Modal](https://modal.com)'s serverless GPUs, following the pattern published in [Borda/affordable-GPU-CI](https://github.com/Borda/affordable-GPU-CI).

## Why it is fail-closed

Every test in `tests/unittests/test_triton.py` and `tests/integrations/test_rfdetr_gpu.py` skips itself when the host cannot honestly run a compiled kernel, and the CUDA tests in `tests/unittests/test_assignment.py` do the same. A GPU run that quietly skipped everything would report success while proving nothing, so `test_runner.py` checks eligibility before pytest starts and exits nonzero when the host does not qualify:

- an available CUDA device on a CUDA-enabled PyTorch build,
- PyTorch 2.4 or newer,
- the `triton` package,
- at least `--min-devices` devices with compute capability 8.0 or newer. CI sets it to the leg's device count.

T4 (`sm_75`) therefore fails the gate rather than passing with a fully skipped suite. Use L4, A10, L40S, A100, or H100.

## Local use

```bash
python -m pip install --group modal
export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...

modal run .modal/test_runner.py
modal run .modal/test_runner.py --test-path tests/unittests/test_triton.py --pytest-args "-v -k compiled"
modal run .modal/test_runner.py --test-path tests/integrations/
MODAL_GPU=A100 modal run .modal/test_runner.py
MODAL_GPU=L4:2 modal run .modal/test_runner.py --min-devices 2
MODAL_GPU_FALLBACKS= modal run .modal/test_runner.py   # L4 only, no fallback
modal run .modal/test_runner.py --benchmark-preset rfdetr --benchmark-args "--process-rounds 1"
```

The container installs the project with the pinned `develop` and `rfdetr` dependency groups, so the GPU environment matches the hosted CPU snapshot except for the CUDA PyTorch wheel, and the image build fails unless the package is imported from the checkout. The default `tests/` path runs both `tests/unittests` and `tests/integrations`. The pytest log is mirrored to `test-outputs/pytest-output.log` next to the checkout; a `--benchmark-preset` run saves its records to `test-outputs/benchmark-<preset>.jsonl` and its printed table to `test-outputs/benchmark-<preset>.txt` instead.

## CI use

| Workflow                                        | Trigger                                                                                                            |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `.github/workflows/_modal-gpu-tests.yml`        | reusable; called by the two test workflows below                                                                   |
| `.github/workflows/_modal-gpu-benchmarks.yml`   | reusable; runs the presets and reports them, called by the two workflows below                                     |
| `.github/workflows/dispatch_gpu-tests.yml`      | push to `main` and `develop`, plus manual dispatch with a GPU type and a list of device counts                     |
| `.github/workflows/trigger-gpu-labels.yml`      | the `run-gpu-tests` label runs the tests and the `run-gpu-benchmarks` label runs the benchmarks, on a pull request |
| `.github/workflows/dispatch_gpu-benchmarks.yml` | manual dispatch with a GPU, preset and arguments                                                                   |

Required repository secrets: `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. The label lanes also need the `run-gpu-tests` and `run-gpu-benchmarks` labels to exist; each is removed again once its run reports back.

The test workflow runs one leg per device count. Leave `nb_devices` blank to run the whole suite on one GPU and the two marked isolation tests on two GPUs. Revisions that predate the `multi_device` marker default to one GPU. An explicit list must contain distinct positive integers, and counts above one require the marker at the selected revision. Set `nb_devices` to `[1]` to run only the full suite. The default two-GPU request is `L4:2`, with `A10:2` and `L40S:2` as fallbacks. A custom accelerator uses that type with the selected device count.

The `run-gpu-benchmarks` label and manual preset `all` both run `rfdetr` and `wide` in parallel. A named preset runs by itself. The lane fails if any case misses exact parity, any case is incomplete, or the public call uses a backend other than Triton. The workflow posts medians across process rounds and any unmeasured cases in the run summary and, for a label run, as a pull request comment. It uploads JSONL records as `gpu-benchmark-<preset>` artifacts. The private Triton lane runs all four validation modes to measure validation overhead. `rfdetr` and `wide` default to three fresh-process rounds. Pass `--process-rounds 1` on dispatch for a shorter run.

For a kernel speedup claim, run the same workloads against both the baseline and candidate installed from their exact revisions on the same GPU and software stack. Use `make benchmark-rfdetr BENCHMARK_ARGS="--output baseline-rfdetr.jsonl"` for the matcher. For batch-208 large shapes, run `make benchmark BENCHMARK_ARGS="--batches 208 --dtypes float32 --backends scipy,public_cuda,triton --workers 300 --tasks 300,600 --output baseline-large-300.jsonl"` and, separately, `make benchmark BENCHMARK_ARGS="--batches 208 --dtypes float32 --backends scipy,public_cuda,triton --workers 512 --tasks 512 --output baseline-large-512.jsonl"`. Use different output paths for the candidate revision. Match the seed, warmup, repetitions and process rounds, and retain the JSONL with exact parity and input digests. Compare warm medians for the same backend, validation mode and shape across revisions. The displayed speedup column uses the current run's SciPy CPU time. Set a matcher regression tolerance before comparing the samples.

In GitHub Actions, run `readme` by choosing it explicitly on `dispatch_gpu-benchmarks.yml` and setting `gpu` to the desired model, such as `H100`. You can also run it directly on Modal with `MODAL_GPU=H100 modal run .modal/test_runner.py --benchmark-preset readme`. Both paths and `make benchmark-readme` use the three batch-208 shapes (`300 x 100`, `300 x 300`, `300 x 600`) with five fresh-process rounds, three warmup calls and thirty repetitions by default.

Publish these results separately from the historical README table. Use **Fresh-process cold median** for the median first call across isolated workers with empty Triton caches and **Warm median** for the warm result. The historical `first, shared` rows (earlier marked `first*`) describe first calls within a shared benchmark process, while a fresh-process cold median is the README's `first, fresh` row. For public API timings, use the `public_cuda/triton` lane with full validation. Cite the tested revision and source-bound JSONL records, including the GPU and SciPy host CPU model. Compare SciPy times across runs only when the host CPUs match. The historical table lacks CPU model metadata.

Both label lanes trigger on `pull_request` and never on `pull_request_target`. GitHub withholds secrets from fork pull requests, so fork code cannot reach the Modal credentials even if the label is applied by mistake.

## Cost

The test lane asks for L4 and accepts A10 or L40S when L4 is unavailable. Modal's pricing page lists L4, A10 and L40S at $0.000222, $0.000306 and $0.000542 per second, about $0.80, $1.10 and $1.95 per hour. The benchmark lane defaults to L4 and requests only the selected GPU type, with no fallback, so each run measures the intended device. Modal's free tier covers $30 of compute per month. This suite runs in a few minutes, so an L4 run costs roughly $0.01 and the free tier absorbs far more runs than this project produces. Cold runs also pay a one-time image build. The image installs the dependency groups from `pyproject.toml` alone in its own layer, ahead of the source copy, so a source-only change reuses that layer and rebuilds only the checkout install, and a change to the dependency pins rebuilds all of it.
