# Changelog

## Unreleased

### Added

- Split the suite into `tests/unittests/` for the package's own tests and `tests/integrations/` for tests against downstream projects. `make test` and the CPU matrix run the unit tests; the Modal lane runs both. The compiled-kernel host checks move to `tests/_triton_eligibility.py`, shared by both suites.
- Test the RF-DETR integration against RF-DETR's own SciPy matcher. `tests/integrations/test_rfdetr.py` compares RF-DETR 1.12.1's batched solve with its per-problem SciPy oracle at training shapes: 300 queries in 13 groups, 0 to 30 targets per image, several decoder layers, more targets than queries per group, and mixed query counts. Continuous costs must return the oracle's exact index pairs, and tied costs its objective. `tests/integrations/test_rfdetr_gpu.py` repeats the comparison on the Triton kernel and checks RF-DETR's CUDA matcher, single- and multi-layer, against the host matcher; each test asserts that the kernel ran, so an unsupported GPU cannot turn it into SciPy against SciPy.
- Add an `rfdetr` dependency group and `make install-rfdetr`, which installs the checkout after RF-DETR with `--no-deps` and fails unless the package is imported from the checkout. RF-DETR's `train` extra pins an older `torch-hungarian` and is not used. A separate CPU workflow job runs the host tests, and the Modal GPU image installs the group and checks the import origin at build time.
- Add `--preset rfdetr` and `--preset wide` to the benchmark, with `make benchmark-rfdetr`, `make benchmark-wide`, and `modal run .modal/test_runner.py --benchmark-preset <name>`. A preset is a list of explicit `(batch, workers, tasks)` shapes, so one run can mix worker counts. Every Triton row also reports validation-off solver time from CUDA events, including workspace initialization, with dtype promotion, transpose, validation, and result conversion kept outside the timed window.
- Run the GPU benchmark presets on Modal through manual dispatch via `dispatch_gpu-benchmarks.yml` or the maintainer-applied `gpu-benchmarks` pull request label via `trigger-gpu-labels.yml`. It writes the results table and every unmeasured record to the run summary, comments them on the pull request for the label, uploads the JSONL records, and removes the label. The Modal runner now also saves the printed table as `test-outputs/benchmark-<preset>.txt`.
- Skip benchmark runs that measure the Triton backend, or expect the public call to use it, on a GPU below compute capability 8.0, such as a T4. The run writes one `gpu_ineligible` record naming the device and its capability before any case starts, prints it under a new `Limitations` list, and exits nonzero unless `--allow-incomplete` is given, like the existing no-GPU skip. A `public_cuda`-only run still measures a legacy build or the SciPy fallback on such a GPU.
- Add `batch_linear_assignment_indices(cost, *, validate="full")`, which returns SciPy's `(row_ind, col_ind)` as `torch.long` tensors of shape `(B, min(W, T))` on the input's device. On Triton, pair conversion adds no host reads. Full validation reads back to check floating-point inputs before launch and to check for infeasibility after a solve. For calls routed to Triton, `validate="off"` skips those checks and allows CUDA graph capture after a side-stream warm-up. Invalid or infeasible costs then return unspecified indices. SciPy-backed paths always validate. The README includes a CUDA graph example.

### Changed

- Solve small CUDA batches with SciPy on the host, without the fallback warning: one problem, two problems with `max(W, T) <= 64`, or up to four problems with `max(W, T) <= 32`. The audit measured the host solve faster there even with the transfers, and the result still returns on the input's device. The default benchmark batches drop 1 for the same reason, since the public call would not measure the GPU; pass `--batches 1` to keep it.
- Print one benchmark table row per case instead of one per process round. Warm, kernel, and cold times are medians across rounds, warm p95 is the worst round's, parity is exact only when every round was, and the speedup is recomputed from the medians; skipped and failed records still print individually, and the JSONL keeps every round.
- Mark the pull request comments from the Modal runs with a green check or a red cross: the title and job result of the `gpu-tests` and `gpu-benchmarks` comments, and each benchmark preset heading, which takes its outcome from that preset's own matrix job. The benchmark report job gains `actions: read` for this, and its comment is bounded below GitHub's size limit, with the label removed even when posting fails. The report moves out of the workflow into `.github/scripts/gpu-benchmark-report.js`, which `.github/scripts/gpu-benchmark-report.test.js` covers with Node's built-in test runner in a new `cpu-tests-workflow-scripts` job of the CPU workflow.
- Name both the lane and the implementation in the benchmark table's backend column when they differ, so the public call reads `public_cuda/triton` and no longer prints the same label as the private `triton` lane.
- Name the tested commit in the `gpu-tests` and `gpu-benchmarks` pull request comments and in the benchmark run summary, with how many commits the pull request head is ahead of it, since a run can start well after the label and the branch keeps moving. A new `.github/scripts/tested-commit.js`, with its own tests, builds the line for both workflows; the `gpu-tests-report` job now checks out `.github/scripts` from the tested commit for it.
- Print up to 20 benchmark progress lines per nonterminal run, with a final line after the last completed case. A terminal still redraws a live bar.
- Pass matrix dimensions to the Triton kernel at runtime so shapes in a power-of-two column bucket can share a binary for a fixed GPU target, solver dtype, pointer alignment and warp count. An earlier L4 audit of the approach measured 100 to 1 binaries for 1–100 tasks at 300 workers and 40 to 3 binaries for 40 mixed shapes. The solver packs its workspaces into two backing allocations with each section aligned to a 16-byte boundary.
- Launch the Triton kernel with 4 warps when the solver's column block is at least 512 and it has at least 128 rows. Other launches, including RF-DETR matcher shapes, use one warp. The augmenting-path loop has barriers after its `col4row` read and after each step. An earlier L4 prototype audit measured 1.7–3.0× on 300×300, 512×512 and 300×600 with 4 to 8 warps and about 1.1× at the RF-DETR shape. The GPU suite checks both assignment maps against SciPy across more than 100,000 problems at each of 1, 2, 4 and 8 warps, with separate coverage for float64 and forbidden edges.
- Route the GPU label lanes through `trigger-gpu-labels.yml`, which runs only the lane named by the label: `gpu-tests` calls `_modal-gpu-tests.yml` and `gpu-benchmarks` calls the new `_modal-gpu-benchmarks.yml`. Each reusable workflow now runs and reports its own lane (the test report moved there from the label workflow), and every job name is unique across the workflows: `gpu-tests-run`, `gpu-tests-report`, `gpu-benchmarks-resolve`, `gpu-benchmarks-run`, `gpu-benchmarks-report`, and one caller job per trigger. The CPU test jobs carry the matching `cpu-` prefix (`cpu-tests`, `cpu-tests-rfdetr`, `cpu-tests-workflow-scripts`), so test and benchmark jobs say which hardware they run on. `dispatch_gpu-benchmarks.yml` is now only the manual dispatch trigger for the benchmark workflow. One label used to start both label workflows and skip one, so a pair of labels showed four runs, two of them skipped; now each label starts one run named after the label and the pull request, with the other lane's job skipped. A label that is neither lane still starts a run in which both lane jobs skip, because GitHub cannot filter the trigger by label name.
- Let the Modal test lane fall back to A10 or L40S when it asks for L4. `MODAL_GPU` stays `L4` by default, `MODAL_GPU_FALLBACKS` (default `A10,L40S` for an L4 request, none otherwise, empty to turn off) names the other types, and a device count such as `L4:2` is copied to each fallback. The benchmark lane never uses fallbacks, so its timings remain comparable. The image now installs the dependency groups from `pyproject.toml` in a layer before the source copy, so a source-only change reuses it: in five earlier runs the image build took 52 to 70 s on every commit, of which 24 to 35 s was downloading packages. An earlier run spent 26 minutes between the image build and the GPU container starting, so the wait for a GPU, not the build, is the larger cost.

### Fixed

- Reduce large common cost offsets in the Triton search when every finite row subtraction is exact. Matrices with unsafe shifts retain the original search arithmetic. An earlier L4 audit of row reduction measured 949 to zero cases with a worse float32 assignment than SciPy on its square-matrix workload.
- Solve a cost on any device other than the CPU or CUDA, such as MPS or XPU, on the host and return the assignment on the input's device. The CPU path used to call `numpy()` on the accelerator tensor and raise `TypeError`.
- Launch the Triton solve on the cost's own GPU when it is not the current one. The launch used to go to the current device, so a cost on `cuda:1` with `cuda:0` current could fail or run on the wrong GPU; the common case, where the device is already current, opens no device context.
- Compute the per-batch offsets in the Triton kernel in 64 bits. Batches of more than 2\*\*31 elements wrapped the 32-bit offsets and read and wrote the wrong addresses.
- Say which of its two conditions failed when the multi-layer RF-DETR GPU test asserts, and resolve the checkout path in the Modal import check the same way as the imported package path.

## 0.1.0 - 2026-11-21

### Added

- Add a hosted GPU test lane on Modal's serverless NVIDIA hardware, ported from [Borda/affordable-GPU-CI](https://github.com/Borda/affordable-GPU-CI). A reusable workflow runs `.modal/test_runner.py`, which installs the pinned `develop` dependency group into a CUDA container, with the Modal client coming from a separate `modal` dependency group, and uploads the pytest log. It is called on pushes to `main` and `develop`, through manual dispatch with a GPU and device count, and from the maintainer-applied `gpu-tests` pull request label, which also posts the result and removes itself. The label lane triggers on `pull_request` rather than `pull_request_target`, so fork code never receives the Modal credentials.
- Fail the Modal lane closed on GPU eligibility. Every compiled Triton test skips itself on an ineligible host, so the runner checks for a CUDA-enabled PyTorch 2.4 or newer build, the `triton` package, and the requested number of devices with compute capability 8.0 or newer before pytest starts, and exits nonzero otherwise. A fully skipped suite can no longer be reported as a passing GPU run.

### Changed

- Point the README citation section at the published Neurocomputing and AAAI Conference on Artificial Intelligence versions of the HoTPP and DeTPP papers instead of the arXiv preprints.

### Fixed

- Accept gradient-bearing CPU costs and unsupported-CUDA fallback costs without changing the input's autograd graph.
- Reject complex costs consistently before backend dispatch or private Triton conversion, including empty batches and private validation modes.
- Bind benchmark evidence to each worker's imported dispatch source and record the benchmark checkout's dirty state even when the revision is supplied by the copied runner.
- Bound benchmark workers to 900 seconds by default, configurable with `--worker-timeout-seconds`; record timeouts as ineligible `worker_timeout` results.

## 0.1.0rc1 - 2026-09-08

### Added

- Add a hosted packaging gate: one job builds the wheel and the source distribution and uploads them, and a second job downloads that artifact without a checkout, installs the wheel with pip, and imports the package to report its version.

### Changed

- Continue the project as the [Borda/torch-hungarian](https://github.com/Borda/torch-hungarian) fork, published on PyPI as `torch-hungarian`. Ivan Karpukhin remains the author of the original project and of the `0.0.x` compiled CUDA line; Jirka Borovec is the maintainer of this fork.
- Keep the import name `torch_linear_assignment`, so code written against the upstream project and against `torch-hungarian==0.1.0rc0` needs no change.
- Point the package metadata, README badges, install command, and Colab reproduction cells at the fork, and add source, changelog, and upstream project URLs.
- Resolve the installed version from `torch-hungarian` first and from `torch-linear-assignment` second in `tests/benchmark.py` and `make benchmark`, so the fork and the frozen `0.0.6` upstream baseline are both recognized.
- Uninstall `torch-hungarian` in `make install-legacy` before installing the `0.0.6` baseline. Both distributions provide `torch_linear_assignment`, so without this the target could report the fork's version while running the legacy backend.
- Raise the PyTorch floor from `1.12.0` to `2.0.0`. The Triton path already required `2.4` at runtime, and the SciPy fallback carries the older environments no further.
- Declare all packaging metadata in `pyproject.toml` through the standard `[project]` table and delete `setup.py`. The published runtime requirements are unchanged; `Home-page` becomes a `Homepage` project URL, and the author and maintainer fields take their canonical `Name <email>` form.
- Move the package sources under `src/`, so tests and tooling can no longer import the package from the working directory by accident.
- Export `torch_linear_assignment.__version__` and read the distribution version from it through `[tool.setuptools.dynamic]`, so the package holds the single version declaration.
- Fold `requirements.txt` into `[project.dependencies]` and `requirements-ci.txt` into a pinned `develop` dependency group, alongside an unpinned `test` group for local tooling. Dependency groups stay out of the published metadata, unlike extras.
- Install the hosted validation environment with `uv`, which resolves the pinned `develop` group and reports the dependency check. The Makefile keeps `pip`, because `uv` ignores user site-packages and would shadow a local user-site install.

### Removed

- Remove the legacy CUDA and C++ sources, the `TLA_BUILD_LEGACY_CUDA` opt-in build, and the in-process `legacy_cuda` benchmark backend. The `0.1.0+` public API never dispatched to that extension, nothing built or tested it, and the sources remain available upstream and in this repository's history.
- Legacy comparisons keep working through the documented cross-environment route: `make install-legacy` installs the exact PyPI `0.0.6` baseline, and `make benchmark` still identifies, pins, and accepts a run that executes `legacy_cuda`.

### Fixed

- Restrict the `triton` runtime dependency to Linux `x86_64`, matching the marker PyTorch itself uses. Triton publishes no aarch64 wheel and no source distribution, so the previous Linux-wide marker made installation fail outright on ARM Linux instead of falling back to SciPy.
- Pin `triton` in the hosted CI snapshot. The CPU-only PyTorch wheel omits the `triton` dependency that the CUDA wheel carries, so `pip check` failed on every matrix job.
- Upload the CPU-only benchmark diagnostic only when the benchmark step actually ran, so a failing install no longer reports a second, misleading missing-artifact error.

## 0.1.0rc0 - 2026-08-20

### Added

- Add a lazy Triton CUDA backend for Linux NVIDIA GPUs with compute capability 8.0 or newer and Torch 2.4 or newer.
- Add exact SciPy parity, dtype, autocast, non-finite input, non-default stream, and large-batch regression coverage for the Triton solver.
- Add reproducible author-run GPU validation and JSONL benchmark reporting with paired SciPy and legacy `0.0.6` comparisons.
- Add a fresh-process benchmark harness: every case runs in its own interpreter with a unique empty Triton cache, retains raw cold and warm samples, and enforces exact SciPy parity on each measured call.
- Add self-authenticating JSONL evidence that binds every record to the benchmark Git revision and source digest, the imported package origin and digest, the executed backend module digest, the input digest, the CPU model, dependency versions, and available `nvidia-smi` state.
- Add `make benchmark-evidence` for paired `off`, `nonfinite_only`, `infeasibility_flag_only`, and `full` validation-overhead, memory, and isolated cold-compilation records across five process rounds.
- Add `make validate-gpu-multi`, a fail-closed two-GPU acceptance gate, and a compiled-backend regression test that keeps stream, current-device, and workspace state isolated across two CUDA devices.
- Add a pinned `requirements-ci.txt` hosted validation snapshot, a Python 3.10, 3.12, and 3.13 test matrix, a CPU-only benchmark diagnostic artifact, and a pre-commit gate in hosted CI.

### Changed

- Ship a pure-source package by default; normal installation no longer compiles the legacy CUDA extension.
- Require Python 3.10 or newer.
- Preserve `float64` solver precision, promote `float16`, `bfloat16`, and integer inputs to `float32`, and match SciPy's infeasible and invalid-cost behavior.
- Fall back to SciPy on unsupported CUDA runtimes and return the assignment to the input device.
- Make `make benchmark` fail closed by default: missing GPU or backend execution, out-of-memory, worker failure, package or implementation identity mismatch, incomplete process rounds, and parity failure all exit nonzero. `--allow-incomplete` is only for CPU-only diagnostics, and its rows remain ineligible for performance acceptance.
- Pin `make install-legacy` to the exact PyPI `0.0.6` baseline and fail if the checkout shadows that installed distribution.
- Split the publish workflow into separate build and publish jobs, pin action digests, publish through a protected environment with trusted publishing, and verify both the wheel and the source distribution against the release tag from outside the workspace.

### Migration

- Existing `batch_linear_assignment` and `assignment_to_indices` calls do not require API changes.
- Users who require the compiled CUDA implementation, including execution on T4 (`sm_75`), should remain on the maintenance-only `0.0.x` release line. Release `0.0.6` can be installed with `python -m pip install "torch-linear-assignment==0.0.6"`.
- The Triton path requires Linux, an NVIDIA GPU with compute capability 8.0 or newer, Torch 2.4 or newer, and Triton. Other supported environments continue through SciPy.

### Validation status

- Exact assignment parity and all GPU tests pass on NVIDIA L4 (`sm_89`), A100 (`sm_80`), and RTX PRO 6000 Blackwell (`sm_120`) author-run environments.
- H100 validation and isolated fresh-process cold-start measurements remain pending. Published warm timings are paired with same-machine SciPy and legacy baselines; they are not generalized across hardware.
- The published tables predate this benchmark harness and carry no JSONL provenance, so they are provisional author-run evidence. Publication-grade cold claims require a source-bound, five-process rerun on each advertised GPU class.
