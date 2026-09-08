PYTHON ?= python
BENCHMARK_ARGS ?=

.PHONY: help install install-legacy install-dev test benchmark benchmark-evidence validate validate-cpu validate-gpu validate-gpu-multi clean

help:
	@echo "install      editable pure-source install"
	@echo "install-legacy install and verify the exact PyPI 0.0.6 baseline"
	@echo "install-dev  install plus the test tooling"
	@echo "test         run the test suite"
	@echo "benchmark    run SciPy and available GPU benchmark comparisons"
	@echo "benchmark-evidence record paired Triton validation, memory, and isolated cold evidence"
	@echo "validate-cpu install-dev and test - runs anywhere"
	@echo "validate-gpu GPU test and metadata lane - skips cleanly without a CUDA device"
	@echo "validate-gpu-multi two-GPU compiled Triton acceptance - fails unless two eligible GPUs are visible"
	@echo "validate     correctness tests once, with GPU metadata when available"
	@echo ""
	@echo "Linting is not part of this gate; it runs through pre-commit."
	@echo "Set TLA_BUILD_LEGACY_CUDA=1 to build the old CUDA backend for comparisons."

install:
	$(PYTHON) -m pip install -e . --no-build-isolation

install-legacy:
	$(PYTHON) -m pip install pandas tqdm
	$(PYTHON) -m pip uninstall -y torch-hungarian
	$(PYTHON) -m pip install --no-build-isolation --no-deps --force-reinstall "torch-linear-assignment==0.0.6"
	@set -e; \
	verify_dir=$$(mktemp -d); \
	trap 'rm -rf "$$verify_dir"' EXIT; \
	cd "$$verify_dir"; \
	TLA_CHECKOUT="$(CURDIR)" $(PYTHON) -c "import importlib.metadata as m, os; from pathlib import Path; import torch_linear_assignment as package; version = m.version('torch-linear-assignment'); origin = Path(package.__file__).resolve(); checkout = Path(os.environ['TLA_CHECKOUT']).resolve(); assert version == '0.0.6', version; assert checkout not in origin.parents, f'checkout shadows legacy install: {origin}'; print(f'Legacy distribution: {version} | module: {origin}')"

install-dev: install
	$(PYTHON) -m pip install pandas pytest tqdm

test: install-dev
	$(PYTHON) -m pytest tests/ -v

benchmark:
	@set -e; \
	benchmark_dir=$$(mktemp -d); \
	trap 'rm -rf "$$benchmark_dir"' EXIT; \
	cp tests/benchmark.py "$$benchmark_dir/benchmark.py"; \
	package_version=$$(cd "$$benchmark_dir" && $(PYTHON) -c "import importlib.metadata as m; installed = {d.metadata['Name']: d.version for d in m.distributions()}; print(installed.get('torch-hungarian') or installed.get('torch-linear-assignment') or '')"); \
	if [ "$$package_version" = "0.0.6" ]; then expected_implementation=legacy_cuda; else expected_implementation=triton; fi; \
	TLA_BENCHMARK_SOURCE_PATH="$(CURDIR)/tests/benchmark.py" \
	TLA_BENCHMARK_GIT_REVISION="$$(git rev-parse HEAD)" \
	env -u CUDA_LAUNCH_BLOCKING $(PYTHON) "$$benchmark_dir/benchmark.py" \
		--expect-package-version "$$package_version" \
		--expect-implementation "$$expected_implementation" \
		$(BENCHMARK_ARGS)

benchmark-evidence:
	$(MAKE) benchmark BENCHMARK_ARGS="--backends scipy,triton --workers 300 --tasks 100,300,600 --batches 208 --dtypes float32 --validation-modes off,nonfinite_only,infeasibility_flag_only,full --process-rounds 5 --backend-order alternate --warmup 3 --repetitions 30 --expect-implementation triton $(BENCHMARK_ARGS)"

validate-cpu: test

# Reports the device it validated against, so benchmark numbers pasted into a PR
# carry their hardware. Exits 0 with a notice when no CUDA device is visible.
validate-gpu: test
	@set -e; \
	if ! $(PYTHON) -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then \
		echo "GPU lane skipped: no CUDA device visible"; \
		exit 0; \
	fi; \
	echo "GPU validation: AUTHOR-RUN (not hosted CI)"; \
	$(PYTHON) -c "import platform, sys; print(f'Python: {sys.version.split()[0]} | platform: {platform.platform()}')"; \
	$(PYTHON) -c "import importlib.metadata as m, importlib.util as u; print('Triton:', m.version('triton') if u.find_spec('triton') else 'unavailable')"; \
	$(PYTHON) -c "import torch; p = torch.cuda.get_device_properties(0); print(f'GPU: {p.name} sm_{p.major}{p.minor} | torch {torch.__version__} | cuda {torch.version.cuda}')"

# This target is a fail-closed acceptance gate: a skipped test must not be
# mistaken for evidence that the private compiled backend works on two GPUs.
validate-gpu-multi: install-dev
	@set -e; \
	$(PYTHON) -c "import importlib, importlib.metadata as m, importlib.util as u, platform, re, sys, torch; triton = m.version('triton') if u.find_spec('triton') else 'unavailable'; print('Python:', sys.version.replace(chr(10), ' ')); print(f'Platform: {platform.platform()}'); print(f'Torch: {torch.__version__}'); print(f'Torch CUDA: {torch.version.cuda}'); print(f'Triton: {triton}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); [print(f'GPU {index}: name={torch.cuda.get_device_properties(index).name} | compute_capability={torch.cuda.get_device_capability(index)[0]}.{torch.cuda.get_device_capability(index)[1]} | total_memory_bytes={torch.cuda.get_device_properties(index).total_memory} | multiprocessors={torch.cuda.get_device_properties(index).multi_processor_count}') for index in range(torch.cuda.device_count())]; version = re.match(r'(\\d+)\\.(\\d+)', torch.__version__); eligible = [index for index in range(torch.cuda.device_count()) if torch.cuda.get_device_capability(index) >= (8, 0)]; valid = sys.platform == 'linux' and torch.cuda.is_available() and torch.version.cuda is not None and version is not None and tuple(map(int, version.groups())) >= (2, 4) and u.find_spec('triton') is not None and len(eligible) >= 2; sys.exit(0 if valid else 'validate-gpu-multi requires Linux, CUDA-enabled PyTorch 2.4+, Triton, and at least two visible GPUs with compute capability >= 8.0')"; \
	if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; else echo "nvidia-smi: unavailable"; fi; \
	$(PYTHON) -m pytest -v tests/test_triton.py::test_batch_linear_assignment_compiled_isolates_two_cuda_devices

validate: export CUDA_LAUNCH_BLOCKING := 1
validate: validate-cpu validate-gpu

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find torch_linear_assignment -name "_backend*.so" -delete
