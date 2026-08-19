PYTHON ?= python

.PHONY: help install install-legacy install-dev test benchmark validate validate-cpu validate-gpu clean

help:
	@echo "install      editable pure-source install"
	@echo "install-legacy install the latest PyPI release below 0.1.0"
	@echo "install-dev  install plus the test tooling"
	@echo "test         run the test suite"
	@echo "benchmark    run SciPy and available GPU benchmark comparisons"
	@echo "validate-cpu install-dev and test - runs anywhere"
	@echo "validate-gpu GPU test and metadata lane - skips cleanly without a CUDA device"
	@echo "validate     correctness tests once, with GPU metadata when available"
	@echo ""
	@echo "Linting is not part of this gate; it runs through pre-commit."
	@echo "Set TLA_BUILD_LEGACY_CUDA=1 to build the old CUDA backend for comparisons."

install:
	$(PYTHON) -m pip install -e . --no-build-isolation

install-legacy:
	$(PYTHON) -m pip install pandas tqdm
	$(PYTHON) -m pip install --no-build-isolation --no-deps --force-reinstall "torch-linear-assignment<0.1.0"

install-dev: install
	$(PYTHON) -m pip install pandas pytest tqdm

test: install-dev
	$(PYTHON) -m pytest tests/ -v

benchmark:
	env -u CUDA_LAUNCH_BLOCKING $(PYTHON) tests/benchmark.py

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

validate: export CUDA_LAUNCH_BLOCKING := 1
validate: validate-cpu validate-gpu

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find torch_linear_assignment -name "_backend*.so" -delete
