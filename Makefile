.PHONY: help build compile compile-cpp compile-cython clean test test-fast test-imports test-types test-download test-download-no-network test-verbose test-coverage test-quick test-models format lint all tree cloc

# Config
PYTHON := python3
PIP := $(PYTHON) -m pip
CMAKE_BUILD_DIR := build/cmake
INSTALL_DIR := perturblab/kernels/statistics/backends/cpp

.DEFAULT_GOAL := help

# =============================================================================
# Core Commands
# =============================================================================

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build          Build everything (deps + compile)"
	@echo "  compile        Compile C++ and Cython"
	@echo "  compile-cpp    Compile C++ only"
	@echo "  compile-cython Compile Cython only"
	@echo "  clean          Clean build artifacts"
	@echo "  format         Format and fix code (all tools)"
	@echo "  lint           Run linters"
	@echo "  test                    Run all tests"
	@echo "  test-fast               Run tests (fail fast)"
	@echo "  test-imports            Run import tests only"
	@echo "  test-types              Run type tests only"
	@echo "  test-metrics            Run metrics tests only"
	@echo "  test-preprocessing      Run preprocessing tests only"
	@echo "  test-pp                 Alias for test-preprocessing"
	@echo "  test-download           Run download & registry tests"
	@echo "  test-download-no-network Run download tests (skip network)"
	@echo "  test-verbose            Run tests with verbose output"
	@echo "  test-coverage           Run tests with coverage report"
	@echo "  test-quick              Quick sanity test (import check)"
	@echo "  test-models             Test model registry"
	@echo "  cloc           Count lines of code"
	@echo "  tree           Show git-tracked files tree (filtered)"

all: clean build format lint test

# =============================================================================
# Build & Compile
# =============================================================================

setup-deps:
	@[ -f scripts/setup_cpp_deps.sh ] && ./scripts/setup_cpp_deps.sh || true

compile-cpp: setup-deps
	@mkdir -p $(CMAKE_BUILD_DIR)
	@cd $(CMAKE_BUILD_DIR) && cmake ../.. -DCMAKE_BUILD_TYPE=Release
	@cd $(CMAKE_BUILD_DIR) && cmake --build . --config Release -j$$(nproc 2>/dev/null || echo 4)
	@cd $(CMAKE_BUILD_DIR) && cmake --install .

compile-cython:
	@$(PYTHON) -c "import numpy, Cython" 2>/dev/null || $(PIP) install numpy cython
	@$(PYTHON) setup.py build_ext --inplace

compile: compile-cpp compile-cython

build: compile

rebuild: clean build

# =============================================================================
# Code Quality & Formatting
# =============================================================================

format: format-imports format-code fix-lint
	@echo "✓ Code formatted and fixed"

format-imports:
	@command -v isort >/dev/null 2>&1 || $(PIP) install isort
	@isort perturblab/ --profile black

format-code:
	@command -v black >/dev/null 2>&1 || $(PIP) install black
	@black perturblab/ --line-length 100

fix-lint:
	@command -v ruff >/dev/null 2>&1 || $(PIP) install ruff
	@ruff check perturblab/ --fix --select I,F401,F841,UP,C90,N,E,W
	@ruff format perturblab/

remove-unused-imports:
	@command -v autoflake >/dev/null 2>&1 || $(PIP) install autoflake
	@autoflake --in-place --remove-all-unused-imports --remove-unused-variables \
		--recursive perturblab/

lint: lint-ruff lint-pyright lint-mypy
	@echo "✓ All linters passed"

lint-ruff:
	@command -v ruff >/dev/null 2>&1 || $(PIP) install ruff
	@ruff check perturblab/

lint-pyright:
	@command -v pyright >/dev/null 2>&1 || npm install -g pyright 2>/dev/null || echo "pyright not available"
	@command -v pyright >/dev/null 2>&1 && pyright perturblab/ || true

lint-mypy:
	@command -v mypy >/dev/null 2>&1 || $(PIP) install mypy
	@mypy perturblab/ --ignore-missing-imports --no-strict-optional || true

lint-flake8:
	@command -v flake8 >/dev/null 2>&1 || $(PIP) install flake8
	@flake8 perturblab/ --max-line-length=100 --ignore=E203,W503,E501

check: format lint test

# =============================================================================
# Testing & Analysis
# =============================================================================

test:
	@echo "Running tests..."
	@command -v pytest >/dev/null 2>&1 || $(PIP) install pytest
	@uv run pytest tests/ -v --tb=short || $(PYTHON) -m pytest tests/ -v --tb=short

test-fast:
	@echo "Running tests (fail fast)..."
	@command -v pytest >/dev/null 2>&1 || $(PIP) install pytest
	@uv run pytest tests/ -x -v || $(PYTHON) -m pytest tests/ -x -v

test-imports:
	@echo "Running import tests..."
	@command -v pytest >/dev/null 2>&1 || $(PIP) install pytest
	@uv run pytest tests/test_imports.py -v || $(PYTHON) -m pytest tests/test_imports.py -v

test-types:
	@echo "Running type tests..."
	@command -v pytest >/dev/null 2>&1 || $(PIP) install pytest
	@uv run pytest tests/test_types.py -v || $(PYTHON) -m pytest tests/test_types.py -v

test-download:
	@echo "Running download and registry tests..."
	@command -v pytest >/dev/null 2>&1 || $(PIP) install pytest
	@uv run pytest tests/test_download.py -v || $(PYTHON) -m pytest tests/test_download.py -v

test-download-no-network:
	@echo "Running download tests (skipping network tests)..."
	@command -v pytest >/dev/null 2>&1 || $(PIP) install pytest
	@uv run pytest tests/test_download.py -v -m "not network" || $(PYTHON) -m pytest tests/test_download.py -v -m "not network"

test-metrics:
	@echo "Running metrics tests..."
	@command -v pytest >/dev/null 2>&1 || $(PIP) install pytest
	@uv run pytest tests/test_metrics.py -v

test-preprocessing:
	@echo "Running preprocessing tests..."
	@command -v pytest >/dev/null 2>&1 || $(PIP) install pytest
	@uv run pytest tests/test_preprocessing.py -v

test-pp: test-preprocessing

test-verbose:
	@echo "Running tests (verbose)..."
	@command -v pytest >/dev/null 2>&1 || $(PIP) install pytest
	@uv run pytest tests/ -vv -s || $(PYTHON) -m pytest tests/ -vv -s

test-coverage:
	@echo "Running tests with coverage..."
	@command -v pytest >/dev/null 2>&1 || $(PIP) install pytest pytest-cov
	@uv run pytest tests/ --cov=perturblab --cov-report=html --cov-report=term || \
		$(PYTHON) -m pytest tests/ --cov=perturblab --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

test-quick:
	@echo "Running quick sanity test..."
	@uv run python -c "import perturblab; print(f'✓ PerturbLab v{perturblab.__version__} imported successfully')" || \
		$(PYTHON) -c "import perturblab; print(f'✓ PerturbLab v{perturblab.__version__} imported successfully')"

test-models:
	@echo "Testing model imports..."
	@uv run python -c "from perturblab import MODELS; print(f'✓ Found {len(list(MODELS._child_registries))} model registries: {list(MODELS._child_registries.keys())}')" || \
		$(PYTHON) -c "from perturblab import MODELS; print(f'✓ Found {len(list(MODELS._child_registries))} model registries: {list(MODELS._child_registries.keys())}')"

cloc:
	@command -v cloc >/dev/null 2>&1 || { echo "Install: sudo apt install cloc"; exit 1; }
	@cloc . --exclude-dir=build,dist,__pycache__,.git,.eggs,perturblab.egg-info,external,weights,forks,perturblab_v0.1,perturblab_v0.2,_fast_transformers \
		--exclude-ext=.pyc,.pyo,.so,.pyd,.o,.a,.dylib,.dll --vcs=git

tree:
	@git ls-tree -r --name-only HEAD | grep -v '_fast_transformers' | tree --fromfile .

benchmark: compile
	@[ -f benchmarks/benchmark.py ] && $(PYTHON) benchmarks/benchmark.py || echo "No benchmarks"

# =============================================================================
# Clean
# =============================================================================

clean: clean-build clean-pyc clean-compiled

clean-build:
	@rm -rf build/ dist/ *.egg-info .eggs/

clean-pyc:
	@find . -type f -name '*.py[co]' -delete
	@find . -type d -name '__pycache__' -delete
	@find . -type d -name '.pytest_cache' -delete
	@find . -type d -name '.mypy_cache' -delete
	@find . -type d -name '.ruff_cache' -delete

clean-compiled:
	@rm -rf $(INSTALL_DIR)/libmwu_kernel.*
	@find . -name '*.so' -not -path "./external/*" -delete
	@find . -name '*.pyd' -not -path "./external/*" -delete
	@find . -name '*.c' -path "*/perturblab/kernels/*" -delete
	@find . -name '*.cpp' -path "*/perturblab/kernels/*" -path "*/cython/*" -delete

# =============================================================================
# Development
# =============================================================================

dev-install:
	@$(PIP) install black isort ruff flake8 pytest mypy autoflake autopep8

check-cpp:
	@[ -f $(INSTALL_DIR)/libmwu_kernel.so ] && echo "✓ C++ library installed" || echo "✗ C++ library missing"

check-cython:
	@find perturblab -name "*.so" -o -name "*.pyd" | head -5

git-status:
	@git status --short | grep -v '\.so$$' | grep -v '\.pyd$$' | grep -v '__pycache__' || echo "Clean"

info:
	@echo "Python:  $$($(PYTHON) --version 2>&1)"
	@echo "CMake:   $$(cmake --version 2>&1 | head -n1)"
	@echo "Git:     $$(git --version 2>&1)"
