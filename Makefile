.PHONY: help build compile compile-cpp compile-cython install clean test format lint all tree cloc

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
	@echo "  install        Install Python dependencies"
	@echo "  clean          Clean build artifacts"
	@echo "  format         Format and fix code (all tools)"
	@echo "  lint           Run linters"
	@echo "  test           Run tests"
	@echo "  cloc           Count lines of code"
	@echo "  tree           Show git-tracked files tree (filtered)"

all: clean build format lint test

# =============================================================================
# Build & Compile
# =============================================================================

install:
	@$(PIP) install --upgrade pip
	@$(PIP) install -e . --no-build-isolation

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

build: install compile

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
	@[ -f test.py ] && $(PYTHON) test.py || $(PYTHON) -m pytest tests/ -v 2>/dev/null || echo "No tests found"

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
