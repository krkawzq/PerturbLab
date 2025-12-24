#!/usr/bin/env bash
# Setup dependencies for Mann-Whitney U C++ kernel

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LIB_DIR="${PROJECT_ROOT}/external/lib"

echo "============================================"
echo "Installing Dependencies"
echo "============================================"
echo "Library directory: ${LIB_DIR}"
echo ""

# Create lib directory
mkdir -p "${LIB_DIR}"

# ============================================
# 1. Install Highway SIMD library
# ============================================
echo "===> Installing Highway SIMD library"
HIGHWAY_DIR="${LIB_DIR}/highway"

if [ -d "${HIGHWAY_DIR}" ]; then
    echo "Highway already exists, updating..."
    (cd "${HIGHWAY_DIR}" && git pull)
else
    echo "Cloning Highway from GitHub..."
    git clone --depth 1 https://github.com/google/highway.git "${HIGHWAY_DIR}"
fi

echo "✅ Highway installed to ${HIGHWAY_DIR}"
echo ""

# ============================================
# 2. Check OpenMP availability
# ============================================
echo "===> Checking OpenMP"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"

if [[ "${OS}" == "linux" ]]; then
    if command -v apt &>/dev/null; then
        if ! dpkg -l | grep -q libomp-dev; then
            echo "Installing libomp-dev..."
            sudo apt-get update && sudo apt-get install -y libomp-dev
        else
            echo "libomp-dev already installed"
        fi
    elif command -v yum &>/dev/null; then
        if ! rpm -qa | grep -q libomp-devel; then
            echo "Installing libomp-devel..."
            sudo yum install -y libomp-devel
        else
            echo "libomp-devel already installed"
        fi
    fi
elif [[ "${OS}" == "darwin" ]]; then
    if command -v brew &>/dev/null; then
        if ! brew list libomp &>/dev/null; then
            echo "Installing libomp via Homebrew..."
            brew install libomp
        else
            echo "libomp already installed"
        fi
    else
        echo "⚠️  Homebrew not found. Please install from https://brew.sh/"
    fi
fi

echo "✅ OpenMP check complete"
echo ""

# ============================================
# Done
# ============================================
echo "============================================"
echo "Dependencies installation complete!"
echo "============================================"
echo ""
echo "Highway library: ${HIGHWAY_DIR}"
echo ""
echo "Next steps:"
echo "  1. Build the C++ kernels: ./build_cpp_kernels.sh"
echo "  2. Compile Cython extensions: python setup.py build_ext --inplace"
echo "  3. Run tests from project root"

