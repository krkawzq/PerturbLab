#!/bin/bash
# Script to compile all PerturbLab extensions (Cython + C++)
# Usage: bash scripts/compile_extensions.sh

set -e  # Exit on error

echo "=========================================="
echo "PerturbLab Extensions Compilation"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# ==========================================
# Step 1: Compile Cython Extensions
# ==========================================

echo "Step 1: Compiling Cython extensions..."
echo "------------------------------------------"

if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "❌ Error: Python not found"
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo ""

# Check for required packages
echo "Checking dependencies..."
$PYTHON_CMD -c "import numpy; print(f'  NumPy: {numpy.__version__}')" || {
    echo "❌ NumPy not found. Install with: pip install numpy"
    exit 1
}

$PYTHON_CMD -c "import Cython; print(f'  Cython: {Cython.__version__}')" || {
    echo "⚠️  Cython not found. Install with: pip install cython"
    echo "Skipping Cython compilation..."
    SKIP_CYTHON=1
}

if [ -z "$SKIP_CYTHON" ]; then
    echo ""
    echo "Compiling Cython extensions..."
    $PYTHON_CMD setup.py build_ext --inplace
    echo "✅ Cython extensions compiled"
else
    echo "⚠️  Skipping Cython compilation"
fi

echo ""

# ==========================================
# Step 2: Compile C++ Library
# ==========================================

echo "Step 2: Compiling C++ library..."
echo "------------------------------------------"

if [ -f "$SCRIPT_DIR/build_cpp_kernels.sh" ]; then
    bash "$SCRIPT_DIR/build_cpp_kernels.sh"
else
    echo "⚠️  C++ build script not found"
    echo "   Expected: $SCRIPT_DIR/build_cpp_kernels.sh"
fi

echo ""
echo "=========================================="
echo "Compilation Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✅ Cython extensions: compiled (if available)"
echo "  ✅ C++ library: compiled (if available)"
echo ""
echo "To test the installation, run:"
echo "  python -c 'import perturblab; print(perturblab.__version__)'"
echo ""

