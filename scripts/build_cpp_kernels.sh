#!/bin/bash
# Build script for PerturbLab C++ statistical kernels

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build_cpp"

echo "============================================"
echo "Building PerturbLab C++ Statistical Kernels"
echo "============================================"
echo "Project root: ${PROJECT_ROOT}"
echo "Build directory: ${BUILD_DIR}"
echo ""

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Run CMake
echo "Running CMake..."
cmake "${PROJECT_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo "Compiling..."
make -j$(nproc)

# Install
echo ""
echo "Installing..."
make install

echo ""
echo "============================================"
echo "Build complete!"
echo "============================================"
echo "Library installed to: ${PROJECT_ROOT}/perturblab/kernels/statistics/backends/cpp/"
ls -lh "${PROJECT_ROOT}"/perturblab/kernels/statistics/backends/cpp/libmwu_kernel.* 2>/dev/null || echo "Note: Library files may have different extensions on different platforms"

echo ""
echo "✅ C++ backend ready for use!"

