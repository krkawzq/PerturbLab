"""Setup script for PerturbLab with optional Cython and C++ extensions.

This setup.py provides:
1. Cython extension compilation (optional, for performance kernels)
2. Graceful fallback if compilation fails
3. Cross-platform support (Linux, macOS, Windows)

The C++ library (libmwu_kernel.so) must be compiled separately using CMake.
See scripts/setup_cpp_deps.sh and scripts/build_cpp_kernels.sh.

Available Cython Extensions:
- perturblab.kernels.mapping.backends.cython._lookup
- perturblab.kernels.mapping.backends.cython._bipartite_query
- perturblab.kernels.statistics.backends.cython.mannwhitneyu
- perturblab.kernels.statistics.backends.cython.ttest
- perturblab.kernels.statistics.backends.cython.group_ops
- perturblab.kernels.statistics.backends.cython._hvg (NEW)
- perturblab.kernels.statistics.backends.cython._scale (NEW)

Usage:
    # Install package only (no compilation)
    pip install -e .
    
    # Compile Cython extensions (optional, for 5-10x speedup)
    python setup.py build_ext --inplace
    
    # Compile C++ library (optional, for 100-300x speedup)
    ./scripts/setup_cpp_deps.sh
    ./scripts/build_cpp_kernels.sh

Platform-Specific Notes:
    Linux: Uses GCC with -fopenmp
    macOS: Uses Clang with libomp from Homebrew (brew install libomp)
    Windows: Uses MSVC with /openmp
"""

import os
import sys
import platform
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

# =============================================================================
# Check for Cython and NumPy
# =============================================================================

USE_CYTHON = False
NUMPY_AVAILABLE = False

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    print("⚠️  Cython not available, skipping Cython extensions")
    print("   Install with: pip install cython")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("⚠️  NumPy not available, skipping extensions")
    print("   Install with: pip install numpy")


# =============================================================================
# Custom build_ext Command
# =============================================================================

class build_ext(_build_ext):
    """Custom build_ext to handle compilation failures gracefully."""
    
    def run(self):
        """Run build_ext with graceful error handling."""
        try:
            super().run()
            print("✅ Extensions compiled successfully")
        except Exception as e:
            print(f"⚠️  Extension compilation failed: {e}")
            print("   Package will still work with slower fallback implementations")
    
    def build_extension(self, ext):
        """Build individual extension with error handling."""
        try:
            super().build_extension(ext)
            print(f"  ✅ {ext.name}")
        except Exception as e:
            print(f"  ❌ {ext.name}: {e}")


# =============================================================================
# Extension Configuration
# =============================================================================

def get_extensions():
    """Get list of Cython extensions to compile."""
    if not (USE_CYTHON and NUMPY_AVAILABLE):
        return []
    
    extensions = []
    
    # Determine OpenMP flags based on platform
    if sys.platform == "darwin":
        # macOS with Homebrew libomp
        # Check both ARM64 (Apple Silicon) and x86_64 (Intel) paths
        openmp_compile_args = ["-Xpreprocessor", "-fopenmp"]
        openmp_link_args = ["-lomp"]
        openmp_libs = []
        
        # Potential libomp paths (ARM64 and Intel)
        potential_paths = [
            "/opt/homebrew/opt/libomp",      # ARM64 (Apple Silicon)
            "/usr/local/opt/libomp",         # x86_64 (Intel)
        ]
        
        openmp_lib_dirs = []
        openmp_include_dirs = []
        
        for base_path in potential_paths:
            lib_path = f"{base_path}/lib"
            inc_path = f"{base_path}/include"
            if Path(lib_path).exists():
                openmp_lib_dirs.append(lib_path)
            if Path(inc_path).exists():
                openmp_include_dirs.append(inc_path)
        
        if not openmp_lib_dirs:
            print("⚠️  Warning: libomp not found. Install with: brew install libomp")
            
    elif sys.platform == "win32":
        # Windows MSVC
        openmp_compile_args = ["/openmp"]
        openmp_link_args = []
        openmp_libs = []
        openmp_lib_dirs = []
        openmp_include_dirs = []
    else:
        # Linux GCC/Clang
        openmp_compile_args = ["-fopenmp"]
        openmp_link_args = ["-fopenmp"]
        openmp_libs = ["gomp"]
        openmp_lib_dirs = []
        openmp_include_dirs = []
    
    # Base compile args
    if sys.platform == "win32":
        base_compile_args = ["/O2", "/W3"]
    else:
        base_compile_args = ["-O3", "-ffast-math", "-Wall"]
        
        # Add -march=native if not cross-compiling
        # Skip for ARM64 macOS to avoid compatibility issues
        import platform
        machine = platform.machine().lower()
        if not (sys.platform == "darwin" and machine in ["arm64", "aarch64"]):
            base_compile_args.append("-march=native")
    
    # ==========================================================================
    # Mapping Kernels
    # ==========================================================================
    
    mapping_cython_dir = "perturblab/kernels/mapping/backends/cython"
    
    # Check if files exist
    if Path(f"{mapping_cython_dir}/_lookup.pyx").exists():
        extensions.append(
            Extension(
                "perturblab.kernels.mapping.backends.cython._lookup",
                sources=[f"{mapping_cython_dir}/_lookup.pyx"],
                include_dirs=[np.get_include()],
                language="c",
                extra_compile_args=base_compile_args,
            )
        )
    
    if Path(f"{mapping_cython_dir}/_bipartite_query.pyx").exists():
        extensions.append(
            Extension(
                "perturblab.kernels.mapping.backends.cython._bipartite_query",
                sources=[f"{mapping_cython_dir}/_bipartite_query.pyx"],
                include_dirs=[np.get_include()],
                language="c",
                extra_compile_args=base_compile_args,
            )
        )
    
    # ==========================================================================
    # Statistics Kernels (with OpenMP)
    # ==========================================================================
    
    statistics_cython_dir = "perturblab/kernels/statistics/backends/cython"
    
    # Mann-Whitney U kernel
    if Path(f"{statistics_cython_dir}/mannwhitneyu.pyx").exists():
        extensions.append(
            Extension(
                "perturblab.kernels.statistics.backends.cython.mannwhitneyu",
                sources=[f"{statistics_cython_dir}/mannwhitneyu.pyx"],
                include_dirs=[np.get_include()] + openmp_include_dirs,
                libraries=openmp_libs,
                library_dirs=openmp_lib_dirs,
                language="c",
                extra_compile_args=base_compile_args + openmp_compile_args,
                extra_link_args=openmp_link_args,
            )
        )
    
    # T-test kernel
    if Path(f"{statistics_cython_dir}/ttest.pyx").exists():
        extensions.append(
            Extension(
                "perturblab.kernels.statistics.backends.cython.ttest",
                sources=[f"{statistics_cython_dir}/ttest.pyx"],
                include_dirs=[np.get_include()] + openmp_include_dirs,
                libraries=openmp_libs,
                library_dirs=openmp_lib_dirs,
                language="c",
                extra_compile_args=base_compile_args + openmp_compile_args,
                extra_link_args=openmp_link_args,
            )
        )
    
    # Group operations kernel
    if Path(f"{statistics_cython_dir}/group_ops.pyx").exists():
        extensions.append(
            Extension(
                "perturblab.kernels.statistics.backends.cython.group_ops",
                sources=[f"{statistics_cython_dir}/group_ops.pyx"],
                include_dirs=[np.get_include()] + openmp_include_dirs,
                libraries=openmp_libs,
                library_dirs=openmp_lib_dirs,
                language="c",
                extra_compile_args=base_compile_args + openmp_compile_args,
                extra_link_args=openmp_link_args,
            )
        )
    
    # HVG (Highly Variable Genes) kernel
    if Path(f"{statistics_cython_dir}/_hvg.pyx").exists():
        extensions.append(
            Extension(
                "perturblab.kernels.statistics.backends.cython._hvg",
                sources=[f"{statistics_cython_dir}/_hvg.pyx"],
                include_dirs=[np.get_include()] + openmp_include_dirs,
                libraries=openmp_libs,
                library_dirs=openmp_lib_dirs,
                language="c",
                extra_compile_args=base_compile_args + openmp_compile_args,
                extra_link_args=openmp_link_args,
            )
        )
    
    # Scale (standardization) kernel
    if Path(f"{statistics_cython_dir}/_scale.pyx").exists():
        extensions.append(
            Extension(
                "perturblab.kernels.statistics.backends.cython._scale",
                sources=[f"{statistics_cython_dir}/_scale.pyx"],
                include_dirs=[np.get_include()] + openmp_include_dirs,
                libraries=openmp_libs,
                library_dirs=openmp_lib_dirs,
                language="c",
                extra_compile_args=base_compile_args + openmp_compile_args,
                extra_link_args=openmp_link_args,
            )
        )
    
    return extensions


# =============================================================================
# Main Setup
# =============================================================================

if __name__ == "__main__":
    if USE_CYTHON and NUMPY_AVAILABLE:
        print("="*80)
        print("Compiling Cython Extensions")
        print("="*80)
        print(f"Platform: {sys.platform}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"Cython: available")
        print(f"NumPy: {np.__version__}")
        print("="*80)
        
        ext_modules = cythonize(
            get_extensions(),
            compiler_directives={
                "language_level": "3",
                "boundscheck": False,
                "wraparound": False,
                "nonecheck": False,
                "cdivision": True,
                "initializedcheck": False,
                "infer_types": True,
            },
            annotate=False,  # Set to True to generate HTML annotation files for debugging
        )
    else:
        print("="*80)
        print("Skipping Cython Extensions")
        print("="*80)
        if not USE_CYTHON:
            print("Reason: Cython not available")
        if not NUMPY_AVAILABLE:
            print("Reason: NumPy not available")
        print("Package will use slower fallback implementations")
        print("="*80)
        ext_modules = []
    
    setup(
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
    )
