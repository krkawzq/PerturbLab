"""
Preprocessing module for PerturbLab.

High-performance implementations of single-cell preprocessing algorithms,
optimized with C++/SIMD/OpenMP for 2-15x speedup over scanpy.
"""

from ._normalization import normalize_total
from ._scale import scale

__all__ = [
    "normalize_total",
    "scale",
]
