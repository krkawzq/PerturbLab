"""
Numba JIT wrapper for soft dependency.

This module provides wrappers for numba's JIT decorators that gracefully degrade
when numba is not available, avoiding hard dependencies.
"""

import functools
from typing import Any, Callable, Optional, TypeVar

# 尝试导入 numba
try:
    import numba
    from numba import cuda
    
    NUMBA_AVAILABLE = True
    _numba_jit = numba.jit
    _numba_njit = numba.njit
    _numba_vectorize = numba.vectorize
    _numba_guvectorize = numba.guvectorize
    _numba_cuda_jit = cuda.jit if hasattr(cuda, 'jit') else None
    
except ImportError:
    NUMBA_AVAILABLE = False
    numba = None
    _numba_jit = None
    _numba_njit = None
    _numba_vectorize = None
    _numba_guvectorize = None
    _numba_cuda_jit = None


F = TypeVar('F', bound=Callable[..., Any])


def _identity_decorator(*args, **kwargs) -> Callable[[F], F]:
    """
    Identity decorator that does nothing.
    
    Used as a fallback when numba is not available.
    """
    def decorator(func: F) -> F:
        # 添加一个标记，表明这是未加速的版本
        func.__numba_available__ = False  # type: ignore
        return func
    
    # 处理 @jit 和 @jit(...) 两种用法
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        # @jit 直接使用
        return decorator(args[0])
    else:
        # @jit(...) 带参数使用
        return decorator


def jit(*args, **kwargs) -> Callable[[F], F]:
    """
    JIT decorator wrapper.
    
    If numba is available, uses numba.jit. Otherwise, returns the original function.
    
    Args:
        *args: Positional arguments to pass to numba.jit
        **kwargs: Keyword arguments to pass to numba.jit
        
    Returns:
        Decorated function (or original function if numba is not available)
        
    Examples:
        >>> @jit
        ... def add(a, b):
        ...     return a + b
        
        >>> @jit(nopython=True)
        ... def multiply(a, b):
        ...     return a * b
    """
    if NUMBA_AVAILABLE:
        return _numba_jit(*args, **kwargs)
    else:
        return _identity_decorator(*args, **kwargs)


def njit(*args, **kwargs) -> Callable[[F], F]:
    """
    No-Python JIT decorator wrapper.
    
    If numba is available, uses numba.njit. Otherwise, returns the original function.
    
    Args:
        *args: Positional arguments to pass to numba.njit
        **kwargs: Keyword arguments to pass to numba.njit
        
    Returns:
        Decorated function (or original function if numba is not available)
        
    Examples:
        >>> @njit
        ... def add(a, b):
        ...     return a + b
        
        >>> @njit(parallel=True)
        ... def sum_array(arr):
        ...     total = 0
        ...     for i in range(len(arr)):
        ...         total += arr[i]
        ...     return total
    """
    if NUMBA_AVAILABLE:
        return _numba_njit(*args, **kwargs)
    else:
        return _identity_decorator(*args, **kwargs)


def vectorize(*args, **kwargs) -> Callable[[F], F]:
    """
    Vectorize decorator wrapper.
    
    If numba is available, uses numba.vectorize. Otherwise, returns the original function.
    
    Args:
        *args: Positional arguments to pass to numba.vectorize
        **kwargs: Keyword arguments to pass to numba.vectorize
        
    Returns:
        Decorated function (or original function if numba is not available)
        
    Examples:
        >>> @vectorize(['float64(float64, float64)'])
        ... def add(a, b):
        ...     return a + b
    """
    if NUMBA_AVAILABLE:
        return _numba_vectorize(*args, **kwargs)
    else:
        return _identity_decorator(*args, **kwargs)


def guvectorize(*args, **kwargs) -> Callable[[F], F]:
    """
    Generalized universal function decorator wrapper.
    
    If numba is available, uses numba.guvectorize. Otherwise, returns the original function.
    
    Args:
        *args: Positional arguments to pass to numba.guvectorize
        **kwargs: Keyword arguments to pass to numba.guvectorize
        
    Returns:
        Decorated function (or original function if numba is not available)
        
    Examples:
        >>> @guvectorize(['void(float64[:], float64[:])'], '(n)->(n)')
        ... def cumsum(x, res):
        ...     res[0] = x[0]
        ...     for i in range(1, len(x)):
        ...         res[i] = res[i-1] + x[i]
    """
    if NUMBA_AVAILABLE:
        return _numba_guvectorize(*args, **kwargs)
    else:
        return _identity_decorator(*args, **kwargs)


def cuda_jit(*args, **kwargs) -> Callable[[F], F]:
    """
    CUDA JIT decorator wrapper.
    
    If numba.cuda is available, uses numba.cuda.jit. Otherwise, returns the original function.
    
    Args:
        *args: Positional arguments to pass to numba.cuda.jit
        **kwargs: Keyword arguments to pass to numba.cuda.jit
        
    Returns:
        Decorated function (or original function if numba.cuda is not available)
        
    Examples:
        >>> @cuda_jit
        ... def kernel(arr):
        ...     i = cuda.grid(1)
        ...     if i < arr.size:
        ...         arr[i] += 1
    """
    if NUMBA_AVAILABLE and _numba_cuda_jit is not None:
        return _numba_cuda_jit(*args, **kwargs)
    else:
        return _identity_decorator(*args, **kwargs)


def is_numba_available() -> bool:
    """
    Check if numba is available.
    
    Returns:
        bool: True if numba is available, False otherwise
        
    Examples:
        >>> if is_numba_available():
        ...     print("Numba acceleration enabled")
        ... else:
        ...     print("Running in pure Python mode")
    """
    return NUMBA_AVAILABLE


def get_numba_version() -> Optional[str]:
    """
    Get the version of numba if available.
    
    Returns:
        Optional[str]: Version string if numba is available, None otherwise
        
    Examples:
        >>> version = get_numba_version()
        >>> if version:
        ...     print(f"Numba version: {version}")
    """
    if NUMBA_AVAILABLE and numba is not None:
        return numba.__version__
    return None


__all__ = [
    'jit',
    'njit',
    'vectorize',
    'guvectorize',
    'cuda_jit',
    'is_numba_available',
    'get_numba_version',
    'NUMBA_AVAILABLE',
]

