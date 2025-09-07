"""
Registry for numpy function compatibility.
"""

import numpy as np
from typing import Optional, Dict, Callable
from functools import wraps

_numpy_functions: Dict[str, Callable] = {}


def _make_key(np_func: Callable) -> str:
    """Return a unique key for a NumPy function based on its module and name."""
    return f"{np_func.__module__}.{np_func.__name__}"


def register_numpy_function(full_name: str, func: Callable) -> None:
    """Register a numpy function with its wrapped version."""
    _numpy_functions[full_name] = func


def get_numpy_function(full_name: str) -> Optional[Callable]:
    """Get the wrapped version of a numpy function if it exists."""
    return _numpy_functions.get(full_name)


def wrap_and_register_numpy_function(np_func: Callable) -> Callable:
    """Decorator to wrap and register a numpy function."""
    from .array import DimensionalArray

    key = _make_key(np_func)
    if key in _numpy_functions:
        return _numpy_functions[key]

    @wraps(np_func)
    def wrapper(*args, **kwargs):
        # Convert dbz arrays to numpy arrays for input
        args = [(arg.array if isinstance(arg, DimensionalArray) else arg) for arg in args]
        kwargs = {k: (v.array if isinstance(v, DimensionalArray) else v) for k, v in kwargs.items()}

        # Call original numpy function
        result = np_func(*args, **kwargs)

        # Convert result back to dbz array
        if isinstance(result, np.ndarray):
            return DimensionalArray(result)
        elif isinstance(result, tuple):
            return tuple(DimensionalArray(r) if isinstance(r, np.ndarray) else r for r in result)
        return result

    register_numpy_function(key, wrapper)
    return wrapper
