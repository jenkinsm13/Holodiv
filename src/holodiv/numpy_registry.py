"""
Registry for numpy function compatibility.
"""

import numpy as np
from typing import Optional, Dict, Callable
from functools import wraps

_numpy_functions: Dict[str, Callable] = {}

def register_numpy_function(name: str, func: Callable) -> None:
    """Register a numpy function with its wrapped version."""
    _numpy_functions[name] = func

def get_numpy_function(name: str) -> Optional[Callable]:
    """Get the wrapped version of a numpy function if it exists."""
    return _numpy_functions.get(name)

def wrap_and_register_numpy_function(np_func: Callable) -> Callable:
    """Decorator to wrap and register a numpy function."""
    from .array import DimensionalArray

    if np_func.__name__ in _numpy_functions:
        return _numpy_functions[np_func.__name__]

    @wraps(np_func)
    def wrapper(*args, **kwargs):
        # Convert hd arrays to numpy arrays for input
        args = [(arg.array if isinstance(arg, DimensionalArray) else arg) for arg in args]
        kwargs = {k: (v.array if isinstance(v, DimensionalArray) else v) for k, v in kwargs.items()}

        # Call original numpy function
        result = np_func(*args, **kwargs)

        # Convert result back to hd array
        if isinstance(result, np.ndarray):
            return DimensionalArray(result)
        elif isinstance(result, tuple):
            return tuple(DimensionalArray(r) if isinstance(r, np.ndarray) else r for r in result)
        return result

    register_numpy_function(np_func.__name__, wrapper)
    return wrapper
