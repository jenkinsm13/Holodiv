"""
Dividebyzero: A framework for handling singularities in numerical computations.
"""

import numpy as np
from .array import DimensionalArray
from .registry import ErrorRegistry, ErrorData
from .exceptions import DimensionalError, ReconstructionError
from . import numpy_compat  # This will register all numpy functions
from .numpy_registry import _numpy_functions  # Import registered functions
from . import quantum
from . import linalg
from . import random
from .linalg import logm

# Create a module-level registry
_REGISTRY = ErrorRegistry()

# Add mathematical constants
pi = np.pi
e = np.e
inf = np.inf
nan = np.nan

# Add newaxis attribute
newaxis = None

# Add commonly used functions
zeros = lambda *args, **kwargs: array(np.zeros(*args, **kwargs))
ones = lambda *args, **kwargs: array(np.ones(*args, **kwargs))
eye = lambda *args, **kwargs: array(np.eye(*args, **kwargs))
zeros_like = lambda *args, **kwargs: array(np.zeros_like(*args, **kwargs))
ones_like = lambda *args, **kwargs: array(np.ones_like(*args, **kwargs))

# Add commonly used mathematical functions
abs = lambda x: array(np.abs(x.array if isinstance(x, DimensionalArray) else x))
sin = lambda x: array(np.sin(x.array if isinstance(x, DimensionalArray) else x))
cos = lambda x: array(np.cos(x.array if isinstance(x, DimensionalArray) else x))
tan = lambda x: array(np.tan(x.array if isinstance(x, DimensionalArray) else x))
exp = lambda x: array(np.exp(x.array if isinstance(x, DimensionalArray) else x))
log = lambda x: array(np.log(x.array if isinstance(x, DimensionalArray) else x))
sqrt = lambda x: array(np.sqrt(x.array if isinstance(x, DimensionalArray) else x))
linspace = lambda *args, **kwargs: array(np.linspace(*args, **kwargs))
arange = lambda *args, **kwargs: array(np.arange(*args, **kwargs))

def get_registry():
    """Get the global error registry."""
    return _REGISTRY

def array(array_like, dtype=None):
    """Create a DimensionalArray."""
    return DimensionalArray(array_like, error_registry=_REGISTRY, dtype=dtype)

# Add all registered numpy functions to the module namespace
globals().update(_numpy_functions)

# Build __all__ list
base_exports = [
    'array', 'zeros', 'ones', 'eye', 'zeros_like', 'ones_like',
    'DimensionalArray', 'ErrorRegistry', 'ErrorData',
    'DimensionalError', 'ReconstructionError',
    'get_registry', 'quantum', 'linalg', 'random',
    'pi', 'e', 'inf', 'nan', 'newaxis',
    'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'logm',
    'linspace', 'arange', 'abs'
]

__all__ = base_exports + [name for name in _numpy_functions if name not in base_exports]