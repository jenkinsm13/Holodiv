"""
DivideByZero: A NumPy extension implementing division by zero as dimensional reduction.
"""

from .array import array, DimensionalArray
from .registry import ErrorRegistry
from . import operators

__version__ = "0.1.0"
__all__ = ['array', 'DimensionalArray', 'ErrorRegistry']

# Global error registry
_ERROR_REGISTRY = ErrorRegistry()

def get_registry():
    """Get the global error registry"""
    return _ERROR_REGISTRY

def zeros(shape, dtype=None):
    """Create array of zeros"""
    import numpy as np
    return array(np.zeros(shape, dtype=dtype))

def ones(shape, dtype=None):
    """Create array of ones"""
    import numpy as np
    return array(np.ones(shape, dtype=dtype))

def empty(shape, dtype=None):
    """Create empty array"""
    import numpy as np
    return array(np.empty(shape, dtype=dtype))