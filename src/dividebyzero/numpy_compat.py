"""
Numpy compatibility layer for dividebyzero.
Allows dbz to act as a complete drop-in replacement for numpy.
"""

import numpy as np
from functools import wraps
import inspect
from .numpy_registry import wrap_and_register_numpy_function, get_numpy_function

# Register numpy functions
for name in dir(np):
    if name.startswith('_'):
        continue
    
    obj = getattr(np, name)
    # Only wrap actual functions, not classes, modules, or other objects
    if callable(obj) and not inspect.isclass(obj) and not inspect.ismodule(obj) and hasattr(obj, '__name__'):
        try:
            wrap_and_register_numpy_function(obj)
        except (AttributeError, TypeError):
            # Skip objects that can't be wrapped
            continue

# Export numpy functions
__all__ = [name for name in dir(np) if not name.startswith('_')] 