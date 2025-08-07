"""
NumPy compatibility layer for DivideByZero.

Access this module via ``dbz.numpy_compat`` after ``import dividebyzero as
dbz``.  It lets DivideByZero act as a drop-in replacement for NumPy, but most
code can simply use the top-level ``dbz`` API directly.
"""

import numpy as np
import inspect
from .numpy_registry import wrap_and_register_numpy_function

# Register numpy functions and constants in the module namespace
for name in dir(np):
    if name.startswith("_"):
        continue

    obj = getattr(np, name)

    if callable(obj) and not inspect.isclass(obj) and not inspect.ismodule(obj) and hasattr(obj, "__name__"):
        try:
            # Wrap numpy functions so they understand DimensionalArray inputs
            globals()[name] = wrap_and_register_numpy_function(obj)
        except (AttributeError, TypeError):
            # Some numpy objects (like ufuncs without names) cannot be wrapped
            continue
    else:
        # Expose non-callable objects (constants, etc.) directly
        globals()[name] = obj

# Export all public names we just populated
__all__ = [name for name in globals() if not name.startswith("_")]

