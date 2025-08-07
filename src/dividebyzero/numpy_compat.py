"""
Numpy compatibility layer for dividebyzero.
Allows dbz to act as a complete drop-in replacement for numpy.
"""

import numpy as np
import inspect

from .numpy_registry import wrap_and_register_numpy_function


# Keep track of names we intentionally export
_exported_names = []

# Register numpy functions and constants in the module namespace
for name in dir(np):
    if name.startswith("_"):
        continue

    obj = getattr(np, name)

    if (
        callable(obj)
        and not inspect.isclass(obj)
        and not inspect.ismodule(obj)
        and hasattr(obj, "__name__")
    ):
        try:
            # Wrap numpy functions so they understand DimensionalArray inputs
            globals()[name] = wrap_and_register_numpy_function(obj)
            _exported_names.append(name)
        except (AttributeError, TypeError):
            # Some numpy objects (like ufuncs without names) cannot be wrapped
            continue
    else:
        # Expose non-callable objects (constants, etc.) directly
        globals()[name] = obj
        _exported_names.append(name)

# Export only the numpy names we populated above
# Using a tuple prevents accidental mutation after cleanup
__all__ = tuple(_exported_names)

# Prevent leaking internal helpers
del np, inspect, wrap_and_register_numpy_function, _exported_names

