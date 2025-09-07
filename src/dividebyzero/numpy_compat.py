"""NumPy compatibility layer for DivideByZero.

This module exposes the full NumPy API but transparently converts
``DimensionalArray`` arguments to plain ``numpy.ndarray`` objects and back.
It is primarily used internally when importing ``dividebyzero`` as a
drop-in replacement for NumPy, but it can also be accessed explicitly via
``dbz.numpy_compat``.
"""

from __future__ import annotations

import inspect
import types
import numpy as np

from .numpy_registry import wrap_and_register_numpy_function


def _wrap_module(np_module: types.ModuleType, visited: dict[types.ModuleType, types.ModuleType] | None = None) -> types.ModuleType:
    """Recursively wrap a NumPy module.

    Callables are wrapped so they understand ``DimensionalArray`` inputs
    and outputs. Submodules are processed recursively so that functions
    like ``numpy.linalg.pinv`` are available and behave correctly.
    """

    if visited is None:
        visited = {}
    if np_module in visited:
        return visited[np_module]

    module = types.ModuleType(np_module.__name__)
    visited[np_module] = module

    names = [name for name in dir(np_module) if not name.startswith("_")]

    # Register callables and other objects first
    for name in names:
        obj = getattr(np_module, name)

        if callable(obj) and not inspect.isclass(obj) and not inspect.ismodule(obj) and hasattr(obj, "__name__"):
            try:
                setattr(module, name, wrap_and_register_numpy_function(obj))
            except (AttributeError, TypeError):
                setattr(module, name, obj)
        elif not inspect.ismodule(obj):
            setattr(module, name, obj)

    # Process submodules afterwards so they don't override top-level registrations
    for name in names:
        obj = getattr(np_module, name)
        if inspect.ismodule(obj):
            setattr(module, name, _wrap_module(obj, visited))

    module.__all__ = [name for name in dir(module) if not name.startswith("_")]
    return module


# Populate the module namespace with the wrapped NumPy API
_wrapped_np = _wrap_module(np)
for name in _wrapped_np.__all__:
    globals()[name] = getattr(_wrapped_np, name)

# Export all public names we just populated
__all__ = [name for name in globals() if not name.startswith("_")]

