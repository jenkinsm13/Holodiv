"""Error registry system for tracking dimensional operations."""

from typing import Dict, Optional, NamedTuple, Tuple
import numpy as np
import weakref
import uuid
from .exceptions import RegistryError

class ErrorData:
    """Container for error data from dimensional operations."""
    def __init__(self, original_shape: Tuple[int, ...], error_tensor: np.ndarray, reduction_type: str, mask: Optional[np.ndarray] = None):
        if not original_shape:
            raise ValueError("Original shape cannot be empty")
        if error_tensor.shape != original_shape:
            raise ValueError("Error tensor shape does not match original shape")
        if reduction_type not in ('complete', 'partial'):
            raise ValueError("Invalid reduction type")
        self._original_shape = original_shape
        self._error_tensor = error_tensor
        self._reduction_type = reduction_type
        self._mask = mask

    @property
    def original_shape(self) -> Tuple[int, ...]:
        return self._original_shape

    @property
    def error_tensor(self) -> np.ndarray:
        return self._error_tensor

    @property
    def reduction_type(self) -> str:
        return self._reduction_type

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

class ErrorRegistry:
    """
    Registry for tracking error information from dimensional operations.
    Uses weak references to avoid memory leaks.
    """
    def __init__(self):
        self._errors = {}
        
    def store(self, error_data: ErrorData) -> str:
        """
        Store error information and return a unique identifier.
        
        Parameters:
            error_data: ErrorData object containing error information
            
        Returns:
            str: Unique identifier for stored error data
        """
        # Validate error data
        if not isinstance(error_data.error_tensor, np.ndarray):
            raise RegistryError("Error tensor must be a numpy array")
        if error_data.error_tensor.shape != error_data.original_shape:
            raise RegistryError("Error tensor shape does not match original shape")
        if error_data.reduction_type not in ('complete', 'partial'):
            raise RegistryError("Invalid reduction type")
        
        error_id = str(uuid.uuid4())
        self._errors[error_id] = error_data
        return error_id
        
    def retrieve(self, error_id: str) -> Optional[ErrorData]:
        """
        Retrieve error information by ID.
        
        Parameters:
            error_id: Unique identifier for error data
            
        Returns:
            Optional[ErrorData]: Error data if found, None if expired
        """
        return self._errors.get(error_id)
    
    def clear(self) -> None:
        """Clear all stored error information."""
        self._errors.clear()
        
    def __len__(self) -> int:
        """Get number of active error records."""
        return len(self._errors)
    
    def __contains__(self, error_id: str) -> bool:
        """Check if error ID exists in registry."""
        return error_id in self._errors