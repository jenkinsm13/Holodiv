"""Mathematical operators for dimensional calculations."""

import numpy as np
from typing import Tuple, Optional, Dict
from .exceptions import DimensionalError
import logging
from scipy.linalg import expm
from .quantum.tensor import EntanglementSpectrum, QuantumTensor

# Instead, create a module-level logger
logger = logging.getLogger(__name__)

def reduce_dimension(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce dimension of input data."""
    if data.size == 0:
        raise DimensionalError("Cannot reduce dimension of an empty array.")
    if data.ndim == 0:
        raise DimensionalError("Cannot reduce dimension of a scalar.")
    elif data.ndim == 1:
        reduced, error = _reduce_vector(data)
    else:
        reduced, error = _reduce_tensor(data)
    
    # Ensure error has the same shape as the input data
    error = error.reshape(data.shape)
    
    # Center the error around zero
    error -= error.mean()
    
    return reduced, error

def _reduce_vector(data: np.ndarray) -> Tuple[float, np.ndarray]:
    """Reduce a vector to a scalar."""
    magnitude = np.abs(data).mean()
    error = data - magnitude
    return magnitude, error

def _reduce_tensor(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce tensor dimension."""
    if matrix.ndim <= 1:
        return _reduce_vector(matrix)
    
    original_shape = matrix.shape
    matrix_2d = matrix.reshape(-1, matrix.shape[-1])
    
    try:
        u, s, vh = np.linalg.svd(matrix_2d, full_matrices=False)
        if np.isclose(s[0], 0):
            raise DimensionalError("Cannot reduce dimension of a singular matrix.")
        reduced = u[:, 0] * s[0]
        reconstructed = np.outer(u[:, 0], vh[0, :]) * s[0]
        error = matrix - reconstructed.reshape(original_shape)
    except np.linalg.LinAlgError:
        raise DimensionalError("Cannot reduce dimension of a singular matrix.")
    
    return reduced.reshape(original_shape[:-1]), error

def elevate_dimension(reduced_data: np.ndarray,
                    error: np.ndarray,
                    target_shape: Tuple[int, ...],
                    noise_scale: float = 1e-6) -> np.ndarray:
    """
    Implement tensor network elevation as defined in Section 4 of the paper.
    Uses quantum state reconstruction with entanglement preservation.
    """
    # Validate input shapes
    if not isinstance(reduced_data, np.ndarray) or not isinstance(error, np.ndarray):
        raise ValueError("Both reduced_data and error must be numpy arrays")
    
    if np.prod(reduced_data.shape) > np.prod(target_shape):
        raise ValueError("Cannot elevate to lower dimensions")
    
    # Validate error matrix shape
    error_flat = error.flatten()
    error_size = int(np.sqrt(len(error_flat)))
    if error_size * error_size != len(error_flat):
        raise ValueError("Error matrix must be square when flattened")
    
    # Validate compatibility between reduced_data and error
    if len(reduced_data.flatten()) * error_size != np.prod(target_shape):
        raise ValueError("Incompatible shapes between reduced data and error matrix")
    
    # Normalize input data and convert to state vector
    reduced_flat = reduced_data.flatten()
    reduced_flat = reduced_flat / np.linalg.norm(reduced_flat)
    
    # Create density matrix from reduced state
    rho = np.outer(reduced_flat, reduced_flat.conj())
    
    # Create error operator from error matrix
    error_matrix = error_flat.reshape(error_size, error_size)
    
    # Ensure error matrix is square and matches target dimensions
    if error_matrix.shape[0] != np.prod(target_shape):
        # Pad or truncate error matrix
        new_error = np.zeros((np.prod(target_shape), np.prod(target_shape)))
        min_size = min(error_matrix.shape[0], new_error.shape[0])
        new_error[:min_size, :min_size] = error_matrix[:min_size, :min_size]
        error_matrix = new_error
    
    # Create unitary error operator
    error_op = expm(1j * error_matrix)
    
    # Pad density matrix if needed
    if rho.shape[0] < error_op.shape[0]:
        padded_rho = np.zeros_like(error_op)
        padded_rho[:rho.shape[0], :rho.shape[1]] = rho
        rho = padded_rho
    
    # Apply error operator to density matrix
    rho_error = error_op @ rho @ error_op.conj().T
    
    # Ensure Hermiticity and positive semi-definiteness
    rho_error = (rho_error + rho_error.conj().T) / 2
    eigenvals, eigenvecs = np.linalg.eigh(rho_error)
    eigenvals = np.maximum(eigenvals, 0)  # Ensure positivity
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Add controlled quantum noise to maintain coherence
    noise = np.random.normal(0, noise_scale, len(eigenvals))
    eigenvals = eigenvals + noise
    eigenvals = np.maximum(eigenvals, 0)  # Ensure positivity after noise
    eigenvals = eigenvals / np.sum(eigenvals)  # Normalize
    
    # Reconstruct elevated state
    elevated = np.zeros(np.prod(target_shape), dtype=np.complex128)
    for i in range(len(eigenvals)):
        elevated += np.sqrt(eigenvals[i]) * eigenvecs[:, i]
    
    # Ensure proper normalization
    elevated = elevated / np.linalg.norm(elevated)
    
    # Reshape to target shape
    return elevated.reshape(target_shape)

def _elevate_tensor(reduced: np.ndarray, error: np.ndarray, target_shape: Tuple[int, ...], noise_scale: float = 1e-6) -> np.ndarray:
    """Reconstruct the original matrix from reduced matrix and error tensor."""
    if reduced.ndim != 2 or error.ndim != 2:
        raise ValueError("Both reduced and error tensors must be 2D for matrix reconstruction.")
    if target_shape != reduced.shape:
        raise ValueError("Target shape must match the reduced matrix shape for matrices.")
    
    # Add scaled error tensor to create a noisy reconstruction
    noise = noise_scale * np.random.randn(*error.shape)
    reconstructed = reduced + error + noise
    return reconstructed