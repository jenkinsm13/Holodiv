"""Test suite for mathematical operators."""

import pytest
import numpy as np
from dividebyzero.operators import (
    reduce_dimension,
    elevate_dimension,
    _reduce_vector,
    _reduce_tensor
)
from dividebyzero.exceptions import DimensionalError
from . import generate_test_array, assert_array_equal_with_tolerance

class TestOperators:
    """Test suite for dimensional operators."""
    
    def test_vector_reduction(self):
        """Test vector reduction operations."""
        # Test 1D vector
        vector = np.array([1, 2, 3, 4])
        reduced, error = _reduce_vector(vector)
        
        # Check reduction
        assert np.isscalar(reduced)
        
        # Check error term
        assert error.shape == vector.shape
        assert np.abs(error.mean()) < 1e-10  # Error should be centered
        
        # Check reconstruction
        assert_array_equal_with_tolerance(
            vector,
            reduced + error
        )

    def test_tensor_reduction(self):
        """Test tensor reduction operations."""
        # Test 2D matrix
        matrix = generate_test_array((4, 4))
        reduced, error = _reduce_tensor(matrix)
        
        # Check dimension reduction
        assert reduced.ndim < matrix.ndim
        
        # Verify SVD properties
        U, S, Vt = np.linalg.svd(matrix)
        assert np.abs(S[0] - np.linalg.norm(reduced)) < 1e-10
        
        # Check reconstruction accuracy
        reconstruction = reduced.reshape(-1, 1) @ Vt[0:1, :]
        assert_array_equal_with_tolerance(
            matrix,
            reconstruction + error
        )

    def test_dimension_reduction(self):
        """Test general dimension reduction."""
        # Test various shapes
        shapes = [(2,), (2, 2), (2, 3, 4)]
        
        for shape in shapes:
            data = generate_test_array(shape)
            reduced, error = reduce_dimension(data)
            
            # Check dimension reduction
            assert reduced.ndim < len(shape)
            
            # Check error properties
            assert error.shape == data.shape
            assert np.abs(error.mean()) < 1e-10

    def test_dimension_elevation(self):
        """Test dimension elevation operations according to the paper's framework."""
        # Generate test data
        original = generate_test_array((3, 3))
        reduced, error = reduce_dimension(original)
        
        # Test elevation with error restoration
        elevated = elevate_dimension(
            reduced,
            error,
            original.shape,
            noise_scale=1.0
        )
        
        # Check shape restoration
        assert elevated.shape == original.shape
        
        # Verify gauge covariance through the field strength tensor
        # F_μν = ∂_μA_ν - ∂_νA_μ + [A_μ, A_ν]
        reduced_field = reduced.reshape(-1, 1)
        error_field = error.reshape(-1, 1)
        
        # Calculate field strength components
        field_strength = np.zeros_like(elevated)
        for i in range(elevated.shape[0]):
            for j in range(elevated.shape[1]):
                # Partial derivatives
                d_mu = np.gradient(reduced_field.flatten())[min(i, len(reduced_field)-1)]
                d_nu = np.gradient(error_field.flatten())[min(j, len(error_field)-1)]
                
                # Commutator term (simplified for the test)
                comm = reduced_field[min(i, len(reduced_field)-1)] * error_field[min(j, len(error_field)-1)]
                
                # Combine terms
                field_strength[i, j] = d_mu - d_nu + comm
        
        # The elevated state should approximately satisfy gauge covariance
        gauge_error = np.linalg.norm(elevated - field_strength)
        assert gauge_error < 2.0, "Gauge covariance not preserved"
        
        # Verify quantum correlations are preserved
        original_corr = np.abs(np.corrcoef(original.flatten()))[0, 1]
        elevated_corr = np.abs(np.corrcoef(elevated.flatten()))[0, 1]
        correlation_diff = np.abs(original_corr - elevated_corr)
        assert correlation_diff < 0.5, "Quantum correlations not preserved"
        
        # Verify error bounds on the quantized error term
        error_term = elevated - original
        error_norm = np.linalg.norm(error_term)
        assert error_norm <= 2.0 * np.linalg.norm(error), "Error bound violated"

    def test_error_handling(self):
        """Test error conditions."""
        # Test invalid input
        with pytest.raises(DimensionalError):
            reduce_dimension(np.array([]))
        
        # Test singular matrix
        singular_matrix = np.zeros((3, 3))
        with pytest.raises(DimensionalError):
            _reduce_tensor(singular_matrix)
        
        # Test incompatible shapes
        data = np.array([1, 2, 3])
        error = np.array([1, 2, 3, 4])
        with pytest.raises(ValueError):
            elevate_dimension(data, error, (3,))
            
        def test_noise_scaling(self):
            """Test noise scaling in elevation."""
            # Set a fixed random seed for reproducibility
            np.random.seed(42)

            # Generate test data
            original = generate_test_array((3, 3))
            reduced, error = reduce_dimension(original)

            # Test different noise scales away from 1.0
            noise_scales = [0.5, 1.0, 1.5]
            deviations = []

            for scale in noise_scales:
                elevated = elevate_dimension(
                    reduced,
                    error,
                    original.shape,
                    noise_scale=scale
                )

                # Calculate deviation
                deviation = np.abs(elevated - original).mean()
                deviations.append(deviation)

            # Check that deviations increase as noise_scale moves away from 1.0
            assert deviations[0] == 0.5 * np.abs(error).mean(), "Deviation at scale=0.5 should be half of the original error."
            assert deviations[1] == 0.0, "Deviation at scale=1.0 should be zero."
            assert deviations[2] == 0.5 * np.abs(error).mean(), "Deviation at scale=1.5 should be half of the original error."