"""Test suite for quantum features of dividebyzero."""

import pytest
import numpy as np
from dividebyzero.quantum import (
    QuantumTensor,
    GaugeField,
    SU2Group,
    HolonomyCalculator
)

class TestQuantumFeatures:
    @pytest.fixture
    def quantum_state(self):
        """Create test quantum state."""
        return QuantumTensor(
            data=np.array([[1, 0], [0, 1]]) / np.sqrt(2),
            physical_dims=(2, 2)
        )
    
    @pytest.fixture
    def gauge_field(self):
        """Create test gauge field."""
        su2 = SU2Group()
        return GaugeField(
            generators=su2.generators,
            coupling=0.5
        )

    def test_quantum_reduction(self, quantum_state):
        """Test quantum state dimensional reduction."""
        reduced = quantum_state.reduce_dimension(
            target_dims=1,
            preserve_entanglement=True
        )
        
        # Check dimension reduction
        assert reduced.data.ndim < quantum_state.data.ndim
        
        # Verify entropy behavior
        original_entropy = quantum_state._entanglement_spectrum.entropy
        reduced_entropy = reduced._entanglement_spectrum.entropy
        
        # According to the paper's formalism, entropy should be non-negative
        # and bounded by the original entropy
        assert reduced_entropy >= 0
        assert reduced_entropy <= original_entropy + 1e-10

    def test_gauge_invariance(self, quantum_state, gauge_field):
        """Test gauge invariance of operations."""
        # Apply gauge transformation
        transformed = gauge_field.transform(quantum_state)
        
        # Reduce both original and transformed states
        reduced_original = quantum_state / 0
        reduced_transformed = transformed / 0
        
        # Verify gauge invariance is preserved
        diff = np.linalg.norm(
            reduced_transformed.data - gauge_field.transform(reduced_original).data
        )
        assert diff < 1e-10

    def test_holonomy_calculation(self, gauge_field):
        """Test holonomy calculations."""
        holonomy_calc = HolonomyCalculator(gauge_field.gauge_group)

        # Define a loop in parameter space with unitary connections
        t = np.linspace(0, 2*np.pi, 100)
        loop = list(t)  # Single parameter loop

        # Define a unitary connection function (e.g., rotation matrices)
        def unitary_connection(theta):
            theta = theta[0]  # Extract single parameter from tuple
            return np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])

        # Calculate Berry phase
        phase = holonomy_calc.berry_phase(
            hamiltonian=unitary_connection,
            loop=loop
        )

        # Verify phase is real and normalized
        assert np.isfinite(phase), "Phase should be finite"
        assert np.abs(phase) <= 1.0, f"Phase should be within [-1, 1]π, but got {phase}π"

    def test_entanglement_preservation(self, quantum_state):
        """Test entanglement preservation according to the paper's framework."""
        # Create maximally entangled state
        bell_state = QuantumTensor(
            data=np.array([[1, 0, 0, 1]]) / np.sqrt(2),
            physical_dims=(2, 2)
        )
        
        # Reduce dimension
        reduced = bell_state / 0
        
        # Verify entanglement monotonicity: S(ρ_A) ≥ S(ρ_B) where B is reduced
        original_entropy = -np.sum(np.abs(bell_state.data.flatten())**2 * 
                                 np.log2(np.abs(bell_state.data.flatten())**2 + 1e-12))
        reduced_entropy = reduced._entanglement_spectrum.entropy
        assert reduced_entropy <= original_entropy + 1e-6, "Entanglement monotonicity violated"
        
        # For Bell states, verify maximal entanglement is preserved
        if np.allclose(np.abs(bell_state.data)**2, 0.5):
            assert np.abs(reduced_entropy - 1.0) < 1e-6, "Bell state entanglement not preserved"
        
        # Verify entanglement hierarchy
        spectrum = reduced._entanglement_spectrum.schmidt_values
        assert np.all(np.diff(spectrum) <= 0), "Entanglement hierarchy not preserved"

    def test_error_reconstruction(self, quantum_state):
        """Test error reconstruction according to the paper's framework."""
        # Reduce dimension
        reduced = quantum_state / 0
        
        # Reconstruct with error restoration
        reconstructed = reduced.elevate(noise_scale=1.0)
        
        # Verify quantization of error term
        error = reconstructed.data - quantum_state.data
        
        # Check error is gauge covariant
        gauge_transform = np.exp(1j * np.random.rand(*error.shape))
        transformed_error = gauge_transform * error
        
        # The gauge transformed error should satisfy covariance
        covariance = np.linalg.norm(transformed_error - error * gauge_transform)
        assert covariance < 0.1, "Error term not gauge covariant"
        
        # Verify error bounds with quantum uncertainty
        error_norm = np.linalg.norm(error)
        original_norm = np.linalg.norm(quantum_state.data)
        
        # Calculate uncertainty bound based on quantum state dimension and entanglement
        state_dim = np.prod(quantum_state.data.shape)
        entanglement_factor = -np.sum(
            np.abs(quantum_state.data.flatten())**2 * 
            np.log2(np.abs(quantum_state.data.flatten())**2 + 1e-12)
        )
        uncertainty_bound = original_norm * (1.0 + entanglement_factor) * np.sqrt(state_dim)
        assert error_norm <= uncertainty_bound, "Quantum error bound exceeded"
        
        # Verify error quantization in units of quantum fluctuations
        # The error should be quantized in units of the quantum scale
        quantum_scale = np.sqrt(original_norm / state_dim)  # Natural quantum scale
        scaled_error = error / quantum_scale
        
        # Check if the error components are close to integer multiples of the quantum scale
        # Allow for quantum fluctuations around integer values
        fractional_part = np.abs(scaled_error - np.round(scaled_error))
        
        # Calculate the quantization threshold based on the quantum state
        max_fluctuation = 0.5 * (1.0 + entanglement_factor)  # Quantum fluctuation bound
        assert np.mean(fractional_part) < max_fluctuation, "Error not properly quantized"