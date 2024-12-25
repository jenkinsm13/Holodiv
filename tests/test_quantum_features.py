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
        
        # Verify entropy preservation
        original_entropy = quantum_state._entanglement_spectrum.entropy
        reduced_entropy = reduced._entanglement_spectrum.entropy
        assert np.abs(original_entropy - reduced_entropy) < 1e-10

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
        loop = [theta for theta in t]

        # Define a unitary connection function (e.g., rotation matrices)
        def unitary_connection(theta):
            return np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])

        # Calculate Berry phase
        phase = holonomy_calc.berry_phase(
            lambda theta: unitary_connection(theta),
            loop
        )

        # Remove NaN values
        phase = phase[~np.isnan(phase)]

        # Verify phase is real and normalized
        assert np.all(np.abs(np.imag(phase)) < 1e-10), "Phase should be real."
        assert np.all(np.abs(phase) <= np.pi), f"Phase should be within [-π, π], but got {phase}"

    def test_entanglement_preservation(self, quantum_state):
        """Test entanglement preservation during reduction."""
        # Create maximally entangled state
        bell_state = QuantumTensor(
            data=np.array([[1, 0, 0, 1]]) / np.sqrt(2),
            physical_dims=(2, 2)
        )

        # Reduce dimension
        reduced = bell_state / 0

        # Verify entanglement entropy
        expected_entropy = np.log(2)
        actual_entropy = reduced._entanglement_spectrum.entropy
        assert np.isclose(actual_entropy, expected_entropy, atol=1e-6), f"Expected entropy {expected_entropy}, got {actual_entropy}"

    def test_error_reconstruction(self, quantum_state):
        """Test error reconstruction capabilities."""
        # Reduce dimension
        reduced = quantum_state / 0

        # Reconstruct with full error restoration
        reconstructed = reduced.elevate(
            noise_scale=1.0  # Fully restore the error
        )

        # Verify reconstruction error bounds
        error = np.linalg.norm(reconstructed.data - quantum_state.data)
        assert error < 1e-6, f"Reconstruction error {error} exceeds threshold."