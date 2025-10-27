import numpy as np
import pytest
from holodiv.quantum.holonomy import HolonomyCalculator
from holodiv.quantum.gauge_groups import SU2Group

@pytest.fixture
def holonomy_calculator():
    """Returns a HolonomyCalculator with an SU(2) group."""
    return HolonomyCalculator(SU2Group())

def test_wilson_loop(holonomy_calculator):
    """Test the wilson_loop method."""
    connection = np.random.rand(2, 2, 2)
    loop = [(0., 0.), (1., 0.), (1., 1.), (0., 1.), (0., 0.)]

    # Test adaptive method
    wilson_loop_adaptive = holonomy_calculator.wilson_loop(connection, loop, method='adaptive')
    assert wilson_loop_adaptive.shape == (2, 2)

    # Test fixed method
    wilson_loop_fixed = holonomy_calculator.wilson_loop(connection, loop, method='fixed')
    assert wilson_loop_fixed.shape == (2, 2)

def test_berry_phase():
    """Test the berry_phase method."""
    # Simple Hamiltonian for a spin-1/2 particle in a magnetic field
    def hamiltonian(params):
        B_x, B_y, B_z = params
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        return B_x * sigma_x + B_y * sigma_y + B_z * sigma_z

    # Loop in parameter space
    loop = [(np.sin(t), np.cos(t), 0.1) for t in np.linspace(0, 2 * np.pi, 50)]

    calculator = HolonomyCalculator(SU2Group())
    phase = calculator.berry_phase(hamiltonian, loop)
    # The phase should be close to +/- 1, depending on the path
    assert np.isclose(np.abs(phase), 1.0, atol=0.1)

def test_compute_chern_number(holonomy_calculator):
    """Test the compute_chern_number method."""
    # A simple Berry curvature function (constant for simplicity)
    def berry_curvature(kx, ky):
        return 1.0

    # A square surface in k-space
    surface = [(0, 0), (2 * np.pi, 0), (2 * np.pi, 2 * np.pi), (0, 2 * np.pi)]

    chern_number = holonomy_calculator.compute_chern_number(berry_curvature, surface)
    assert chern_number == 6

def test_parallel_propagator(holonomy_calculator):
    """Test the parallel_propagator method."""
    connection = np.random.rand(2, 2, 2)
    path = [(0., 0.), (1., 0.), (1., 1.)]

    # Without reference frame
    propagator = holonomy_calculator.parallel_propagator(connection, path)
    assert propagator.shape == (2, 2)

    # With reference frame
    ref_frame = np.random.rand(2, 2)
    propagator_with_ref = holonomy_calculator.parallel_propagator(connection, path, reference_frame=ref_frame)
    assert propagator_with_ref.shape == (2, 2)
    assert not np.allclose(propagator, propagator_with_ref)
