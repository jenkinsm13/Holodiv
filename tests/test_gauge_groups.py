import numpy as np
import pytest
from holodiv.quantum.gauge_groups import SU2Group, SU3Group, U1Group

def test_su2_group():
    """Test the SU2Group class."""
    su2 = SU2Group()
    assert su2.dimension == 2
    assert su2.generators.shape == (3, 2, 2)
    assert np.allclose(su2.structure_constants[0, 1, 2], 1)

    # Test symmetric state
    state = su2.create_symmetric_state()
    assert state.shape == (2, 2)
    assert np.allclose(state, state.conj().T) # Hermitian

    # Test Wilson line
    connection = np.random.rand(2, 2, 2)
    path = [(0, 0), (0, 1)]
    wilson_line = su2.compute_wilson_line(connection, path)
    assert wilson_line.shape == (2, 2)

def test_su3_group():
    """Test the SU3Group class."""
    su3 = SU3Group()
    assert su3.dimension == 3
    assert su3.generators.shape == (8, 3, 3)
    # Check if generators are traceless
    assert np.allclose(np.trace(su3.generators, axis1=1, axis2=2), 0)

    # Test symmetric state
    state = su3.create_symmetric_state()
    assert state.shape == (3, 3)
    assert np.allclose(state, state.conj().T)

    # Test Chern number
    field_strength = np.random.rand(3, 3, 3, 3)
    volume_element = np.zeros((3, 3, 3, 3))
    volume_element[0, 1, 0, 1] = 1
    volume_element[1, 0, 1, 0] = -1
    chern_number = su3.compute_chern_number(field_strength, volume_element)
    assert isinstance(chern_number, float)

def test_u1_group():
    """Test the U1Group class."""
    u1 = U1Group()
    assert u1.dimension == 1
    assert u1.generator.shape == (1, 1)
    assert np.allclose(u1.structure_constants, 0)

    # Test symmetric state
    state = u1.create_symmetric_state()
    assert state.shape == (1, 1)
    assert np.allclose(state, state.conj().T)

    # Test magnetic charge
    field_strength = np.random.rand(3, 3, 3)
    surface = np.random.rand(3, 3)
    charge = u1.magnetic_charge(field_strength, surface)
    assert isinstance(charge, float)
