"""
Quantum Extensions for DivideByZero Framework

This module implements quantum mechanical extensions to the dimensional division
framework, providing tensor network operations and gauge-invariant computations.

Key Components:
    - QuantumTensor: Quantum-aware tensor operations
    - GaugeField: Gauge field implementations
    - HolographicMapping: AdS/CFT inspired dimensional mappings
"""

from .tensor import (
    QuantumTensor,
    TensorNetwork,
    EntanglementSpectrum,
    reduce_entanglement,
)
from .gauge import (
    GaugeField,
    GaugeTransformation,
    parallel_transport,
    compute_holonomy,
)
from .gauge_groups import SU2Group, SU3Group
from .holonomy import HolonomyCalculator

# Make QuantumTensor directly available as dbz.quantum.QuantumTensor
QuantumTensor = QuantumTensor

__all__ = [
    'QuantumTensor',
    'TensorNetwork',
    'EntanglementSpectrum',
    'reduce_entanglement',
    'GaugeField',
    'GaugeTransformation',
    'parallel_transport',
    'compute_holonomy',
    'SU2Group',
    'SU3Group',
    'HolonomyCalculator',
]

# Physical constants
PLANCK_LENGTH = 1.616255e-35  # meters
PLANCK_TIME = 5.391247e-44    # seconds
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K

# Quantum configuration
DEFAULT_BOND_DIMENSION = 16
ENTANGLEMENT_CUTOFF = 1e-10
MAX_TRUNCATION_ERROR = 1e-12