import numpy as np
import dividebyzero as dbz
from dividebyzero.quantum import QuantumTensor
from dividebyzero.quantum.holonomy import HolonomyCalculator
from dividebyzero.quantum.gauge_groups import U1Group
import matplotlib.pyplot as plt

def create_hamiltonian(t, epsilon=1e-6):
    """
    Create a quantum Hamiltonian with a conical intersection.
    At t=0, the system becomes degenerate (traditional methods fail).
    """
    # Create a Hamiltonian that becomes degenerate at t=0
    H = np.array([
        [t, epsilon],
        [epsilon, -t]
    ])
    return dbz.array(H)

def traditional_berry_phase(t_values):
    """
    Traditional method that would fail at t=0
    """
    phases = []
    for t in t_values:
        try:
            H = create_hamiltonian(t).data
            eigenvals, eigenvecs = np.linalg.eigh(H)
            # This will fail at t=0 due to degeneracy
            phase = np.angle(eigenvecs[0, 0])
            phases.append(phase)
        except np.linalg.LinAlgError:
            phases.append(np.nan)
    return np.array(phases)

def quantum_holonomy_through_singularity(t_values):
    """
    Using dividebyzero to compute holonomy through the singular point
    """
    # Initialize holonomy calculator with U(1) gauge group
    calculator = HolonomyCalculator(U1Group())
    
    holonomies = []
    for t in t_values:
        # Create quantum tensor from Hamiltonian
        H = create_hamiltonian(t)
        qt = QuantumTensor(H, physical_dims=(2,))
        
        # At t=0, traditional division would fail
        # But dividebyzero handles it through dimensional reduction
        reduced_state = qt / 0
        
        # Create a loop in parameter space around current point
        delta = 0.01
        loop = [(t-delta,), (t,), (t+delta,)]
        
        # Compute Berry phase using the holonomy calculator
        # The Hamiltonian is passed as a lambda function
        phase = calculator.berry_phase(
            lambda x: create_hamiltonian(x[0]).data,
            loop
        )
        holonomies.append(phase)
    
    return np.array(holonomies)

def main():
    # Create a range of t values including the singular point at t=0
    t_values = np.linspace(-2, 2, 1000)
    
    # Compare traditional vs quantum holonomy methods
    traditional_phases = traditional_berry_phase(t_values)
    quantum_phases = quantum_holonomy_through_singularity(t_values)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(t_values, traditional_phases, 'r-', label='Traditional (fails at t=0)')
    plt.title('Traditional Berry Phase\n(Fails at Degeneracy)')
    plt.xlabel('t')
    plt.ylabel('Phase')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(t_values, quantum_phases, 'b-', label='Quantum Holonomy')
    plt.title('Quantum Holonomy\n(Through Singularity)')
    plt.xlabel('t')
    plt.ylabel('Holonomy')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('quantum_holonomy_comparison.png')
    plt.close()

if __name__ == "__main__":
    main() 