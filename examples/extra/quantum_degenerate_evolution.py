import numpy as np
from dividebyzero import QuantumTensor
from dividebyzero.quantum.holonomy import HolonomyCalculator
from dividebyzero.quantum.gauge_groups import U1Group
import matplotlib.pyplot as plt

def create_hamiltonian(t, epsilon=1e-6):
    """Create a 2x2 Hamiltonian with a conical intersection at t=0."""
    # Handle tuple input by taking first element
    if isinstance(t, tuple):
        t = t[0]
    H = np.array([
        [t, epsilon],
        [epsilon, -t]
    ])
    return QuantumTensor(H)

def traditional_evolution(t_values, psi0):
    """Evolve the state using traditional numerical integration."""
    states = []
    current_state = psi0
    dt = t_values[1] - t_values[0]
    
    for t in t_values:
        H = create_hamiltonian(t)
        # Simple Euler integration
        U = np.eye(2) - 1j * H.data * dt
        current_state = QuantumTensor(U @ current_state.data)
        states.append(current_state)
    
    return states

def quantum_holonomy_evolution(t_values, psi0, epsilon=1e-6):
    """Evolve the state using quantum holonomy through the degenerate point."""
    states = []
    current_state = psi0
    dt = t_values[1] - t_values[0]
    
    # Initialize holonomy calculator with U(1) gauge group
    gauge_group = U1Group()
    holonomy_calc = HolonomyCalculator(gauge_group)
    
    for t in t_values:
        # Create Hamiltonian at current time
        H = create_hamiltonian(t)
        
        # Compute holonomy-based evolution
        # Create a small loop around current time point
        loop = [t, t + dt/2, t + dt, t + dt/2]
        phase = holonomy_calc.berry_phase(
            lambda s: create_hamiltonian(s).data,
            loop
        )
        
        # Construct evolution operator
        U = np.array([
            [np.exp(1j * phase), epsilon],
            [epsilon, np.exp(-1j * phase)]
        ])
        
        # Ensure current_state.data is a column vector
        if len(current_state.data.shape) == 1:
            current_state.data = current_state.data.reshape(-1, 1)
            
        # Apply evolution
        evolved_data = U @ current_state.data
        current_state = QuantumTensor(evolved_data)
        states.append(current_state)
    
    return states

def compute_survival_probability(states):
    """Compute survival probability relative to initial state."""
    initial_state = states[0]
    probabilities = []
    
    for state in states:
        overlap = np.abs(np.vdot(initial_state.data.flatten(), state.data.flatten()))**2
        probabilities.append(overlap)
    
    return probabilities

def main():
    # Set up parameters
    t_values = np.linspace(-1, 1, 50)
    psi0 = QuantumTensor(np.array([1/np.sqrt(2), 1/np.sqrt(2)]))  # Initial superposition state
    
    # Evolve using both methods
    traditional_states = traditional_evolution(t_values, psi0)
    holonomy_states = quantum_holonomy_evolution(t_values, psi0)
    
    # Compute survival probabilities
    traditional_probs = compute_survival_probability(traditional_states)
    holonomy_probs = compute_survival_probability(holonomy_states)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, traditional_probs, 'b-', label='Traditional Evolution')
    plt.plot(t_values, holonomy_probs, 'r--', label='Holonomy Evolution')
    plt.axvline(x=0, color='k', linestyle=':', label='Degenerate Point')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title('Quantum Evolution Through Degenerate Point')
    plt.legend()
    plt.grid(True)
    plt.savefig('quantum_evolution_comparison.png')
    plt.close()

if __name__ == '__main__':
    main() 