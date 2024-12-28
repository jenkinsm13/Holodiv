import numpy as np
import matplotlib.pyplot as plt
import dividebyzero as dbz
from dividebyzero.quantum import QuantumTensor
from dividebyzero.quantum.holonomy import HolonomyCalculator
from dividebyzero.quantum.gauge_groups import U1Group
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def create_hamiltonian(t, epsilon):
    """Create a 2x2 Hamiltonian with a conical intersection at t=0."""
    if isinstance(t, tuple):
        t = t[0]
    H = dbz.array([
        [t, epsilon * (1 + 1j*t)],
        [epsilon * (1 - 1j*t), -t]
    ])
    return H

def simulate_quantum_singularity(initial_state, epsilon, t_range):
    quantum_tensor = QuantumTensor(initial_state)

    # Time evolution parameters
    t_values = np.linspace(t_range[0], t_range[1], 400)
    dt = t_values[1] - t_values[0]
    
    # Initialize holonomy calculator with U(1) gauge group
    gauge_group = U1Group()
    holonomy_calculator = HolonomyCalculator(gauge_group)

    # Simulate time evolution
    evolved_states = [quantum_tensor]
    holonomies = []

    for t in t_values[1:]:  # Start from the second time step
        H = create_hamiltonian(t, epsilon)
        
        # Evolve the quantum state using RK4 method
        k1 = -1j * H.array @ evolved_states[-1].data
        k2 = -1j * H.array @ (evolved_states[-1].data + 0.5 * dt * k1)
        k3 = -1j * H.array @ (evolved_states[-1].data + 0.5 * dt * k2)
        k4 = -1j * H.array @ (evolved_states[-1].data + dt * k3)
        
        new_state = evolved_states[-1].data + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        new_state /= np.linalg.norm(new_state)
        
        evolved_states.append(QuantumTensor(new_state))

        loop = np.linspace(t - 0.1, t + 0.1, 20)
        holonomy = holonomy_calculator.berry_phase(lambda x: create_hamiltonian(x, epsilon), loop)
        holonomies.append(np.real(holonomy))

    return t_values, evolved_states, holonomies

def run_multiple_simulations():
    # Define different initial states
    initial_states = [
        dbz.array([1, 0]) / np.sqrt(2),  # |+⟩ state
        dbz.array([0, 1]) / np.sqrt(2),  # |-⟩ state
        dbz.array([1, 1]) / np.sqrt(2),  # |+⟩ + |-⟩ state
    ]
    
    # Define different epsilon values
    epsilons = [1e-3, 1e-2, 1e-1]
    
    # Define time ranges
    t_ranges = [(-1, 1), (-5, 5)]
    
    results = []
    
    for initial_state in initial_states:
        for epsilon in epsilons:
            for t_range in t_ranges:
                result = simulate_quantum_singularity(initial_state, epsilon, t_range)
                results.append({
                    'initial_state': initial_state,
                    'epsilon': epsilon,
                    't_range': t_range,
                    'result': result
                })
    
    return results

def plot_results(results):
    for i, res in enumerate(results):
        t_values, evolved_states, holonomies = res['result']
        initial_state = res['initial_state']
        epsilon = res['epsilon']
        t_range = res['t_range']
        
        plt.figure(figsize=(12, 8))

        # Plot state evolution
        plt.subplot(2, 1, 1)
        state_evolutions = np.array([state.data.flatten() for state in evolved_states])
        for j in range(2):
            plt.plot(t_values, np.abs(state_evolutions[:, j])**2, label=f'|ψ{j}|²')
        plt.title(f'Quantum State Evolution (ε={epsilon}, t∈{t_range})')
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.legend()

        # Plot holonomy
        plt.subplot(2, 1, 2)
        plt.plot(t_values[1:], holonomies)
        plt.title('Berry Phase Evolution')
        plt.xlabel('Time')
        plt.ylabel('Berry Phase')

        plt.tight_layout()
        plt.savefig(f'quantum_singularity_simulation_{i}.png')
        plt.close()

if __name__ == "__main__":
    results = run_multiple_simulations()
    plot_results(results)
    print(f"Simulation complete. Results saved in 'quantum_singularity_simulation_X.png' files.")