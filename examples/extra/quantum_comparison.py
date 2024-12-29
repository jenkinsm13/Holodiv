import dividebyzero as dbz
import qutip as qt
import matplotlib.pyplot as plt
from dividebyzero.quantum import QuantumTensor
from dividebyzero.quantum.holonomy import HolonomyCalculator
from dividebyzero.quantum.gauge_groups import U1Group

EPSILON = 1e-6  # Global epsilon value for consistency

def create_qutip_hamiltonian(t):
    """Create the Hamiltonian using QuTiP."""
    if isinstance(t, tuple):
        t = t[0]
    H = dbz.array([[t, EPSILON], [EPSILON, -t]])
    return qt.Qobj(H)

def create_dividebyzero_hamiltonian(t):
    """Create the Hamiltonian using dividebyzero."""
    if isinstance(t, tuple):
        t = t[0]
    H = dbz.array([[t, EPSILON], [EPSILON, -t]])
    return QuantumTensor(H)

def qutip_evolution(t_values):
    """Evolve the system using QuTiP's master equation solver."""
    # Initial state
    psi0 = qt.Qobj(dbz.array([1/dbz.sqrt(2), 1/dbz.sqrt(2)]))
    
    # Time-dependent Hamiltonian
    H = [create_qutip_hamiltonian(t) for t in t_values]
    dt = t_values[1] - t_values[0]
    
    # Initialize states list with initial state
    states = [psi0]
    
    # Evolve through each time step
    for i in range(len(t_values)-1):
        result = qt.sesolve(H[i], states[-1], [0, dt])
        states.append(result.states[-1])
    
    # Calculate survival probability
    probs = []
    for state in states:
        overlap = abs(state.overlap(psi0))**2
        probs.append(float(overlap))
    
    return probs

def dividebyzero_evolution(t_values):
    """Evolve the system using dividebyzero's holonomy approach."""
    # Initial state
    psi0 = QuantumTensor(dbz.array([1/dbz.sqrt(2), 1/dbz.sqrt(2)]))
    
    # Initialize holonomy calculator
    gauge_group = U1Group()
    holonomy_calc = HolonomyCalculator(gauge_group)
    
    # Calculate survival probability for each time point
    probs = []
    current_state = psi0
    dt = t_values[1] - t_values[0]
    
    for t in t_values:
        # Create a loop in parameter space around current point
        radius = dt/4  # Smaller radius for better approximation
        num_points = 8  # More points for smoother loop
        
        # Create a circular loop in parameter space
        theta = dbz.linspace(0, 2*dbz.pi, num_points)
        loop = [(t + radius*dbz.cos(th), radius*dbz.sin(th)) for th in theta]
        
        # Define the Hamiltonian function for the loop
        def H_loop(params):
            t_val, eps = params
            H = dbz.array([[t_val, eps], [eps, -t_val]])
            return H
        
        # Calculate Berry phase
        phase = holonomy_calc.berry_phase(H_loop, loop)
        
        # Construct evolution operator
        U = dbz.array([
            [dbz.exp(1j * phase), 0],
            [0, dbz.exp(-1j * phase)]
        ])
        
        # Ensure current state is a column vector
        if len(current_state.data.shape) == 1:
            current_state.data = current_state.data.reshape(-1, 1)
        
        # Evolve state
        evolved_data = U @ current_state.data
        current_state = QuantumTensor(evolved_data)
        
        # Calculate survival probability
        overlap = dbz.abs(dbz.vdot(psi0.data.flatten(), current_state.data.flatten()))**2
        probs.append(overlap)
    
    return probs

def main():
    # Use more points and larger range for better resolution
    t_values = dbz.linspace(-1.0, 1.0, 100)
    
    # Compare both methods
    qutip_probs = qutip_evolution(t_values)
    dividebyzero_probs = dividebyzero_evolution(t_values)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, qutip_probs, 'b-', label='QuTiP Evolution', linewidth=2)
    plt.plot(t_values, dividebyzero_probs, 'r--', label='dividebyzero Evolution', linewidth=2)
    plt.axvline(x=0, color='k', linestyle=':', label='Degenerate Point')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title('Quantum Evolution: QuTiP vs dividebyzero')
    plt.legend()
    plt.grid(True)
    plt.savefig('quantum_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main() 