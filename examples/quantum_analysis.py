import matplotlib.pyplot as plt
import dividebyzero as dbz
from dividebyzero.quantum.tensor import QuantumTensor
import seaborn as sns
from scipy.linalg import logm

def create_density_matrix(state_vector):
    """Create density matrix from state vector."""
    return dbz.outer(state_vector, state_vector.conj())

def plot_density_matrix(rho, title="Density Matrix"):
    """Plot density matrix with phase information."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Magnitude plot
    im1 = ax1.imshow(dbz.abs(rho), cmap='viridis')
    ax1.set_title(f"{title} (Magnitude)")
    plt.colorbar(im1, ax=ax1)
    
    # Phase plot
    im2 = ax2.imshow(dbz.angle(rho), cmap='twilight')
    ax2.set_title(f"{title} (Phase)")
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    return fig

def analyze_quantum_properties(state, title="Quantum State Analysis"):
    """Comprehensive analysis of quantum properties."""
    print(f"\n=== {title} ===")
    
    # Calculate density matrix
    rho = create_density_matrix(state.data.flatten())
    
    # Eigenvalue analysis
    eigvals = dbz.linalg.eigvals(rho)
    print(f"Eigenvalues: {eigvals}")
    
    # Purity
    purity = dbz.trace(rho @ rho).real
    print(f"Purity: {purity:.6f}")
    
    # Von Neumann entropy
    entropy = -dbz.trace(rho @ dbz.logm(rho)).real
    print(f"Von Neumann Entropy: {entropy:.6f}")
    
    # Plot density matrix
    plot_density_matrix(rho, title)
    
    return {
        'density_matrix': rho,
        'eigenvalues': eigvals,
        'purity': purity,
        'entropy': entropy
    }

# Create test states
def create_ghz_state():
    """Create GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2"""
    psi = dbz.zeros((2, 2, 2), dtype=complex)
    psi[0,0,0] = 1/dbz.sqrt(2)
    psi[1,1,1] = 1/dbz.sqrt(2)
    return QuantumTensor(psi)

def create_w_state():
    """Create W state |W⟩ = (|100⟩ + |010⟩ + |001⟩)/√3"""
    psi = dbz.zeros((2, 2, 2), dtype=complex)
    psi[1,0,0] = 1/dbz.sqrt(3)
    psi[0,1,0] = 1/dbz.sqrt(3)
    psi[0,0,1] = 1/dbz.sqrt(3)
    return QuantumTensor(psi)

# Test states
ghz = create_ghz_state()
w = create_w_state()

# Create zero divisors
complete_zero = QuantumTensor(dbz.zeros((2, 2, 2)))
strategic_zero = QuantumTensor(dbz.ones((2, 2, 2)))
strategic_zero.data[0,0,0] = 0  # Zero at |000⟩

# Analyze original states
ghz_analysis = analyze_quantum_properties(ghz, "GHZ State (Original)")
w_analysis = analyze_quantum_properties(w, "W State (Original)")

# Perform division and analyze results
ghz_divided = ghz / complete_zero
w_divided = w / strategic_zero

ghz_div_analysis = analyze_quantum_properties(ghz_divided, "GHZ State (After Division)")
w_div_analysis = analyze_quantum_properties(w_divided, "W State (After Division)")

# Plot entanglement spectrum evolution
def plot_entanglement_spectrum(original, divided, title):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.stem(original._entanglement_spectrum.schmidt_values)
    plt.title(f"{title} - Original Schmidt Values")
    plt.ylabel("Schmidt Value")
    plt.xlabel("Index")
    
    plt.subplot(1, 2, 2)
    plt.stem(divided._entanglement_spectrum.schmidt_values)
    plt.title(f"{title} - After Division Schmidt Values")
    plt.ylabel("Schmidt Value")
    plt.xlabel("Index")
    
    plt.tight_layout()

plot_entanglement_spectrum(ghz, ghz_divided, "GHZ State")
plot_entanglement_spectrum(w, w_divided, "W State")

# Analyze quantum correlations
def compute_mutual_information(rho, subsys_dims):
    """Compute quantum mutual information between subsystems."""
    n = len(subsys_dims)
    mutual_info = dbz.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            # Compute reduced density matrices
            rho_i = dbz.trace(rho.reshape(subsys_dims[i], -1), axis=1)
            rho_j = dbz.trace(rho.reshape(subsys_dims[j], -1), axis=1)
            rho_ij = dbz.trace(rho.reshape(subsys_dims[i]*subsys_dims[j], -1), axis=1)
            
            # Compute von Neumann entropies
            S_i = -dbz.trace(rho_i @ dbz.logm(rho_i)).real
            S_j = -dbz.trace(rho_j @ dbz.logm(rho_j)).real
            S_ij = -dbz.trace(rho_ij @ dbz.logm(rho_ij)).real
            
            mutual_info[i,j] = S_i + S_j - S_ij
            mutual_info[j,i] = mutual_info[i,j]
    
    return mutual_info

# Plot mutual information matrices
subsys_dims = [2, 2, 2]  # For 3-qubit states

mi_ghz_orig = compute_mutual_information(ghz_analysis['density_matrix'], subsys_dims)
mi_ghz_div = compute_mutual_information(ghz_div_analysis['density_matrix'], subsys_dims)

mi_w_orig = compute_mutual_information(w_analysis['density_matrix'], subsys_dims)
mi_w_div = compute_mutual_information(w_div_analysis['density_matrix'], subsys_dims)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
sns.heatmap(mi_ghz_orig, ax=axes[0,0], cmap='viridis')
axes[0,0].set_title("GHZ State - Original MI")

sns.heatmap(mi_ghz_div, ax=axes[0,1], cmap='viridis')
axes[0,1].set_title("GHZ State - After Division MI")

sns.heatmap(mi_w_orig, ax=axes[1,0], cmap='viridis')
axes[1,0].set_title("W State - Original MI")

sns.heatmap(mi_w_div, ax=axes[1,1], cmap='viridis')
axes[1,1].set_title("W State - After Division MI")

plt.tight_layout()

# Print summary statistics
print("\n=== Summary Statistics ===")
print("GHZ State:")
print(f"Original MI mean: {dbz.mean(mi_ghz_orig):.4f}")
print(f"Divided MI mean: {dbz.mean(mi_ghz_div):.4f}")
print(f"MI preservation ratio: {dbz.mean(mi_ghz_div)/dbz.mean(mi_ghz_orig):.4f}")

print("\nW State:")
print(f"Original MI mean: {dbz.mean(mi_w_orig):.4f}")
print(f"Divided MI mean: {dbz.mean(mi_w_div):.4f}")
print(f"MI preservation ratio: {dbz.mean(mi_w_div)/dbz.mean(mi_w_orig):.4f}")