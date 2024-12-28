import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dividebyzero as dbz
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def create_cone_metric(r, theta, alpha=0.5):
    """
    Create the metric tensor for a cone with opening angle alpha.
    The cone has a coordinate singularity at r=0.
    
    Args:
        r: Radial coordinate
        theta: Angular coordinate
        alpha: Cone opening angle (controls the "pointiness", smaller = sharper)
    """
    # For a cone embedded in 3D space with height h = r/alpha
    # The metric includes both the radial and height contributions
    beta = dbz.sqrt(1 + (1/alpha)**2)  # Accounts for slope of cone
    
    # Metric components in (r, theta) coordinates
    g_rr = beta**2  # Includes vertical component
    g_tt = r**2     # Standard angular part
    
    # Create metric tensor using dividebyzero array
    g = dbz.array([[g_rr, 0], [0, g_tt]])
    return g

def parallel_transport_cone(r_path, theta_path, initial_vector):
    """
    Parallel transport a vector along a path on the cone.
    Uses dividebyzero to handle the coordinate singularity at r=0.
    
    Args:
        r_path: Array of radial coordinates
        theta_path: Array of angular coordinates
        initial_vector: Initial vector to transport
    """
    vectors = []
    current_vector = dbz.array(initial_vector)
    original_magnitude = dbz.linalg.norm(initial_vector)
    
    for i in range(1, len(r_path)):
        # Get current point
        r, theta = r_path[i-1], theta_path[i-1]
        dtheta = theta_path[i] - theta
        
        # Get metric at current point
        g = create_cone_metric(r, theta)
        
        # Transport using metric connection
        v = current_vector.data
        
        # Use metric to compute covariant derivative
        v_r_new = v[0] - r * v[1] * dtheta
        v_theta_new = v[1] + (v[0]/r) * dtheta
        
        # Create new vector
        transported = dbz.array([v_r_new, v_theta_new])
        
        # Normalize to preserve magnitude
        transported_np = dbz.array(transported.data, copy=True)
        current_magnitude = dbz.linalg.norm(transported_np)
        if current_magnitude > 0:  # Avoid division by zero
            current_vector = dbz.array(transported_np * (original_magnitude / current_magnitude))
        else:
            current_vector = transported
        
        # Store for plotting
        vectors.append(dbz.array(current_vector.data, copy=True))
    
    return dbz.array(vectors)

def plot_cone_transport(r_path, theta_path, vectors, save_png=False):
    """
    Plot the cone and the parallel transported vectors.
    """
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot cone surface
    theta = dbz.linspace(0, 2*dbz.pi, 100)
    r = dbz.linspace(0, 1, 100)
    theta_grid, r_grid = dbz.meshgrid(theta, r)
    x = r_grid * dbz.cos(theta_grid)
    y = r_grid * dbz.sin(theta_grid)
    z = 2 * r_grid  # Make cone sharper by doubling height
    ax.plot_surface(x, y, z, alpha=0.3)
    
    # Plot transported vectors
    for i in range(len(vectors)):
        # Get point on cone surface
        x = r_path[i] * dbz.cos(theta_path[i])
        y = r_path[i] * dbz.sin(theta_path[i])
        z = 2 * r_path[i]  # Adjust height for sharper cone
        p = dbz.array([x, y, z])
        
        # Convert 2D vector to 3D vector on cone surface
        v_r, v_theta = vectors[i]
        # Tangent vector components in 3D
        v_x = v_r * dbz.cos(theta_path[i]) - r_path[i] * v_theta * dbz.sin(theta_path[i])
        v_y = v_r * dbz.sin(theta_path[i]) + r_path[i] * v_theta * dbz.cos(theta_path[i])
        v_z = 2 * v_r  # Adjust for sharper cone
        v = dbz.array([v_x, v_y, v_z])
        
        # Plot vector
        ax.quiver(p[0], p[1], p[2], v[0], v[1], v[2], length=0.1, normalize=True)
    
    # Set labels and adjust view
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=45)  # Lower viewing angle to see cone better
    
    # Make the plot more interactive
    ax.set_box_aspect([1,1,2])  # Taller aspect ratio for sharper cone
    
    if save_png:
        plt.savefig('cone_transport_sharp.png')
        plt.close()
    else:
        plt.show()

def main():
    """
    Main function to demonstrate parallel transport on a cone.
    """
    # Create a path that circles the cone
    n_points = 50
    theta = dbz.linspace(0, 2*dbz.pi, n_points)
    r = dbz.ones_like(theta) * 0.2  # Closer to tip
    
    # Initial vector
    initial_vector = dbz.array([1.0, 0.0])  # Radial direction
    
    # Perform parallel transport
    vectors = parallel_transport_cone(r[:-1], theta[:-1], initial_vector)
    
    # Check geometric phase
    final_vector = vectors[-1]
    angle = dbz.arctan2(final_vector[1], final_vector[0])
    print(f"Initial vector: {initial_vector}")
    print(f"Final vector: {final_vector}")
    print(f"Geometric phase (angle): {angle * 180/dbz.pi:.2f} degrees")
    
    # Save PNG first
    plot_cone_transport(r[:-1], theta[:-1], vectors, save_png=True)
    # Then show interactive plot
    plot_cone_transport(r[:-1], theta[:-1], vectors, save_png=False)

if __name__ == '__main__':
    main() 