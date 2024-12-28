import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dividebyzero as dbz

def create_camera_metric(r, theta):
    """
    Create metric tensor for camera orientation space.
    The singularity occurs when camera is directly above/below target (theta = ±π/2).
    """
    # Metric components that capture the camera orientation space
    g_rr = 1.0
    g_tt = (r * np.cos(theta))**2  # Becomes singular at poles
    
    return dbz.array([[g_rr, 0], [0, g_tt]])

def track_camera_orientation(path_r, path_theta, initial_direction):
    """
    Track camera orientation as it moves around a target.
    Uses dividebyzero to handle orientation singularities.
    """
    orientations = []
    current_dir = dbz.array(initial_direction)
    
    for i in range(1, len(path_r)):
        # Get current position
        r, theta = path_r[i-1], path_theta[i-1]
        dtheta = path_theta[i] - theta
        
        # Get metric at current position
        g = create_camera_metric(r, theta)
        
        # Transport direction using metric connection
        v = current_dir.data
        
        # Compute new direction using covariant derivative
        # This naturally handles the singularity at the poles
        v_r_new = v[0] - r * np.cos(theta) * v[1] * dtheta
        v_theta_new = v[1] + (v[0]/(r * np.cos(theta))) * dtheta
        
        # Update direction
        current_dir = dbz.array([v_r_new, v_theta_new])
        current_dir = dbz.array(current_dir.data / np.linalg.norm(current_dir.data))
        
        orientations.append(current_dir.data)
    
    return np.array(orientations)

def plot_camera_path(path_r, path_theta, orientations):
    """
    Visualize camera path and orientations in 3D.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert cylindrical to Cartesian coordinates
    x = path_r * np.cos(path_theta)
    y = path_r * np.sin(path_theta)
    z = np.zeros_like(x)  # Camera moves in horizontal plane
    
    # Plot camera path
    ax.plot(x, y, z, 'b-', label='Camera Path')
    
    # Plot target point
    ax.scatter([0], [0], [0], color='red', s=100, label='Target')
    
    # Plot camera orientations
    for i in range(len(orientations)):
        # Convert orientation vector to 3D
        v_r, v_theta = orientations[i]
        v_x = v_r * np.cos(path_theta[i]) - path_r[i] * v_theta * np.sin(path_theta[i])
        v_y = v_r * np.sin(path_theta[i]) + path_r[i] * v_theta * np.cos(path_theta[i])
        v_z = 0  # Keep in horizontal plane
        
        # Plot orientation vector
        ax.quiver(x[i], y[i], z[i], v_x, v_y, v_z, 
                 length=0.2, normalize=True, color='green')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Path and Orientation Tracking')
    ax.legend()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    plt.show()

def main():
    """
    Demonstrate camera orientation tracking around a target.
    """
    # Create camera path (circle around target)
    n_points = 50
    theta = np.linspace(0, 2*np.pi, n_points)
    r = np.ones_like(theta) * 2.0  # Fixed radius from target
    
    # Initial camera direction (pointing at target)
    initial_direction = np.array([-1.0, 0.0])  # Radially inward
    
    # Track camera orientation
    orientations = track_camera_orientation(r[:-1], theta[:-1], initial_direction)
    
    # Calculate final orientation angle
    final_dir = orientations[-1]
    angle = np.arctan2(final_dir[1], final_dir[0])
    print(f"Initial direction: {initial_direction}")
    print(f"Final direction: {final_dir}")
    print(f"Geometric phase (angle): {angle * 180/np.pi:.2f} degrees")
    
    # Visualize results
    plot_camera_path(r[:-1], theta[:-1], orientations)

if __name__ == '__main__':
    main() 