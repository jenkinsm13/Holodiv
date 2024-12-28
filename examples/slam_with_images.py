import matplotlib.pyplot as plt
import cv2
import dividebyzero as dbz
from pathlib import Path
import logging
import numpy as np

logging.getLogger('matplotlib').setLevel(logging.WARNING)

class SLAMProcessor:
    def __init__(self):
        # SIFT for feature detection and matching
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        
        # Store camera trajectory and orientations
        self.camera_positions = []
        self.camera_orientations = []
        self.feature_tracks = {}  # Track features across frames
        
    def detect_features(self, image):
        """Detect SIFT features in an image."""
        keypoints, descriptors = self.feature_detector.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """Match features between consecutive frames."""
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:  # Lowe's ratio test
                good_matches.append(m)
        return good_matches
    
    def create_metric_tensor(self, points):
        """
        Create metric tensor for the feature space.
        Uses dividebyzero to handle degenerate configurations.
        """
        # Convert points to numpy array for mean computation
        points_np = np.array(points.data)
        mean_pt = dbz.array(points_np.mean(axis=0))
        centered = points - mean_pt
        
        # Use dbz.array to handle potential degeneracies
        centered_np = np.array(centered.data)
        cov = dbz.array(centered_np.T @ centered_np)
        return cov
    
    def process_frame_pair(self, kp1, desc1, kp2, desc2):
        """
        Process a pair of frames using dividebyzero framework.
        Returns the relative camera motion and tracked features.
        """
        # Match features
        matches = self.match_features(desc1, desc2)
        
        # Extract matched points as numpy arrays first
        pts1_np = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2_np = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Convert to dbz arrays
        pts1 = dbz.array(pts1_np)
        pts2 = dbz.array(pts2_np)
        
        # Create metric tensor using feature distribution
        g = self.create_metric_tensor(pts1)
        
        # Use dividebyzero to compute essential matrix
        # This naturally handles degenerate cases
        E = self.compute_essential_matrix(pts1, pts2, g)
        
        # Recover camera motion
        R, t = self.decompose_essential_matrix(E)
        
        return R, t, matches
    
    def compute_essential_matrix(self, pts1, pts2, metric):
        """
        Compute essential matrix using dividebyzero framework.
        Handles degenerate point configurations naturally.
        """
        # Points are already in the right shape from process_frame_pair
        p1 = pts1  # Already (N, 2)
        p2 = pts2  # Already (N, 2)
        
        # Normalize points using metric
        # Apply metric to each point
        norm_pts1 = dbz.array([dbz.inner_product(pt, metric) for pt in p1])
        norm_pts2 = dbz.array([dbz.inner_product(pt, metric) for pt in p2])
        
        # Convert to numpy arrays and ensure correct format for OpenCV
        pts1_np = np.float32(norm_pts1.array).reshape(-1, 1, 2)
        pts2_np = np.float32(norm_pts2.array).reshape(-1, 1, 2)
        
        # Assume unit camera matrix for simplicity
        camera_matrix = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=np.float32)
        
        # Compute essential matrix
        E, mask = cv2.findEssentialMat(pts1_np, pts2_np, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        # Convert result back to dbz array
        return dbz.array(E)
    
    def decompose_essential_matrix(self, E):
        """Extract rotation and translation from essential matrix."""
        # Convert to numpy for SVD
        E_np = E.data
        
        # SVD decomposition
        U, _, Vt = dbz.linalg.svd(E_np)
        
        # Basic decomposition (simplified for example)
        W = dbz.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        R = dbz.array(U) @ W @ dbz.array(Vt)
        t = dbz.array(U[:, 2])
        
        return R, t
    
    def process_image_sequence(self, image_paths):
        """
        Process a sequence of images and reconstruct camera trajectory.
        """
        images = []
        features = []
        
        # Load and preprocess images
        for path in image_paths:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image: {path}")
            images.append(img)
            kp, desc = self.detect_features(img)
            features.append((kp, desc))
        
        # Initialize camera pose
        self.camera_positions.append(dbz.zeros(3))
        self.camera_orientations.append(dbz.eye(3))
        
        # Process consecutive frames
        for i in range(len(images)-1):
            kp1, desc1 = features[i]
            kp2, desc2 = features[i+1]
            
            # Get relative motion
            R, t, matches = self.process_frame_pair(kp1, desc1, kp2, desc2)
            
            # Update camera pose
            last_pos = self.camera_positions[-1]
            last_rot = self.camera_orientations[-1]
            
            new_rot = last_rot @ R
            new_pos = last_pos + (last_rot @ t)
            
            self.camera_positions.append(new_pos)
            self.camera_orientations.append(new_rot)
            
            # Update feature tracks
            self.update_feature_tracks(kp1, kp2, matches, i)
    
    def update_feature_tracks(self, kp1, kp2, matches, frame_idx):
        """Update feature tracks across frames."""
        for match in matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            
            # Try to extend existing tracks
            track_extended = False
            for track_id, track in self.feature_tracks.items():
                if track[-1][0] == frame_idx and track[-1][1] == pt1:
                    track.append((frame_idx + 1, pt2))
                    track_extended = True
                    break
            
            # Create new track if not extended
            if not track_extended:
                track_id = len(self.feature_tracks)
                self.feature_tracks[track_id] = [(frame_idx, pt1), (frame_idx + 1, pt2)]
    
    def plot_reconstruction(self):
        """
        Visualize the reconstructed camera trajectory and feature points.
        """
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot camera trajectory
        positions = dbz.array(self.camera_positions)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Camera Path')
        
        # Plot camera orientations
        for pos, rot in zip(self.camera_positions, self.camera_orientations):
            # Plot coordinate axes
            for i, color in enumerate(['r', 'g', 'b']):
                direction = rot[:, i]
                ax.quiver(pos[0], pos[1], pos[2],
                         direction[0], direction[1], direction[2],
                         length=0.2, color=color)
        
        # Plot feature tracks
        for track in self.feature_tracks.values():
            if len(track) > 5:  # Only plot longer tracks
                points = dbz.array([pt for _, pt in track])
                ax.scatter(points[:, 0], points[:, 1], dbz.zeros_like(points[:, 0]),
                          c='g', alpha=0.1, s=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Camera Trajectory and Feature Tracks')
        plt.show()

def main():
    """
    Main function to demonstrate SLAM with real images.
    """
    # Initialize SLAM processor
    slam = SLAMProcessor()
    
    # Get image paths (you'll need to provide actual images)
    image_dir = Path('data/slam_images')
    if not image_dir.exists():
        print(f"Please place your image sequence in {image_dir}")
        print("Creating directory for you...")
        image_dir.mkdir(parents=True, exist_ok=True)
        return
    
    image_paths = sorted(image_dir.glob('*.JPG'))
    if not image_paths:
        print(f"No images found in {image_dir}")
        print("Please add your image sequence (*.jpg) to this directory")
        return
    
    # Process image sequence
    try:
        slam.process_image_sequence(image_paths)
        slam.plot_reconstruction()
    except Exception as e:
        print(f"Error processing images: {e}")
        print("Please ensure your images are valid and in the correct format")

if __name__ == '__main__':
    main() 