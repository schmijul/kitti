import os
import glob
import time
import numpy as np
import glfw
import OpenGL.GL as gl
from simple_opengl_viewer import Simple3DViewer

class Logger:
    """Utility class for consistent debug logging"""
    @staticmethod
    def point_cloud_stats(prefix: str, points: np.ndarray):
        """Print statistics about a point cloud"""
        print(f"\n{prefix}:")
        print(f"  Shape: {points.shape}")
        print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        if points.shape[1] > 3:
            print(f"  Intensity range: [{points[:, 3].min():.2f}, {points[:, 3].max():.2f}]")

class OxtsData:
    """Container for OXTS GPS/IMU data"""
    def __init__(self, line):
        values = [float(x) for x in line.strip().split()]
        # Parse all OXTS fields
        self.lat = values[0]      # latitude (deg)
        self.lon = values[1]      # longitude (deg)
        self.alt = values[2]      # altitude (m)
        self.roll = values[3]     # roll angle (rad)
        self.pitch = values[4]    # pitch angle (rad)
        self.yaw = values[5]      # yaw angle (rad)
        self.vn = values[6]       # velocity towards north (m/s)
        self.ve = values[7]       # velocity towards east (m/s)
        self.vf = values[8]       # forward velocity (m/s)
        
    def __str__(self):
        """String representation for debugging"""
        return (f"OXTS Data:\n"
                f"  Position (lat,lon,alt): {self.lat:.6f}, {self.lon:.6f}, {self.alt:.2f}\n"
                f"  Orientation (roll,pitch,yaw): {self.roll:.2f}, {self.pitch:.2f}, {self.yaw:.2f}")

def load_oxts_data(oxts_file: str) -> OxtsData:
    """Load OXTS data from file"""
    with open(oxts_file, 'r') as f:
        data = OxtsData(f.readline())
        print(f"\nLoaded OXTS data from {os.path.basename(oxts_file)}:")
        print(data)
        return data

def get_transform_matrix(oxts: OxtsData, first_frame: bool = False) -> np.ndarray:
    """
    Create 4x4 transform matrix from OXTS data.
    For debugging, we'll start with minimal transformations.
    """
    # Start with identity matrix
    transform = np.eye(4)
    
    if first_frame:
        # For first frame, just apply a small offset to make points visible
        transform[0:3, 3] = [0, 0, 0]
    else:
        # For subsequent frames, apply simple translation
        # Using a very small scale factor for debugging
        scale = 0.0001  # Small scale to keep points visible
        x = oxts.lon * scale
        y = oxts.lat * scale
        z = oxts.alt * 0.1  # Reduce vertical scale
        transform[0:3, 3] = [x, y, z]
    
    print(f"\nTransform matrix:")
    print(transform)
    return transform

def load_lidar_data(bin_file: str) -> np.ndarray:
    """Load LiDAR point cloud from binary file"""
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    print(f"\nLoaded LiDAR data from {os.path.basename(bin_file)}")
    Logger.point_cloud_stats("Initial point cloud", points)
    return points

def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply transformation to points"""
    # Extract xyz coordinates
    xyz = points[:, :3]
    
    # Convert to homogeneous coordinates
    xyz_homogeneous = np.hstack([xyz, np.ones((len(xyz), 1))])
    
    # Apply transform
    transformed = (transform @ xyz_homogeneous.T).T
    
    # Update xyz coordinates while preserving intensity
    points_transformed = points.copy()
    points_transformed[:, :3] = transformed[:, :3]
    
    Logger.point_cloud_stats("Transformed point cloud", points_transformed)
    return points_transformed

def process_lidar_data(lidar_data: np.ndarray, 
                      transform: np.ndarray = None,
                      max_distance: float = None) -> np.ndarray:
    """Process LiDAR data with transformations and filtering"""
    processed_data = lidar_data.copy()
    
    # Apply distance filtering if specified
    if max_distance is not None:
        initial_points = len(processed_data)
        mask = (processed_data[:, 0] > -max_distance) & (processed_data[:, 0] < max_distance) & \
               (processed_data[:, 1] > -max_distance) & (processed_data[:, 1] < max_distance)
        processed_data = processed_data[mask]
        print(f"\nDistance filtering: {initial_points} -> {len(processed_data)} points")
        Logger.point_cloud_stats("After distance filtering", processed_data)
    
    # Apply transformation if specified
    if transform is not None:
        processed_data = transform_points(processed_data, transform)
    
    return processed_data

def setup_viewer() -> Simple3DViewer:
    """Initialize and configure the 3D viewer"""
    viewer = Simple3DViewer(width=1280, height=720, title="Debug LiDAR Viewer")
    
    # Configure initial view
    viewer.camera_distance = 150.0    # Further back to see more
    viewer.camera_angle_x = -0.7      # Look down more (~40 degrees)
    viewer.camera_angle_y = 0.785     # Rotate 45 degrees for 3/4 view
    viewer.point_size = 2.0           # Smaller points to reduce clutter
    
    
    print(f"\nViewer configuration:")
    print(f"  Camera distance: {viewer.camera_distance}")
    print(f"  Camera angle X: {viewer.camera_angle_x}")
    print(f"  Camera angle Y: {viewer.camera_angle_y}")
    print(f"  Point size: {viewer.point_size}")
    
    return viewer

def debug_single_frame(lidar_file: str, oxts_file: str, max_distance: float = None):
    """Process and visualize a single frame for debugging"""
    print("\n=== Starting single frame debug mode ===")
    
    # Initialize viewer
    viewer = setup_viewer()
    
    # Load and process data
    lidar_data = load_lidar_data(lidar_file)
    oxts_data = load_oxts_data(oxts_file)
    transform = get_transform_matrix(oxts_data, first_frame=True)
    processed_data = process_lidar_data(lidar_data, transform, max_distance)
    
    # Load points into viewer
    viewer.load_points(processed_data)
    print("\nStarting viewer loop. Press ESC to exit.")
    
    # Main loop with explicit GL calls
    while not glfw.window_should_close(viewer.window):
        # Clear buffers
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # Render frame
        viewer.run()
        
        # Swap buffers and poll events
        glfw.swap_buffers(viewer.window)
        glfw.poll_events()
        
        # Small delay to prevent maxing out CPU
        time.sleep(0.01)
    
    glfw.terminate()

def visualize_drive(lidar_folder: str,
                   oxts_folder: str,
                   max_distance: float = None,
                   frame_delay: float = 0.1,
                   accumulate: bool = False):
    """Visualize multiple LiDAR frames with OXTS data"""
    print("\n=== Starting multi-frame visualization ===")
    
    # Get sorted file lists
    bin_files = sorted(glob.glob(os.path.join(lidar_folder, '*.bin')))
    oxts_files = sorted(glob.glob(os.path.join(oxts_folder, '*.txt')))
    
    if not bin_files or not oxts_files:
        print("No data files found!")
        return
    
    print(f"Found {len(bin_files)} LiDAR files and {len(oxts_files)} OXTS files")
    
    viewer = setup_viewer()
    
    try:
        if accumulate:
            accumulated_data = []
            
            for i, (bin_file, oxts_file) in enumerate(zip(bin_files, oxts_files)):
                print(f"\nProcessing frame {i+1}/{len(bin_files)}")
                
                lidar_data = load_lidar_data(bin_file)
                oxts_data = load_oxts_data(oxts_file)
                transform = get_transform_matrix(oxts_data, first_frame=(i==0))
                
                processed_data = process_lidar_data(
                    lidar_data,
                    transform=transform,
                    max_distance=max_distance
                )
                accumulated_data.append(processed_data)
            
            # Combine all frames
            combined_data = np.vstack(accumulated_data)
            Logger.point_cloud_stats("Final combined point cloud", combined_data)
            
            viewer.load_points(combined_data)
            print("\nStarting viewer loop with combined data. Press ESC to exit.")
            
            while not glfw.window_should_close(viewer.window):
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                viewer.run()
                glfw.swap_buffers(viewer.window)
                glfw.poll_events()
                time.sleep(0.01)
        
        else:
            print("\nStarting frame animation. Press ESC to exit.")
            while not glfw.window_should_close(viewer.window):
                for i, (bin_file, oxts_file) in enumerate(zip(bin_files, oxts_files)):
                    if glfw.window_should_close(viewer.window):
                        break
                    
                    lidar_data = load_lidar_data(bin_file)
                    oxts_data = load_oxts_data(oxts_file)
                    transform = get_transform_matrix(oxts_data, first_frame=(i==0))
                    
                    processed_data = process_lidar_data(
                        lidar_data,
                        transform=transform,
                        max_distance=max_distance
                    )
                    
                    viewer.load_points(processed_data)
                    
                    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                    viewer.run()
                    glfw.swap_buffers(viewer.window)
                    glfw.poll_events()
                    
                    time.sleep(frame_delay)
    
    finally:
        glfw.terminate()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug-Enabled LiDAR Viewer')
    parser.add_argument('--lidar_folder', type=str, default='data/velodyne_points/data',
                      help='Folder containing LiDAR .bin files')
    parser.add_argument('--oxts_folder', type=str, default='data/oxts/data',
                      help='Folder containing OXTS .txt files')
    parser.add_argument('--max_dist', type=float, default=50.0,
                      help='Maximum distance for point filtering')
    parser.add_argument('--delay', type=float, default=0.1,
                      help='Delay between frames when animating')
    parser.add_argument('--accumulate', action='store_true',
                      help='Accumulate all frames into one visualization')
    parser.add_argument('--debug', action='store_true',
                      help='Process only first frame with detailed debugging')
    
    args = parser.parse_args()
    
    if args.debug:
        # Process only first frame with debugging
        first_bin = sorted(glob.glob(os.path.join(args.lidar_folder, '*.bin')))[0]
        first_oxts = sorted(glob.glob(os.path.join(args.oxts_folder, '*.txt')))[0]
        debug_single_frame(first_bin, first_oxts, args.max_dist)
    else:
        # Process all frames
        visualize_drive(
            lidar_folder=args.lidar_folder,
            oxts_folder=args.oxts_folder,
            max_distance=args.max_dist,
            frame_delay=args.delay,
            accumulate=args.accumulate
        )