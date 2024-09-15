import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import struct
import open3d as o3d



class SLAM:
    def __init__(self, calibration_dir, data_dir):
        # Load calibration data for cameras and LiDAR
        self.calib_cam_to_cam = self.load_calibration(f"{calibration_dir}/calib_cam_to_cam.txt")
        self.calib_velo_to_cam = self.load_calibration(f"{calibration_dir}/calib_velo_to_cam.txt")
        
        # Set up directories for image and lidar data
        self.image_dir = f"{data_dir}/image_02/data"
        self.lidar_dir = f"{data_dir}/velodyne_points/data"
        
        # Initialize storage for poses and map
        self.poses = []
        self.map = []
        
    def load_calibration(self, calib_file):
        """Load the calibration file, skipping non-numeric lines."""
        calib_data = {}
        with open(calib_file, 'r') as f:
            for line in f:
                key, value = line.strip().split(':', 1)
                try:
                # Try to convert the values to floats
                    calib_data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    # Skip lines that don't contain valid numeric data
                    print(f"Skipping line: {line.strip()}")
                    continue
        return calib_data

    
    def load_image(self, idx):
        """Load the image by index"""
        img_path = f"{self.image_dir}/{idx:010d}.png"
        return Image.open(img_path)
    
    def load_lidar(self, idx):
        """Load the lidar point cloud"""
        lidar_path = f"{self.lidar_dir}/{idx:010d}.bin"
        return np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    
    def visualize_camera(self, idx):
        """Visualize the camera image"""
        image = self.load_image(idx)
        plt.imshow(image)
        plt.title(f"Camera Frame {idx}")
        plt.show()
    
    
    def visualize_lidar(self, idx):
      """Visualize the LiDAR point cloud using Open3D."""
      lidar_points = self.load_lidar(idx)
    
      # Create an Open3D PointCloud object
      pcd = o3d.geometry.PointCloud()
      
      # Assign points from LiDAR (x, y, z)
      pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3])  # Only take the first 3 columns (x, y, z)
      
      # Optionally, assign colors or intensities from LiDAR (4th column is intensity)
      if lidar_points.shape[1] > 3:
          colors = lidar_points[:, 3] / np.max(lidar_points[:, 3])  # Normalize intensity to [0, 1]
          pcd.colors = o3d.utility.Vector3dVector(np.tile(colors.reshape(-1, 1), (1, 3)))  # Grayscale

      # Visualize the point cloud
      o3d.visualization.draw_geometries([pcd])   
    def update_map(self, lidar_points):
        """Update the SLAM map with new LiDAR points"""
        self.map.append(lidar_points)
    def run_slam(self):
        """Run the SLAM algorithm over the dataset"""
        for idx in range(77):  # Adjust the range based on the number of frames
            print(f"Processing frame {idx}...")
            image = self.load_image(idx)
            lidar_points = self.load_lidar(idx)
            
            # Visualize camera and lidar data
            self.visualize_camera(idx)
            self.visualize_lidar(idx)
            
            # Update the map with lidar points (this is a placeholder for the actual SLAM logic)
            self.update_map(lidar_points)

# Usage example
calibration_dir = "./calibrationtxt"
data_dir = "./data"
slam = SLAM(calibration_dir, data_dir)
slam.run_slam()

