import numpy as np
import os
import glob
import cv2

# Constants for scaling and frame size
LIDAR_SCALE_FACTOR = 5
LIDAR_FRAME_SIZE = (800, 800)
CAMERA_FRAME_SIZE = (800, 800)
COMBINED_FRAME_SIZE = (1600, 800)
DISPLAY_SCALE = 0.5
FPS = 5
OUTPUT_VIDEO_PATH = 'lidar_camera_top_view.avi'

def load_lidar_data(bin_file: str) -> np.ndarray:
    """Load Lidar data from a .bin file.
    
    Velodyne Lidar data typically consists of 4 floats per point: x, y, z, and intensity.
    
    Args:
        bin_file: Path to the .bin file containing Lidar data.
    
    Returns:
        A numpy array of shape (n_points, 4) containing the Lidar data points.
    """
    return np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)

def process_lidar_data(lidar_data: np.ndarray, scale_factor: int) -> np.ndarray:
    """Extract and normalize the Lidar x, y coordinates for top-down view.
    
    Args:
        lidar_data: A numpy array containing the Lidar data points.
        scale_factor: Scaling factor to adjust the x, y coordinates for display.
    
    Returns:
        A tuple of two numpy arrays representing scaled x and y coordinates.
    """
    x_coords = ((lidar_data[:, 0] + 80) * scale_factor).astype(np.int32)
    y_coords = ((lidar_data[:, 1] + 80) * scale_factor).astype(np.int32)
    return x_coords, y_coords

def create_lidar_frame(x_coords: np.ndarray, y_coords: np.ndarray, frame_size: tuple) -> np.ndarray:
    """Create a blank frame and draw Lidar points for top-down view.
    
    Args:
        x_coords: The x coordinates of the Lidar points.
        y_coords: The y coordinates of the Lidar points.
        frame_size: The size of the output frame.
    
    Returns:
        An image frame (numpy array) with Lidar points drawn on it.
    """
    frame = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
    for x, y in zip(x_coords, y_coords):
        cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
    return frame

def main():
    """Main function to process Lidar and camera data and generate a video."""
    
    # Paths to the folders containing .bin files and camera images
    lidar_folder = 'data/velodyne_points/data/'
    camera_folder = 'data/image_02/data/'  # Change based on the desired camera
    
    # Get all .bin files and image files in sorted order
    bin_files = sorted(glob.glob(os.path.join(lidar_folder, '*.bin')))
    image_files = sorted(glob.glob(os.path.join(camera_folder, '*.png')))
    
    # Ensure the number of Lidar and camera files match
    if len(bin_files) != len(image_files):
        raise ValueError("Number of Lidar files and camera images should match.")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, COMBINED_FRAME_SIZE)
    
    # Set display window size
    display_size = (int(COMBINED_FRAME_SIZE[0] * DISPLAY_SCALE), 
                    int(COMBINED_FRAME_SIZE[1] * DISPLAY_SCALE))

    # Loop through each .bin and corresponding image file
    for bin_file, image_file in zip(bin_files, image_files):
        # Load Lidar data and process the coordinates for top-down view
        lidar_data = load_lidar_data(bin_file)
        x_img, y_img = process_lidar_data(lidar_data, LIDAR_SCALE_FACTOR)
        
        # Create Lidar top view frame
        lidar_frame = create_lidar_frame(x_img, y_img, LIDAR_FRAME_SIZE)
        
        # Load and resize the corresponding camera image
        camera_frame = cv2.imread(image_file)
        camera_frame = cv2.resize(camera_frame, CAMERA_FRAME_SIZE)
        
        # Concatenate Lidar and camera frames side by side
        combined_frame = np.hstack((lidar_frame, camera_frame))
        
        # Write the combined frame to the video file
        out.write(combined_frame)
        
        # Display the frame in a window, scaled down for visibility
        display_frame = cv2.resize(combined_frame, display_size)
        cv2.imshow('Lidar and Camera View', display_frame)
        
        # Break loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    out.release()
    cv2.destroyAllWindows()
    print(f'Video saved as {OUTPUT_VIDEO_PATH}')

if __name__ == '__main__':
    main()
