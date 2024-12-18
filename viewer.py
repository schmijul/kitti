import os
import glob
import time
import numpy as np
import glfw
import OpenGL.GL as gl
import math
import tracemalloc  # Optional: For memory tracking

# =========================
# Shader Compilation Utilities
# =========================

vertex_shader_source = """
#version 330 core
layout (location = 0) in vec4 in_position;
uniform mat4 u_mvp;
out vec3 v_color;

void main() {
    gl_Position = u_mvp * vec4(in_position.xyz, 1.0);
    
    // Determine color based on vertex ID (for axes)
    if (gl_VertexID % 2 == 0) { // Origin points
        v_color = vec3(1.0, 1.0, 1.0); // White
    } else { // Axis points
        if (gl_VertexID < 2) { // X-axis
            v_color = vec3(1.0, 0.0, 0.0); // Red
        } else if (gl_VertexID < 4) { // Y-axis
            v_color = vec3(0.0, 1.0, 0.0); // Green
        } else { // Z-axis
            v_color = vec3(0.0, 0.0, 1.0); // Blue
        }
    }
}
"""

fragment_shader_source = """
#version 330 core
in vec3 v_color;
out vec4 FragColor;

void main() {
    FragColor = vec4(v_color, 1.0);
}
"""

def compile_shader(source, shader_type):
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)
    # Check compile status
    status = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
    if not status:
        log = gl.glGetShaderInfoLog(shader).decode()
        shader_type_str = "VERTEX_SHADER" if shader_type == gl.GL_VERTEX_SHADER else "FRAGMENT_SHADER"
        gl.glDeleteShader(shader)
        raise RuntimeError(f"{shader_type_str} compilation failed: {log}")
    return shader

def create_program(vs_source, fs_source):
    vs = compile_shader(vs_source, gl.GL_VERTEX_SHADER)
    fs = compile_shader(fs_source, gl.GL_FRAGMENT_SHADER)
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vs)
    gl.glAttachShader(program, fs)
    gl.glLinkProgram(program)
    # Check link status
    status = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
    if not status:
        log = gl.glGetProgramInfoLog(program).decode()
        gl.glDeleteProgram(program)
        raise RuntimeError(f"Program link failed: {log}")
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    return program

# =========================
# Viewer Class (Modified)
# =========================

def create_axes():
    # Define axes in the ENU frame
    # X-axis (East) in Red
    # Y-axis (North) in Green
    # Z-axis (Up) in Blue
    axes = np.array([
        [0, 0, 0, 1], [10, 0, 0, 1],    # X-axis
        [0, 0, 0, 1], [0, 10, 0, 1],    # Y-axis
        [0, 0, 0, 1], [0, 0, 10, 1]     # Z-axis
    ], dtype=np.float32)
    return axes

class Simple3DViewer:
    def __init__(self, width=1280, height=720, title="3D Point Cloud Viewer"):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.set_window_user_pointer(self.window, self)

        # Set input callbacks
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self._mouse_move_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)

        self.program = create_program(vertex_shader_source, fragment_shader_source)
        self.mvp_loc = gl.glGetUniformLocation(self.program, "u_mvp")

        # Camera parameters
        self.camera_distance = 150.0
        self.camera_angle_x = -0.5  # Initial angle (~-28.6 degrees)
        self.camera_angle_y = 0.0

        # Mouse interaction
        self.mouse_pressed = False
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0

        # Point size
        self.point_size = 2.0
        gl.glPointSize(self.point_size)

        # Setup GL state
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.1, 0.1, 0.15, 1.0)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

        # VAO, VBO for LiDAR points
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 16, None)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        # VAO, VBO for Axes
        self.axes_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.axes_vao)
        self.axes_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.axes_vbo)
        axes = create_axes()
        gl.glBufferData(gl.GL_ARRAY_BUFFER, axes.nbytes, axes, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 16, None)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        self.num_axes = len(axes)
        self.num_points = 0

    def load_points(self, points: np.ndarray):
        """
        Load points into the GPU buffer.
        points: Nx4 float32 array (x,y,z,intensity).
        """
        self.num_points = len(points)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, points.nbytes, points, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        print(f"Loaded {self.num_points} points into the GPU buffer.")

    def _key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD:  # '+' key
                self.point_size += 1.0
                gl.glPointSize(self.point_size)
                print(f"Point size increased to: {self.point_size}")
            elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT:  # '-' key
                self.point_size = max(1.0, self.point_size - 1.0)
                gl.glPointSize(self.point_size)
                print(f"Point size decreased to: {self.point_size}")
            elif key == glfw.KEY_UP:
                self.camera_angle_x += 0.05
            elif key == glfw.KEY_DOWN:
                self.camera_angle_x -= 0.05
            elif key == glfw.KEY_LEFT:
                self.camera_angle_y += 0.05
            elif key == glfw.KEY_RIGHT:
                self.camera_angle_y -= 0.05
            elif key == glfw.KEY_W:
                self.camera_distance -= 0.5
                self.camera_distance = max(0.1, self.camera_distance)
                print(f"Camera distance: {self.camera_distance}")
            elif key == glfw.KEY_S:
                self.camera_distance += 0.5
                print(f"Camera distance: {self.camera_distance}")

    def _mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.mouse_pressed = True
                self.last_mouse_x, self.last_mouse_y = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                self.mouse_pressed = False

    def _mouse_move_callback(self, window, xpos, ypos):
        if self.mouse_pressed:
            dx = xpos - self.last_mouse_x
            dy = ypos - self.last_mouse_y
            sensitivity = 0.005
            self.camera_angle_y += dx * sensitivity
            self.camera_angle_x += dy * sensitivity
            self.last_mouse_x, self.last_mouse_y = xpos, ypos
            # Optional: Limit camera_angle_x to prevent flipping
            self.camera_angle_x = max(-math.pi/2, min(math.pi/2, self.camera_angle_x))
            print(f"Camera angles updated: X={self.camera_angle_x:.2f}, Y={self.camera_angle_y:.2f}")

    def _scroll_callback(self, window, xoffset, yoffset):
        zoom_factor = 1.0 - yoffset * 0.1
        if zoom_factor <= 0.0:
            zoom_factor = 0.1
        self.camera_distance *= zoom_factor
        self.camera_distance = max(self.camera_distance, 0.1)
        print(f"Camera distance adjusted to: {self.camera_distance}")

    def draw_axes(self):
        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self.mvp_loc, 1, gl.GL_FALSE, self.mvp_matrix)
        gl.glBindVertexArray(self.axes_vao)
        gl.glDrawArrays(gl.GL_LINES, 0, self.num_axes)
        gl.glBindVertexArray(0)

    def draw_frame(self):
        if glfw.window_should_close(self.window):
            return

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        width, height = glfw.get_framebuffer_size(self.window)
        aspect = width / float(height) if height > 0 else 1.0

        # Compute MVP
        fov = math.radians(60.0)
        znear, zfar = 0.1, 1000.0  # Increased zfar for larger scenes
        f = 1.0 / math.tan(fov/2.0)
        proj = np.array([
            [f/aspect, 0,     0,                                0],
            [0,        f,     0,                                0],
            [0,        0,     (zfar+znear)/(znear - zfar), (2*zfar*znear)/(znear - zfar)],
            [0,        0,    -1,                                0]
        ], dtype=np.float32)

        cx = self.camera_angle_x
        cy = self.camera_angle_y
        dist = self.camera_distance

        # Rotation matrices
        Rx = np.array([
            [1, 0,           0,          0],
            [0, math.cos(cx), -math.sin(cx), 0],
            [0, math.sin(cx), math.cos(cx),  0],
            [0, 0,           0,          1]
        ], dtype=np.float32)

        Ry = np.array([
            [ math.cos(cy), 0, math.sin(cy), 0],
            [0,             1, 0,            0],
            [-math.sin(cy), 0, math.cos(cy), 0],
            [0,             0, 0,            1]
        ], dtype=np.float32)

        T = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -dist],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        view = Rx @ Ry @ T
        mvp = proj @ view
        self.mvp_matrix = mvp  # Store for axes drawing

        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self.mvp_loc, 1, gl.GL_FALSE, mvp)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.num_points)
        gl.glBindVertexArray(0)

        # Draw axes
        self.draw_axes()

# =========================
# Main Script
# =========================

class OxtsData:
    ref_lat = None
    ref_lon = None
    ref_alt = None

    def __init__(self, line):
        values = [float(x) for x in line.strip().split()]
        if len(values) < 6:
            raise ValueError("OXTS data line does not have enough values.")
        self.lat = values[0]
        self.lon = values[1]
        self.alt = values[2]
        self.roll = values[3]
        self.pitch = values[4]
        self.yaw = values[5]

        if OxtsData.ref_lat is None:
            OxtsData.ref_lat = self.lat
            OxtsData.ref_lon = self.lon
            OxtsData.ref_alt = self.alt
            print(f"Reference position set: lat={self.lat:.6f}, lon={self.lon:.6f}, alt={self.alt:.2f}")

def load_lidar_data(bin_file: str) -> np.ndarray:
    if not os.path.isfile(bin_file):
        raise FileNotFoundError(f"LiDAR data file not found: {bin_file}")
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    print(f"\nLoaded {os.path.basename(bin_file)}: {points.shape[0]} points")
    return points

def setup_viewer() -> Simple3DViewer:
    viewer = Simple3DViewer(width=1280, height=720, title="LiDAR Visualization")
    viewer.camera_distance = 150.0
    viewer.camera_angle_x = -0.5  # Look down at ~30 degrees
    viewer.camera_angle_y = 0.0    # Face forward
    viewer.point_size = 2.0
    return viewer

def rotation_matrix(axis, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s,  c]])
    elif axis == 'y':
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]])
    else:  # 'z'
        return np.array([[ c, -s, 0],
                         [ s,  c, 0],
                         [ 0,  0, 1]])

def get_transform_matrix(oxts: OxtsData) -> np.ndarray:
    # Initialize reference if needed
    if not hasattr(get_transform_matrix, 'ref_lat'):
        get_transform_matrix.ref_lat = oxts.lat
        get_transform_matrix.ref_lon = oxts.lon
        get_transform_matrix.ref_alt = oxts.alt
        get_transform_matrix.ref_yaw = oxts.yaw
        print(f"Reference position set: lat={oxts.lat:.6f}, lon={oxts.lon:.6f}, alt={oxts.alt:.2f}")

    # Compute local ENU offsets from lat/lon/alt differences
    dlat = (oxts.lat - get_transform_matrix.ref_lat)
    dlon = (oxts.lon - get_transform_matrix.ref_lon)
    dalt = (oxts.alt - get_transform_matrix.ref_alt)

    # Approximate meters per degree latitude/longitude
    meters_per_deg_lat = 111319.9
    meters_per_deg_lon = 111319.9 * np.cos(np.radians(get_transform_matrix.ref_lat))

    dx = dlon * meters_per_deg_lon
    dy = dlat * meters_per_deg_lat
    dz = dalt

    # Compute rotation from roll, pitch, yaw
    Rz = rotation_matrix('z', oxts.yaw)
    Ry = rotation_matrix('y', oxts.pitch)
    Rx = rotation_matrix('x', oxts.roll)
    R = Rz @ Ry @ Rx

    # Construct the transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = [dx, dy, dz]
    return transform

def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    xyz = points[:, :3]
    intensity = points[:, 3]
    xyz_homogeneous = np.hstack([xyz, np.ones((len(xyz), 1))])
    transformed = (transform @ xyz_homogeneous.T).T
    return np.hstack([transformed[:, :3], intensity.reshape(-1, 1)])

def downsample_points(points: np.ndarray, max_points: int = 100000) -> np.ndarray:
    """
    Downsample the point cloud to a maximum number of points.
    """
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        return points[indices]
    return points

def process_frame(lidar_data: np.ndarray, transform: np.ndarray, max_distance: float, max_points: int = 100000) -> np.ndarray:
    if max_distance is not None:
        initial_points = len(lidar_data)
        # Only include points in front of the vehicle
        mask = (lidar_data[:, 0] > 0) & (lidar_data[:, 0] < max_distance) & \
               (lidar_data[:, 1] > -max_distance / 2) & (lidar_data[:, 1] < max_distance / 2) & \
               (lidar_data[:, 2] > -max_distance / 2) & (lidar_data[:, 2] < max_distance / 2)
        lidar_data = lidar_data[mask]
        print(f"Filtered: {initial_points} -> {len(lidar_data)} points")

    # Downsample the points to limit memory usage
    lidar_data = downsample_points(lidar_data, max_points)
    print(f"Downsampled to: {len(lidar_data)} points")

    return transform_points(lidar_data, transform)

def visualize_progressive_mapping(lidar_folder: str,
                                  oxts_folder: str,
                                  max_distance: float = 50.0,
                                  frame_delay: float = 0.1,
                                  window_size: int = 1,  # Reduced window_size
                                  max_points_per_frame: int = 100000):
    bin_files = sorted(glob.glob(os.path.join(lidar_folder, '*.bin')))
    oxts_files = sorted(glob.glob(os.path.join(oxts_folder, '*.txt')))

    if len(bin_files) != len(oxts_files):
        raise ValueError("Number of LiDAR files and OXTS files do not match.")

    viewer = setup_viewer()
    recent_frames = []
    frame_count = 0
    running = True

    # Start memory tracking (optional)
    tracemalloc.start()

    def key_callback(window, key, scancode, action, mods):
        nonlocal running
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            running = False

    glfw.set_key_callback(viewer.window, key_callback)

    while running and not glfw.window_should_close(viewer.window):
        if frame_count < len(bin_files):
            print(f"\rFrame {frame_count + 1}/{len(bin_files)}", end="", flush=True)

            # Load current frame
            try:
                lidar_data = load_lidar_data(bin_files[frame_count])
            except Exception as e:
                print(f"\nError loading LiDAR data: {e}")
                break

            try:
                with open(oxts_files[frame_count], 'r') as f:
                    oxts_line = f.readline()
                    oxts_data = OxtsData(oxts_line)
            except Exception as e:
                print(f"\nError loading OXTS data: {e}")
                break

            # Process frame
            transform = get_transform_matrix(oxts_data)
            processed_frame = process_frame(lidar_data, transform, max_distance, max_points_per_frame)

            # Update frame buffer
            recent_frames.append(processed_frame)
            if len(recent_frames) > window_size:
                recent_frames.pop(0)

            # Update visualization
            combined_cloud = np.vstack(recent_frames) if recent_frames else processed_frame
            viewer.load_points(combined_cloud)

            # Render a single frame
            viewer.draw_frame()
            glfw.swap_buffers(viewer.window)
            glfw.poll_events()

            # Optional: Monitor memory usage
            current, peak = tracemalloc.get_traced_memory()
            print(f" | Memory Usage: Current={current / 10**6:.2f}MB, Peak={peak / 10**6:.2f}MB", end='')

            frame_count += 1
            time.sleep(frame_delay)
        else:
            # Optionally, loop the visualization
            frame_count = 0
            recent_frames.clear()
            print("\nAll frames processed. Restarting visualization.")

    # Stop memory tracking and print statistics
    current, peak = tracemalloc.get_traced_memory()
    print(f"\nFinal Memory Usage: Current={current / 10**6:.2f}MB, Peak={peak / 10**6:.2f}MB")
    tracemalloc.stop()

    glfw.terminate()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Progressive LiDAR Visualization')
    parser.add_argument('--lidar_folder', type=str, default='data/velodyne_points/data',
                      help='Folder containing LiDAR .bin files')
    parser.add_argument('--oxts_folder', type=str, default='data/oxts/data',
                      help='Folder containing OXTS .txt files')
    parser.add_argument('--max_dist', type=float, default=50.0,
                      help='Maximum distance for point filtering')
    parser.add_argument('--delay', type=float, default=0.1,
                      help='Delay between frames (in seconds)')
    parser.add_argument('--window', type=int, default=1,  # Reduced window_size
                      help='Number of recent frames to maintain')
    parser.add_argument('--max_points', type=int, default=100000,  # Limit points per frame
                      help='Maximum number of points per frame after downsampling')

    args = parser.parse_args()

    print("\nControls:")
    print("- Mouse drag: Rotate view")
    print("- Scroll wheel: Zoom")
    print("- Arrow keys: Rotate camera")
    print("- W/S: Move camera closer/farther")
    print("- +/-: Adjust point size")
    print("- ESC: Exit\n")

    visualize_progressive_mapping(
        lidar_folder=args.lidar_folder,
        oxts_folder=args.oxts_folder,
        max_distance=args.max_dist,
        frame_delay=args.delay,
        window_size=args.window,
        max_points_per_frame=args.max_points
    )
