import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import csv
import math
class PointCloudViewer:
    """
    A general-purpose 3D point cloud viewer using OpenGL and GLFW.
    
    Attributes:
    -----------
    window : object
        The GLFW window object.
    points : np.ndarray
        The 3D point cloud data (Nx6 array of x, y, z, r, g, b for color).
    cam_position : np.ndarray
        The camera position in 3D space.
    cam_front : np.ndarray
        The direction the camera is facing.
    cam_up : np.ndarray
        The camera's up vector.
    speed : float
        The speed of camera movement.
    yaw : float
        The horizontal angle for camera rotation.
    pitch : float
        The vertical angle for camera rotation.
    """
    
    def __init__(self, width=800, height=600, title="3D Space Traveling Simulator", points=None):
        """Initialize the PointCloudViewer with a window and point cloud data."""
        self.width = width
        self.height = height
        self.title = title
        if points is not None and points.shape[1] == 6:
            self.points = points
            self.has_colors = True
        else:
            self.points = points if points is not None else np.array([])
            self.has_colors = False

        self.cam_position = np.array([0.0, 0.0, 0.0], dtype=float)
        self.cam_front = np.array([0.0, 0.0, -1.0], dtype=float)
        self.cam_up = np.array([0.0, 1.0, 0.0], dtype=float)
        self.speed = 10.0
        self.yaw = -90.0
        self.pitch = 0.0
        self.last_time = glfw.get_time()
        self.first_mouse = True
        self.last_x = self.width // 2
        self.last_y = self.height // 2
        self.keys = {}

        self.mouse_left_pressed = False

        # Initialize GLFW and OpenGL
        self.window = self.init_window()
        self.init_opengl()

    def init_window(self):
        """Initialize the GLFW window."""
        if not glfw.init():
            raise Exception("GLFW initialization failed!")
        window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not window:
            glfw.terminate()
            raise Exception("GLFW window creation failed!")
        glfw.make_context_current(window)
        return window

    def init_opengl(self):
        """Set up the OpenGL context and projection."""
        glEnable(GL_DEPTH_TEST)  # Enable depth testing for 3D rendering
        glEnable(GL_POINT_SMOOTH)
        glPointSize(2)  # Set point size for point cloud
        glClearColor(0.0, 0.0, 0.0, 1.0)  # Background color

        # Set up projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 10000)  # Set perspective
        
        # Set up model view matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def mouse_callback(self, window, xpos, ypos):
        """Handle mouse movements for camera rotation."""
        if self.first_mouse:
            self.last_x, self.last_y = xpos, ypos
            self.first_mouse = False

        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos  # Reversed since y-coordinates go from bottom to top
        self.last_x, self.last_y = xpos, ypos

        sensitivity = 0.1
        xoffset *= sensitivity
        yoffset *= sensitivity

        self.yaw += xoffset
        self.pitch += yoffset

        # Constrain the pitch
        if self.pitch > 89.0:
            self.pitch = 89.0
        if self.pitch < -89.0:
            self.pitch = -89.0

        # Update front vector
        front = np.array([
            math.cos(math.radians(self.yaw)) * math.cos(math.radians(self.pitch)),
            math.sin(math.radians(self.pitch)),
            math.sin(math.radians(self.yaw)) * math.cos(math.radians(self.pitch))
        ])
        self.cam_front = front / np.linalg.norm(front)

    def mouse_button_callback(self, window, button, action, mods):
        """Handle mouse button presses."""
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_left_pressed = (action == glfw.PRESS)

    def scroll_callback(self, window, xoffset, yoffset):
        """Handle scroll wheel for zoom."""
        self.speed += yoffset * 2

    def key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input."""
        if action == glfw.PRESS:
            self.keys[key] = True
        elif action == glfw.RELEASE:
            self.keys[key] = False

    def process_input(self):
        """Process keyboard input for camera movement."""
        current_time = glfw.get_time()
        delta_time = current_time - self.last_time
        self.last_time = current_time

        velocity = self.speed * delta_time

        if self.keys.get(glfw.KEY_W):
            self.cam_position += self.cam_front * velocity
        if self.keys.get(glfw.KEY_S):
            self.cam_position -= self.cam_front * velocity
        if self.keys.get(glfw.KEY_A):
            self.cam_position -= np.cross(self.cam_front, self.cam_up) * velocity
        if self.keys.get(glfw.KEY_D):
            self.cam_position += np.cross(self.cam_front, self.cam_up) * velocity
        if self.keys.get(glfw.KEY_SPACE):
            self.cam_position += self.cam_up * velocity
        if self.keys.get(glfw.KEY_LEFT_SHIFT):
            self.cam_position -= self.cam_up * velocity

    def apply_camera_transformations(self):
        """Apply camera transformations for rotating and zooming the view."""
        center = self.cam_position + self.cam_front
        gluLookAt(
            self.cam_position[0], self.cam_position[1], self.cam_position[2],
            center[0], center[1], center[2],
            self.cam_up[0], self.cam_up[1], self.cam_up[2]
        )

    def render(self):
        """Main rendering loop to display the point cloud."""
        while not glfw.window_should_close(self.window):
            self.process_input()

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            self.apply_camera_transformations()  # Apply the camera view

            # Render the point cloud
            glBegin(GL_POINTS)
            if self.has_colors:
                for point in self.points:
                    glColor3f(point[3], point[4], point[5])
                    glVertex3f(point[0], point[1], point[2])
            else:
                glColor3f(1.0, 1.0, 1.0)  # Default white color
                for x, y, z in self.points:
                    glVertex3f(x, y, z)
            glEnd()

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()

    def save_to_ply(self, filename):
        """Save the point cloud to a .ply file."""
        with open(filename, 'w') as f:
            f.write(f"ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(self.points)}\n")
            f.write(f"property float x\nproperty float y\nproperty float z\n")
            f.write(f"end_header\n")
            for point in self.points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
        print(f"Point cloud saved to {filename}")

    def save_to_csv(self, filename):
        """Save the point cloud to a .csv file."""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z"])
            writer.writerows(self.points)
        print(f"Point cloud saved to {filename}")

    def setup_callbacks(self):
        """Set up GLFW callbacks for mouse and keyboard interaction."""
        glfw.set_cursor_pos_callback(self.window, self.mouse_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)