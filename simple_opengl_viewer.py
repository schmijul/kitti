import glfw
import OpenGL.GL as gl
import numpy as np
import math

vertex_shader_source = """
#version 330 core
layout (location = 0) in vec4 in_position;
uniform mat4 u_mvp;
out float v_intensity;

void main() {
    gl_Position = u_mvp * vec4(in_position.xyz, 1.0);
    v_intensity = in_position.w;
}
"""

fragment_shader_source = """
#version 330 core
in float v_intensity;
out vec4 FragColor;

void main() {
    // Map intensity to color (blue to red)
    vec3 color = vec3(v_intensity, 0.0, 1.0 - v_intensity);
    FragColor = vec4(color, 1.0);
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
        gl.glDeleteShader(shader)
        raise RuntimeError(f"Shader compilation failed ({shader_type}): {log}")
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
        self.camera_distance = 15.0  # Increased to fit all shapes
        self.camera_angle_x = 0.0
        self.camera_angle_y = 0.0

        # Mouse interaction
        self.mouse_pressed = False
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0

        # Point size
        self.point_size = 2.0  # Default point size
        gl.glPointSize(self.point_size)

        # Setup GL state
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.1, 0.1, 0.15, 1.0)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)  # Allow point size from shader or fixed

        # Vertex array object
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        # Vertex buffer object
        self.vbo = gl.glGenBuffers(1)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glEnableVertexAttribArray(0)
        # Each vertex: (x, y, z, intensity) -> 4 floats
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 16, None)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

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
            # Optional: Keyboard camera controls as a fallback.
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
            # Compute mouse deltas
            dx = xpos - self.last_mouse_x
            dy = ypos - self.last_mouse_y
            # Adjust camera angles based on mouse movement
            sensitivity = 0.005
            self.camera_angle_y += dx * sensitivity
            self.camera_angle_x += dy * sensitivity
            # Update last mouse position
            self.last_mouse_x, self.last_mouse_y = xpos, ypos
            print(f"Camera angles updated: X={self.camera_angle_x}, Y={self.camera_angle_y}")

    def _scroll_callback(self, window, xoffset, yoffset):
        # Zoom in/out with scroll wheel
        zoom_factor = 1.0 - yoffset * 0.1
        if zoom_factor <= 0.0:
            zoom_factor = 0.1
        self.camera_distance *= zoom_factor
        # Clamp camera_distance to avoid flipping through the origin
        self.camera_distance = max(self.camera_distance, 0.1)
        print(f"Camera distance adjusted to: {self.camera_distance}")

    def run(self):
        while not glfw.window_should_close(self.window):
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            # Compute MVP matrix
            width, height = glfw.get_framebuffer_size(self.window)
            aspect = width / float(height) if height > 0 else 1.0
            # Projection matrix: perspective
            fov = math.radians(60.0)
            znear, zfar = 0.1, 100.0
            f = 1.0 / math.tan(fov/2.0)
            proj = np.array([
                [f/aspect, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, (zfar+znear)/(znear - zfar), (2*zfar*znear)/(znear - zfar)],
                [0, 0, -1, 0]
            ], dtype=np.float32)

            # View matrix: rotate around X and Y, then translate back
            cx = self.camera_angle_x
            cy = self.camera_angle_y
            dist = self.camera_distance

            Rx = np.array([
                [1, 0,          0,           0],
                [0, math.cos(cx), -math.sin(cx), 0],
                [0, math.sin(cx), math.cos(cx),  0],
                [0, 0,          0,           1]
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

            gl.glUseProgram(self.program)
            gl.glUniformMatrix4fv(self.mvp_loc, 1, gl.GL_FALSE, mvp)

            gl.glBindVertexArray(self.vao)
            gl.glDrawArrays(gl.GL_POINTS, 0, self.num_points)
            gl.glBindVertexArray(0)

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()

