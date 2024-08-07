import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

# Window size
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_SURFACE = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self.occluded_faces = []

    @staticmethod
    def load_from_file(filename):
        vertices = []
        faces = []
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    vertices.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith('f '):
                    faces.append([int(i.split('/')[0]) - 1 for i in line.strip().split()[1:]])
        print(f'Loaded {len(vertices)} vertices and {len(faces)} faces')
        return Mesh(np.array(vertices, dtype=np.float32), faces)

    def render_face(self, face):
        glBegin(GL_TRIANGLES)
        for vertex in face:
            glVertex3fv(self.vertices[vertex])
        glEnd()

    def render(self):
        for face in self.faces:
            self.render_face(face)

    def save_to_file(self, filename):
        with open(filename, 'w') as file:
            for vertex in self.vertices:
                file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
            file.write('\n')
            for face in self.faces:
                file.write('f ' + ' '.join([str(v + 1) for v in face]) + '\n')

    def check_occlusion(self):
        self.occluded_faces = []
        for face in self.faces:
            query = glGenQueries(1)
            query = query[0]  # Ensure query is an integer
            glBeginQuery(GL_SAMPLES_PASSED, query)
            self.render_face(face)
            glEndQuery(GL_SAMPLES_PASSED)

            samples_passed = glGetQueryObjectuiv(query, GL_QUERY_RESULT)
            if samples_passed == 0:
                self.occluded_faces.append(face)

            glDeleteQueries(1, [query])

    def remove_occluded_faces(self):
        self.faces = [face for face in self.faces if face not in self.occluded_faces]
        self.occluded_faces = []

def check_gl_errors():
    error = glGetError()
    if error != GL_NO_ERROR:
        print(f"OpenGL error: {error}")

# Initialisation
pygame.init()
pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("Occlusion Query with OpenGL")

# Initialize OpenGL settings
glEnable(GL_DEPTH_TEST)
glDisable(GL_LIGHTING)  # Disable lighting for debugging
glShadeModel(GL_SMOOTH)

# Load the object
mesh = Mesh.load_from_file('teapot.obj')

# Camera settings
camera_pos = np.array([0, 0, 10], dtype=np.float32)  # Adjusted for better visibility
camera_rot = np.array([0, 0], dtype=np.float32)  # Rotation around Y (yaw) and X (pitch) axes
model_rot = np.array([0, 0, 0], dtype=np.float32)
model_scale = 1.0  # Initial scale factor

# Main Loop
mouse_down = False
clock = pygame.time.Clock()
running = True
show_visible_triangles = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    # Camera movement
    if keys[pygame.K_w]:
        camera_pos[1] += 0.1
    if keys[pygame.K_s]:
        camera_pos[1] -= 0.1
    if keys[pygame.K_a]:
        camera_pos[0] -= 0.1
    if keys[pygame.K_d]:
        camera_pos[0] += 0.1

    # Camera rotation
    if keys[pygame.K_UP]:
        camera_rot[1] += 2
    if keys[pygame.K_DOWN]:
        camera_rot[1] -= 2
    if keys[pygame.K_LEFT]:
        camera_rot[0] += 2
    if keys[pygame.K_RIGHT]:
        camera_rot[0] -= 2

    # Model rotation
    if keys[pygame.K_i]:
        model_rot[0] += 2
    if keys[pygame.K_k]:
        model_rot[0] -= 2
    if keys[pygame.K_j]:
        model_rot[1] += 2
    if keys[pygame.K_l]:
        model_rot[1] -= 2

    # Model scaling
    if keys[pygame.K_EQUALS]:
        print("zoom in ")
        model_scale += 0.1
    if keys[pygame.K_MINUS]:
        print("zoom out")
        model_scale -= 0.1

    # Save to file on 'P' key press
    if keys[pygame.K_p]:
        print("writing to file")
        mesh.save_to_file('output/3__occlusion_result_teapot.txt')

    # Remove occluded faces on 'O' key press
    if keys[pygame.K_o]:
        print("occ test")
        mesh.check_occlusion()
        mesh.remove_occluded_faces()
        mesh.save_to_file('output/3_output_without_occluded_faces.txt')

    # Toggle visible triangles count on 'V' key press
    if keys[pygame.K_v]:
        show_visible_triangles = not show_visible_triangles
        pygame.time.wait(200)  # Add a small delay to avoid rapid toggling

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Calculate camera direction
    camera_direction = np.array([
        np.cos(np.radians(camera_rot[1])) * np.sin(np.radians(camera_rot[0])),
        np.sin(np.radians(camera_rot[1])),
        np.cos(np.radians(camera_rot[1])) * np.cos(np.radians(camera_rot[0]))
    ])

    # Calculate right and up vectors for the camera
    camera_right = np.array([
        np.sin(np.radians(camera_rot[0] - 90)),
        0,
        np.cos(np.radians(camera_rot[0] - 90))
    ])

    camera_up = np.cross(camera_right, camera_direction)

    camera_look_at = camera_pos + camera_direction

    gluLookAt(
        camera_pos[0], camera_pos[1], camera_pos[2],
        camera_look_at[0], camera_look_at[1], camera_look_at[2],
        camera_up[0], camera_up[1], camera_up[2]
    )

    glPushMatrix()
    glRotatef(model_rot[0], 1, 0, 0)
    glRotatef(model_rot[1], 0, 1, 0)
    glRotatef(model_rot[2], 0, 0, 1)
    glScalef(model_scale, model_scale, model_scale)

    # Set color to white for unlit rendering
    glColor3f(1.0, 1.0, 1.0)

    # Render the mesh
    visible_triangles = 0
    for face in mesh.faces:
        glBegin(GL_TRIANGLES)
        for vertex in face:
            glVertex3fv(mesh.vertices[vertex])
        glEnd()
        visible_triangles += 1

    glPopMatrix()

    # Add a simple cube for debugging purposes
    glPushMatrix()
    glColor3f(1.0, 0.0, 0.0)  # Red color for the cube
    #glutWireCube(2.0)  # Add a wireframe cube to ensure rendering is working
    glPopMatrix()

    # Check for OpenGL errors
    #print("Checking for OpenGL errors...")
    check_gl_errors()

    # Print number of visible triangles if toggled
    if show_visible_triangles:
        print(f"Visible triangles: {visible_triangles}")

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
