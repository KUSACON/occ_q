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
        return Mesh(np.array(vertices, dtype=np.float32), faces)

    def render(self):
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            for vertex in face:
                glVertex3fv(self.vertices[vertex])
        glEnd()

    def save_to_file(self, filename):
        with open(filename, 'w') as file:
            #file.write('\nVertices:\n')
            for face in self.faces:
                for vertex in face:
                    v = self.vertices[vertex]
                    file.write(f'v {v[0]} {v[1]} {v[2]}\n')



# Initialisation
pygame.init()
pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("Occlusion Query with OpenGL")

# Initialize OpenGL settings
glEnable(GL_DEPTH_TEST)
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glShadeModel(GL_SMOOTH)

# Define a simple light source
glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, 1, 0))
glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))

# Load the object
mesh = Mesh.load_from_file('teapot.obj')

# Set up an occlusion query
query = glGenQueries(1)
query = query[0]  # Ensure query is an integer

# Camera settings
camera_pos = np.array([0, 0, -10], dtype=np.float32)
model_rot = np.array([0, 0, 0], dtype=np.float32)

# Main Loop
mouse_down = False
clock = pygame.time.Clock()
running = True
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

    # Model rotation
    if keys[pygame.K_i]:
        model_rot[0] += 2
    if keys[pygame.K_k]:
        model_rot[0] -= 2
    if keys[pygame.K_j]:
        model_rot[1] += 2
    if keys[pygame.K_l]:
        model_rot[1] -= 2

    # Save to file on 'P' key press
    if keys[pygame.K_p]:
        print("writing to file")
        mesh.save_to_file('2_occlusion_result_teapot.txt')

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(camera_pos[0], camera_pos[1], camera_pos[2], 0, 0, 0, 0, 1, 0)

    glPushMatrix()
    glRotatef(model_rot[0], 1, 0, 0)
    glRotatef(model_rot[1], 0, 1, 0)
    glRotatef(model_rot[2], 0, 0, 1)

    # Start occlusion query
    glBeginQuery(GL_SAMPLES_PASSED, query)
    mesh.render()
    glEndQuery(GL_SAMPLES_PASSED)

    # Get query result
    samples_passed = glGetQueryObjectuiv(query, GL_QUERY_RESULT)
    #print(f'Samples passed: {samples_passed}')

    glPopMatrix()

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
