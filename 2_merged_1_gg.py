import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

# Window size
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
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
            for face in self.faces:
                file.write('f ' + ' '.join([str(v + 1) for v in face]) + '\n')
            file.write('\nVertices:\n')
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

# Main Loop
mouse_down = False
clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0, 0, -10, 0, 0, 0, 0, 1, 0)

    # Start occlusion query
    glBeginQuery(GL_SAMPLES_PASSED, query)
    mesh.render()
    glEndQuery(GL_SAMPLES_PASSED)

    # Get query result
    samples_passed = glGetQueryObjectuiv(query, GL_QUERY_RESULT)
    print(f'Samples passed: {samples_passed}')

    pygame.display.flip()
    clock.tick(30)

# Save mesh data to a file
mesh.save_to_file('occlusion_result.txt')

pygame.quit()
