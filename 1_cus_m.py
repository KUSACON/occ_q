import pygame
from render import Mesh
from OpenGL.GL import *
from OpenGL.GLU import *

# Window size
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
WINDOW_SURFACE = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

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
