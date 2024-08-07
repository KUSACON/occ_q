import pygame
from render import Camera, Object3D, Triangle, Mesh
from vectors import Vector
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# Window size
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
WINDOW_SURFACE = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE

BLACK = (0, 0, 0)
DARK_BLUE = (3, 5, 54)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# initialisation
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


# teapot_file = open('teapot.obj')
teapot_file = open('office_2.obj')
#teapot = Object3D(Mesh.load_from_file(teapot_file, correct_winding=False))

mesh = Mesh.load_from_file('office_2.obj')
# Set up an occlusion query
query = glGenQueries(1)

"""objects = [teapot]
lights = [Vector([0., 0., -1.])]
camera = Camera(WINDOW_HEIGHT, WINDOW_WIDTH, 1000., 0.1, 120., degrees=True)
camera.translate(Vector([0., 0., -10.]))"""

# Main Loop
mouse_down = False
clock = pygame.time.Clock()
running = True
while running:
    #window.fill(BLACK)
    # Handle user-input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        """elif event.type == pygame.WINDOWRESIZED:
            # canvas = pygame.Surface((window.get_width(), window.get_height()))
            # canvas.fill(BLACK)
            camera = Camera(window.get_height(), window.get_width(),
                            camera.zfar, camera.znear,
                            camera.fov_x, camera.fov_y,
                            pos=camera.translation,
                            rot=camera.rotation)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down = True
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            camera.translate_fps(Vector([0., 0., 0.1]))
        if keys[pygame.K_s]:
            camera.translate_fps(Vector([0., 0., -0.1]))
        if keys[pygame.K_SPACE]:
            camera.translate(Vector([0., 0.1, 0.]))
        if keys[pygame.K_LCTRL]:
            camera.translate(Vector([0., -0.1, 0.]))
        if keys[pygame.K_a]:
            camera.translate_fps(Vector([0.1, 0., 0.]))
        if keys[pygame.K_d]:
            camera.translate_fps(Vector([-0.1, 0., 0.]))
        if keys[pygame.K_f]:
            camera.rotate(Vector([0., 1., 0.]), degrees=True)
        if keys[pygame.K_h]:
            camera.rotate(Vector([0., -1, 0.]), degrees=True)
        if keys[pygame.K_g]:
            camera.rotate(Vector([1, 0, 0]), degrees=True)
        if keys[pygame.K_t]:
            camera.rotate(Vector([-1, 0, 0]), degrees=True)
        if keys[pygame.K_EQUALS]:
            objects[0].scale(Vector([1.1, 1.1, 1.1]))
        if keys[pygame.K_MINUS]:
            objects[0].scale(Vector([0.9, 0.9, 0.9]))
        if keys[pygame.K_z]:
            camera.reset()
            camera.translate(Vector([0., 0., -10.]))
            for obj in objects:
                obj.reset()"""

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0, 0, -10, 0, 0, 0, 0, 1, 0)

    # Start occlusion query
    glBeginQuery(GL_SAMPLES_PASSED, query)
    mesh.render()
    glEndQuery(GL_SAMPLES_PASSED)

    # Get query result
    samples_passed = GLuint(0)
    glGetQueryObjectuiv(query, GL_QUERY_RESULT, samples_passed)
    print(f'Samples passed: {samples_passed.value}')

    # Mouse Movement
    if mouse_down:
        mouse_pos = pygame.mouse.get_pos()
        # pygame.draw.circle(canvas, WHITE, mouse_pos, 5, 0)

    #camera.render(objects, lights, window)

    '''
    for tr in tris:
        for pt in tr.get_points():
            matr = camera.construct_projection_matrix()
            if len(pt) == 3:
                pt.append(1.)
            new_pt = matr * pt
            z = new_pt[3]
            if z:
                new_pt = Vector([x / z for x in new_pt])

            new_pt[0] += 1.
            new_pt[1] += 1.

            new_pt[0]cube *= 0.5 * WINDOW_WIDTH
            new_pt[1] *= 0.5 * WINDOW_HEIGHT


            cent = int(new_pt[0]), int(new_pt[1])
            pygame.draw.circle(window, WHITE, cent, 5)
    '''

    # Update the window
    # window.blit(canvas, (0, 0))
    pygame.display.flip()

    # objects[0].rotate(Vector([1., 1., 1.]), degrees=True)

    # Clamp FPS
    clock.tick(30)

# Save mesh data to a file
mesh.save_to_file('occlusion_result.txt')

pygame.quit()
