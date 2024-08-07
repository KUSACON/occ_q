import pygame
from math import radians, sin, cos, tan

from helper import vector3_by_matrix4

# Constants
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
WINDOW_SURFACE = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Helper functions
def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

def to_2d_point(vec):
    return int(vec[0]), int(vec[1])

# Vector and Matrix classes
class Vector(list):
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], int):
            super().__init__([0.0] * args[0])
        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            super().__init__(args[0])
        elif len(args) > 1:
            super().__init__(args)
        else:
            raise TypeError("Invalid arguments for Vector")

    def __add__(self, other):
        return Vector([x + y for x, y in zip(self, other)])

    def __sub__(self, other):
        return Vector([x - y for x, y in zip(self, other)])

    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector([x * y for x, y in zip(self, other)])
        return Vector([x * other for x in self])

    def __truediv__(self, other):
        return Vector([x / other for x in self])

    def __repr__(self):
        return f"Vector({super().__repr__()})"

    def normalize(self):
        mag = self.mag()
        if mag == 0:
            return self
        return self / mag

    def mag(self):
        return sum(x**2 for x in self) ** 0.5

class Matrix(list):
    def __init__(self, rows):
        super().__init__(rows)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector([sum(a * b for a, b in zip(row, other)) for row in self])
        elif isinstance(other, Matrix):
            return Matrix([[sum(a * b for a, b in zip(row, col)) for col in zip(*other)] for row in self])

# Triangle class
class Triangle:
    def __init__(self, p1: Vector, p2: Vector, p3: Vector):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    def get_points(self):
        return self.p1, self.p2, self.p3

    def get_edges(self):
        return (self.p1, self.p2), (self.p2, self.p3), (self.p3, self.p1)

    def draw_wireframe(self, surface, color=WHITE):
        for ed in self.get_edges():
            p1, p2 = [to_2d_point(x) for x in ed]
            pygame.draw.line(surface, color, p1, p2)
        for pt in self.get_points():
            p2d = to_2d_point(pt)
            pygame.draw.circle(surface, color, p2d, 3)

    def draw_fill_color(self, surface, color=WHITE):
        pygame.draw.polygon(surface, color, [to_2d_point(x) for x in self.get_points()])

    def get_normal(self):
        v1 = Vector((self.p2 - self.p1)[:3])
        v2 = Vector((self.p3 - self.p1)[:3])
        normal = v1 * v2
        magnitude = normal.mag()
        if magnitude == 0:
            return normal  # or return a default normal, e.g., Vector([0, 0, 1])
        return normal.normalize()

    def multiply_by_matrix(self, trans_m: Matrix):
        new_pts = []
        for pt in self.get_points():
            if len(pt) == 3:
                pt.append(1.)
            new_pt = trans_m * pt
            if new_pt[3]:
                new_pt = new_pt / new_pt[3]
            new_pts.append(new_pt)
        return Triangle(*new_pts)

    def transform(self, func):
        new_pts = []
        for pt in self.get_points():
            new_pt = func(pt)
            new_pts.append(new_pt)
        return Triangle(*new_pts)

# Mesh class
class Mesh:
    def __init__(self, _tris: list, correct_winding=True, wireframe_color=WHITE):
        self.default_wireframe_color = wireframe_color
        if not correct_winding:
            self.tris = _tris
        else:
            self.tris = []
            if _tris:
                to_process = [_tris.pop(0)]
                while to_process:
                    current_tr = to_process.pop()
                    i = 0
                    while i < len(_tris):
                        if current_tr.is_connected(_tris[i]):
                            if has_same_winding(current_tr, _tris[i]):
                                to_process.append(_tris.pop(i))
                            else:
                                to_process.append(_tris.pop(i).flip())
                        else:
                            i += 1
                    self.tris.append(current_tr)

    @staticmethod
    def load_from_file(file, correct_winding=True, wireframe_color=WHITE):
        verts = []
        tris = []
        for line in file.readlines():
            ln = line.strip('\n').split(' ')
            if ln[0] == 'v':
                coords = [float(x) for x in ln[1:]]
                verts.append(Vector(coords))
            elif ln[0] == 'f':
                inds = [x.split('/')[0] for x in ln[1:]]  # Modify this line to handle '1/1/1' format
                inds = [int(x) - 1 for x in inds]
                pts = [verts[i] for i in inds]
                tris.append(Triangle(*pts))
        return Mesh(tris, correct_winding=correct_winding, wireframe_color=wireframe_color)

    def get_tris(self):
        return self.tris

    def multiply_by_matrix(self, trans_m: Matrix):
        new_tris = []
        for tri in self.get_tris():
            new_tris.append(tri.multiply_by_matrix(trans_m))
        return Mesh(new_tris, correct_winding=False, wireframe_color=self.default_wireframe_color)

    def transform(self, func):
        new_tris = []
        for tri in self.get_tris():
            new_tris.append(tri.transform(func))
        return Mesh(new_tris, correct_winding=False, wireframe_color=self.default_wireframe_color)

# Object3D class
class Object3D:
    def __init__(self, _mesh: Mesh = None, pos: Vector = None, rot: Vector = None, sc: Vector = None, degrees=False):
        if pos is None:
            pos = Vector(3)
        if rot is None:
            rot = Vector(3)
        if sc is None:
            sc = Vector(3, 1.)
        if degrees:
            rot = Vector([radians(x) for x in rot])
        self.mesh = _mesh
        self.translation, self.rotation, self.scaling = pos, rot, sc

    @staticmethod
    def construct_rotation_matrix(rotation: Vector, degrees=False):
        if degrees:
            rotation = Vector([radians(x) for x in rotation])
        rx, ry, rz, *_ = rotation
        rot_x = Matrix([[1., 0., 0., 0.],
                        [0., cos(rx), -1. * sin(rx), 0.],
                        [0., sin(rx), cos(rx), 0.],
                        [0., 0., 0., 1.]])
        rot_y = Matrix([[cos(ry), 0., sin(ry), 0.],
                        [0., 1., 0., 0.],
                        [-1. * sin(ry), 0., cos(ry), 0.],
                        [0., 0., 0., 1.]])
        rot_z = Matrix([[cos(rz), -1. * sin(rz), 0., 0.],
                        [sin(rz), cos(rz), 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]])
        return rot_z * rot_y * rot_x

    @staticmethod
    def construct_translation_matrix(translation: Vector):
        return Matrix([[1., 0., 0., translation[0]],
                       [0., 1., 0., translation[1]],
                       [0., 0., 1., translation[2]],
                       [0., 0., 0., 1.]])

    @staticmethod
    def construct_scale_matrix(scale: Vector):
        return Matrix([[scale[0], 0., 0., 0.],
                       [0., scale[1], 0., 0.],
                       [0., 0., scale[2], 0.],
                       [0., 0., 0., 1.]])

    @staticmethod
    def construct_transform_matrix(rotation: Vector = None, translation: Vector = None, scale: Vector = None, deg=False):
        if rotation is None:
            deg = False
            rotation = Vector(3, 0.)
        if translation is None:
            translation = Vector(3, 0.)
        if scale is None:
            scale = Vector(3, 1.)
        rot_m = Object3D.construct_rotation_matrix(rotation, degrees=deg)
        trans_m = Object3D.construct_translation_matrix(translation)
        scale_m = Object3D.construct_scale_matrix(scale)
        return scale_m * trans_m * rot_m

    def rotate(self, rotation: Vector, degrees=False):
        if degrees:
            rotation = Vector([radians(x) for x in rotation])
        self.rotation += rotation

    def translate(self, translation: Vector):
        self.translation += translation

    def scale(self, scaling: Vector):
        for i, new_sc in enumerate(scaling):
            self.scaling[i] = self.scaling[i] * new_sc

    def reset(self):
        self.translation = Vector(3, 0.)
        self.rotation = Vector(3, 0.)
        self.scaling = Vector(3, 1.)

    def get_transform_matrix(self):
        return Object3D.construct_transform_matrix(self.rotation, self.translation, self.scaling)

    def to_worldspace(self):
        trans_m = self.get_transform_matrix()
        return self.multiply_by_matrix(trans_m)

    def multiply_by_matrix(self, trans_m: Matrix):
        return Object3D(self.mesh.multiply_by_matrix(trans_m))

    def transform(self, func):
        return Object3D(self.mesh.transform(func))

    def get_mesh(self):
        return self.mesh

# Camera class
class Camera(Object3D):
    def __init__(self, h, w, zfar, znear, fov_x, fov_y=None, pos: Vector = None, rot: Vector = None, degrees=False):
        super().__init__(pos=pos, rot=rot, degrees=degrees)
        if fov_y is None:
            fov_y = fov_x
        if degrees:
            fov_x, fov_y = radians(fov_x), radians(fov_y)
        self.height = h
        self.width = w
        self.zfar = zfar
        self.znear = znear
        self.fov_x = fov_x
        self.fov_y = fov_y

    def aspect_ratio(self):
        return self.height / self.width

    def construct_projection_matrix(self):
        a = self.aspect_ratio()
        fx = 1. / tan(self.fov_x / 2)
        fy = 1. / tan(self.fov_y / 2)
        zn = self.zfar / (self.zfar - self.znear)
        return Matrix([[a * fx, 0., 0., 0.],
                       [0., fy, 0., 0.],
                       [0., 0., zn, 1.],
                       [0., 0., -1. * zn * self.znear, 0.]])

    def scale_to_view(self, vec):
        return Vector([(vec[0] + 1.) * 0.5 * self.width, (vec[1] + 1.) * 0.5 * self.height, vec[2]])

    def translate_fps(self, translation):
        rot_m = self.construct_rotation_matrix(self.rotation)
        translation = vector3_by_matrix4(translation, rot_m)
        self.translation += translation

    def render(self, objects, lights, surface):
        camera_transform_m = self.get_transform_matrix().inv()
        lights = [x.normalize() for x in lights]
        for obj in objects:
            obj = obj.to_worldspace()  # Move into world space
            # obj = obj.multiply_by_matrix(camera_transform_m)  # Move into view space

            for tri in obj.get_mesh().get_tris():
                old_norm = tri.get_normal()
                tri = tri.multiply_by_matrix(camera_transform_m)
                norm = tri.get_normal()
                if norm[2] < 0.:
                    luminosity = sum([old_norm ^ x for x in lights]) / len(lights)
                    cl = clamp(int(255 * luminosity), 0, 255)
                    tri = tri.multiply_by_matrix(self.construct_projection_matrix())  # Perspective projection
                    tri = tri.transform(self.scale_to_view)  # Scale to view
                    tri.draw_fill_color(surface, (cl, cl, cl))

# Main Program
# initialisation
pygame.init()
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), WINDOW_SURFACE)
pygame.display.set_caption("Bad 3D Renderer")

# Load object
teapot_file = open('office_2.obj')
teapot = Object3D(Mesh.load_from_file(teapot_file, correct_winding=False))

objects = [teapot]
lights = [Vector([0., 0., -1.])]
camera = Camera(WINDOW_HEIGHT, WINDOW_WIDTH, 1000., 0.1, 120., degrees=True)
camera.translate(Vector([0., 0., -10.]))

# Main Loop
mouse_down = False
clock = pygame.time.Clock()
running = True
while running:
    window.fill(BLACK)
    # Handle user-input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.WINDOWRESIZED:
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
                obj.reset()

    # Render the scene
    camera.render(objects, lights, window)
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
