from vectors import Vector, Matrix
from pygame import draw
from math import radians, sin, cos, tan
from helper import *
from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np


class Triangle:
    def __init__(self, p1: Vector, p2: Vector, p3: Vector):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    def __eq__(self, other):
        return self.get_points() == other.get_points()

    def get_points(self):
        return self.p1, self.p2, self.p3

    def get_edges(self):
        return (self.p1, self.p2), (self.p2, self.p3), (self.p3, self.p1)

    def flip(self):
        return Triangle(self.p3, self.p2, self.p1)

    def has_edge(self, edge, ordered=False):
        for ed in self.get_edges():
            if ordered:
                if ed == edge:
                    return True
            else:
                if (ed == edge) or (tuple(reversed(ed)) == edge):
                    return True
        return False

    def is_connected(self, other):
        for ed in other.get_edges():
            if self.has_edge(ed):
                return True
        return False

    def __repr__(self):
        return f'Triangle({self.p1}, {self.p2}, {self.p3})'

    def draw_wireframe(self, surface, color=WHITE):
        for ed in self.get_edges():
            p1, p2 = [to_2d_point(x) for x in ed]
            draw.line(surface, color, p1, p2)
        for pt in self.get_points():
            p2d = to_2d_point(pt)
            draw.circle(surface, color, p2d, 3)

    def draw_fill_color(self, surface, color=WHITE):
        draw.polygon(surface, color, [to_2d_point(x) for x in self.get_points()])

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


def fill_triangle(t: Triangle, surface, color=WHITE):
    pts = [to_2d_point(x) for x in t.get_points()]
    fill_triangle_2d(pts, surface, color)


def has_same_winding(t1: Triangle, t2: Triangle):
    if t1.is_connected(t2):
        for ed in t1.get_edges():
            if t2.has_edge(ed, ordered=True):
                return False
        return True
    else:
        raise ValueError("Triangles are not connected!")


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


class Object3D:
    def __init__(self, _mesh: Mesh = None, pos: Vector = None, rot: Vector = None, sc: Vector = None, degrees=False):
        if pos is None:
            pos = Vector(3, 0.)
        if rot is None:
            rot = Vector(3, 0.)
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
        return Object3D.construct_transform_matrix(self.rotation,
                                                   self.translation,
                                                   self.scaling)

    def to_worldspace(self):
        trans_m = self.get_transform_matrix()
        return self.multiply_by_matrix(trans_m)

    def multiply_by_matrix(self, trans_m: Matrix):
        return Object3D(self.mesh.multiply_by_matrix(trans_m))

    def transform(self, func):
        return Object3D(self.mesh.transform(func))

    def get_mesh(self):
        return self.mesh


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
                    # print(f'light: {lights[0]}, norm: {norm}, lum: {luminosity}')

                    tri = tri.multiply_by_matrix(self.construct_projection_matrix())  # Perspective projection
                    tri = tri.transform(self.scale_to_view)  # Scale to view

                    # print(tri)

                    tri.draw_fill_color(surface, (cl, cl, cl))
                    # tri.draw_wireframe(surface)
