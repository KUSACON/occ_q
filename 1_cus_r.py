from vectors import Vector
from OpenGL.GL import *
import numpy as np

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
