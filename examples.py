from vectors import Vector
from render import Triangle, Mesh, Object3D

CUBE_VERTS = [[Vector([0., 0., 0.]), Vector([0., 1., 0.]), Vector([1., 1., 0.])],
              [Vector([0., 0., 0.]), Vector([1., 1., 0.]), Vector([1., 0., 0.])],

              [Vector([1., 0., 0.]), Vector([1., 1., 0.]), Vector([1., 1., 1.])],
              [Vector([1., 0., 0.]), Vector([1., 1., 1.]), Vector([1., 0., 1.])],

              [Vector([1., 0., 1.]), Vector([1., 1., 1.]), Vector([0., 1., 1.])],
              [Vector([1., 0., 1.]), Vector([0., 1., 1.]), Vector([0., 0., 1.])],

              [Vector([0., 0., 1.]), Vector([0., 1., 1.]), Vector([0., 1., 0.])],
              [Vector([0., 0., 1.]), Vector([0., 1., 0.]), Vector([0., 0., 0.])],

              [Vector([0., 1., 0.]), Vector([0., 1., 1.]), Vector([1., 1., 1.])],
              [Vector([0., 1., 0.]), Vector([1., 1., 1.]), Vector([1., 1., 0.])],

              [Vector([1., 0., 1.]), Vector([0., 0., 1.]), Vector([0., 0., 0.])],
              [Vector([1., 0., 1.]), Vector([0., 0., 0.]), Vector([1., 0., 0.])]]

CUBE_TRIS = [Triangle(*x) for x in CUBE_VERTS]
CUBE_MESH = Mesh(CUBE_TRIS)
CUBE = Object3D(CUBE_MESH)
