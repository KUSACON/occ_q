# pygame-3d-render
A simple software 3D renderer made with pygame + a module to work with vectors/matrices.
Made as a learning project.

vectors.py defines classes for a Vector and a Matrix that define some common operations with them 
(addition, dot and cross product, matrix determinant, etc)
render.py defines classes for:
  - Triangle (stores 3 vectors, which are vertices, and has some helper functions)
  - Mesh (collection of triangles + functions to transform them + can load data from obj file)
  - Object3D (an object with a translation, rotation and scale in world space and optionally a mesh to render)
  - Camera (a 3D object that handles rendering)
 main.py is the main file
 examples.py has a cube model
 
 this project is not done yet, I'll clean it up at some point :)
