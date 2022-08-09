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
 
usage: 
  - create a mesh from an array of triangles or a .obj file
    Mesh(<tris>) or Mesh.load_from_file(<file>)
  - create an Object3D with this mesh
    Object3D(<mesh>, <position vector>, <rotation vector>, <scale vector>, <degrees>)
    if degrees is set to true, rotation will be converted from degrees to radians automatically
  - transform it in world space (Object3D.<rotate/translate/scale>)
  - create a camera (pretty much the same, it's a subclass of Object3D)
    Camera(<screen height>, <screen width>,
           <render distance zfar>, <closest distance znear>,
           <horisontal field of view>, <vertical fov>, <pos/rot/trans vectors>)
  - transform the camera in world space (Camera.<rotate/translate>)
  - give it a list of objects to render and a pygame surface
  - use Camera.render(<list of Object3D's>, <surface>)
 
 this project is not done yet, I'll clean it up and document it at some point :)
