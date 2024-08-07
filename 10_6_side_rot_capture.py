import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np

# Window size
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_SURFACE = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Vertex Shader for unlit mode
vertex_shader_unlit = """
#version 330
layout(location = 0) in vec3 position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

# Fragment Shader for unlit mode
fragment_shader_unlit = """
#version 330
out vec4 FragColor;
void main()
{
    FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"""

# Vertex Shader for lit mode
vertex_shader_lit = """
#version 330
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 FragPos;
out vec3 Normal;
void main()
{
    FragPos = vec3(model * vec4(position, 1.0));
    Normal = mat3(transpose(inverse(model))) * normal;
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

# Fragment Shader for lit mode
fragment_shader_lit = """
#version 330
out vec4 FragColor;
in vec3 FragPos;
in vec3 Normal;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;
void main()
{
    // Ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
"""

def load_obj(filename):
    vertices = []
    normals = []
    faces = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('vn '):
                normals.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):
                face = [int(idx.split('/')[0]) - 1 for idx in line.strip().split()[1:]]
                faces.append(face)
    vertices = np.array(vertices, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32) if normals else np.zeros_like(vertices)
    faces = np.array(faces, dtype=np.uint32)
    print(f'Loaded {len(vertices)} vertices and {len(faces)} faces')
    return vertices, normals, faces

def create_shader_program(vertex_shader_src, fragment_shader_src):
    return compileProgram(
        compileShader(vertex_shader_src, GL_VERTEX_SHADER),
        compileShader(fragment_shader_src, GL_FRAGMENT_SHADER)
    )

def create_vao(vertices, normals, faces):
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    nbo = glGenBuffers(1)
    ebo = glGenBuffers(1)

    glBindVertexArray(vao)

    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, nbo)
    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)

    glBindVertexArray(0)
    return vao, len(faces) * 3

def check_gl_errors():
    error = glGetError()
    if error != GL_NO_ERROR:
        print(f"OpenGL error: {error}")

class Mesh:
    def __init__(self, vertices, normals, faces):
        self.vertices = vertices
        self.normals = normals
        self.faces = faces
        self.occluded_faces = []

    def render_face(self, face):
        glBegin(GL_TRIANGLES)
        for vertex in face:
            glVertex3fv(self.vertices[vertex])
        glEnd()

    def render(self):
        for face in self.faces:
            self.render_face(face)

    def save_to_file(self, filename):
        with open(filename, 'w') as file:
            for vertex in self.vertices:
                file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
            file.write('\n')
            for face in self.faces:
                file.write('f ' + ' '.join([str(v + 1) for v in face]) + '\n')

    def save_occluded_faces(self, filename):
        with open(filename, 'w') as file:
            for vertex in self.vertices:
                file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
            file.write('\n')
            for face in self.occluded_faces:
                file.write('f ' + ' '.join([str(v + 1) for v in face]) + '\n')

    def check_occlusion(self):
        self.occluded_faces = []
        for face in self.faces:
            query = glGenQueries(1)
            query = query[0]
            glBeginQuery(GL_SAMPLES_PASSED, query)
            self.render_face(face)
            glEndQuery(GL_SAMPLES_PASSED)

            samples_passed = glGetQueryObjectuiv(query, GL_QUERY_RESULT)
            if samples_passed == 0:
                self.occluded_faces.append(face)

            glDeleteQueries(1, [query])

    def remove_occluded_faces(self):
        face_tuples = [tuple(face) for face in self.faces]
        occluded_tuples = [tuple(face) for face in self.occluded_faces]
        self.faces = [face for face in face_tuples if face not in occluded_tuples]
        self.occluded_faces = []

    def common_occluded_faces(self, occlusions):
        from collections import Counter
        face_tuples = [tuple(face) for occlusion in occlusions for face in occlusion]
        common_faces = [face for face, count in Counter(face_tuples).items() if count == 6]
        self.occluded_faces = [list(face) for face in common_faces]

    def filter_faces_in_area(self, min_bound, max_bound):
        filtered_faces = []
        for face in self.faces:
            if all(np.all(min_bound <= self.vertices[vertex]) and np.all(self.vertices[vertex] <= max_bound) for vertex in face):
                filtered_faces.append(face)
        return filtered_faces

    def render_bounding_box(self, min_bound, max_bound):
        # Draw the bounding box as lines connecting the vertices
        vertices = [
            [min_bound[0], min_bound[1], min_bound[2]], [max_bound[0], min_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]], [max_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]], [max_bound[0], min_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]], [max_bound[0], max_bound[1], max_bound[2]]
        ]
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()

# Initialize Pygame and OpenGL
pygame.init()
pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("Occlusion Query")

# Load the object
vertices, normals, faces = load_obj('house2.obj')

# Scale the model to unit size
scale = 1.0 / np.max(np.linalg.norm(vertices, axis=1))
vertices *= scale

# Create VAO
vao, num_indices = create_vao(vertices, normals, faces)

# Load shaders
shader_unlit = create_shader_program(vertex_shader_unlit, fragment_shader_unlit)
shader_lit = create_shader_program(vertex_shader_lit, fragment_shader_lit)

current_shader = shader_unlit  # Start with unlit mode

# Create Mesh
mesh = Mesh(vertices, normals, faces)

# Camera settings
camera_pos = np.array([0, 0, 5], dtype=np.float32)
camera_rot = np.array([0, 0], dtype=np.float32)
model_rot = np.array([0, 0, 0], dtype=np.float32)
zoom = 1.0

# Define six view matrices for the cube
def get_view_matrices(camera_pos):
    return [
        np.array([
            [1, 0, 0, -camera_pos[0]],
            [0, 1, 0, -camera_pos[1]],
            [0, 0, 1, -camera_pos[2]],
            [0, 0, 0, 1]
        ], dtype=np.float32),    # Front
        np.array([
            [1, 0, 0, -camera_pos[0]],
            [0, 1, 0, -camera_pos[1]],
            [0, 0, -1, camera_pos[2]],
            [0, 0, 0, 1]
        ], dtype=np.float32),   # Back
        np.array([
            [0, 0, 1, -camera_pos[0]],
            [0, 1, 0, -camera_pos[1]],
            [-1, 0, 0, -camera_pos[2]],
            [0, 0, 0, 1]
        ], dtype=np.float32),    # Left
        np.array([
            [0, 0, -1, -camera_pos[0]],
            [0, 1, 0, -camera_pos[1]],
            [1, 0, 0, -camera_pos[2]],
            [0, 0, 0, 1]
        ], dtype=np.float32),   # Right
        np.array([
            [1, 0, 0, -camera_pos[0]],
            [0, 0, 1, -camera_pos[1]],
            [0, -1, 0, -camera_pos[2]],
            [0, 0, 0, 1]
        ], dtype=np.float32),    # Up
        np.array([
            [1, 0, 0, -camera_pos[0]],
            [0, 0, -1, -camera_pos[1]],
            [0, 1, 0, -camera_pos[2]],
            [0, 0, 0, 1]
        ], dtype=np.float32)    # Down
    ]

def capture_occlusions_from_positions(mesh, view_matrices):
    occlusions = []
    for view_matrix in view_matrices:
        glUniformMatrix4fv(glGetUniformLocation(current_shader, "view"), 1, GL_FALSE, view_matrix)
        mesh.check_occlusion()
        occlusions.append(mesh.occluded_faces.copy())
    return occlusions

def input_vector(prompt):
    try:
        return np.array(list(map(float, input(prompt).split())), dtype=np.float32)
    except ValueError:
        print("Invalid input. Please enter three floating-point numbers separated by spaces.")
        return input_vector(prompt)

# Get initial camera position
camera_pos = input_vector("Enter initial camera position (x y z): ")

# Main Loop
clock = pygame.time.Clock()
running = True
show_visible_triangles = False
bounding_box_set = False
min_bound = max_bound = None
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    # Camera movement
    moved = False
    if keys[pygame.K_w]:
        camera_pos[1] += 0.1
        moved = True
    if keys[pygame.K_s]:
        camera_pos[1] -= 0.1
        moved = True
    if keys[pygame.K_a]:
        camera_pos[0] -= 0.1
        moved = True
    if keys[pygame.K_d]:
        camera_pos[0] += 0.1
        moved = True
    if moved:
        print(f"Camera position: {camera_pos}")

    # Camera rotation
    if keys[pygame.K_UP]:
        camera_rot[1] += 2
    if keys[pygame.K_DOWN]:
        camera_rot[1] -= 2
    if keys[pygame.K_LEFT]:
        camera_rot[0] += 2
    if keys[pygame.K_RIGHT]:
        camera_rot[0] -= 2

    # Model rotation
    if keys[pygame.K_i]:
        model_rot[0] += 2
    if keys[pygame.K_k]:
        model_rot[0] -= 2
    if keys[pygame.K_j]:
        model_rot[1] += 2
    if keys[pygame.K_l]:
        model_rot[1] -= 2

    # Zoom
    if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
        zoom += 0.1
    if keys[pygame.K_MINUS] or keys[pygame.K_UNDERSCORE]:
        zoom -= 0.1

    # Toggle shader on '5' key press
    if keys[pygame.K_5]:
        print("Shading Changed")
        current_shader = shader_lit if current_shader == shader_unlit else shader_unlit
        pygame.time.wait(200)  # Add a small delay to avoid rapid toggling

    # Save to file on 'P' key press
    if keys[pygame.K_p]:
        print("writing to file")
        mesh.save_to_file('output/9_p_3_occlusion_result_office_2.txt')

    # Remove occluded faces on 'O' key press
    if keys[pygame.K_o]:
        print("occ test")
        mesh.check_occlusion()
        mesh.save_occluded_faces('output/9_o_4_occluded_faces_teapot_2.txt')
        mesh.remove_occluded_faces()
        mesh.save_to_file('output/9_o_4_output_removedData_teapot_2.txt')

    # Capture scene from six positions and find common occluded triangles on 'Y' key press
    if keys[pygame.K_y]:
        print("Capturing scene from six directions at the current position...")
        # Define the bounding box based on the current camera position
        bounding_box_set = True
        box_size = 2  # Define the size of the bounding box
        min_bound = camera_pos - box_size / 2
        max_bound = camera_pos + box_size / 2
        view_matrices = get_view_matrices(camera_pos)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(current_shader)
        mesh.render()  # Render the scene normally to populate the depth buffer
        filtered_faces = mesh.filter_faces_in_area(min_bound, max_bound)
        original_faces = mesh.faces
        mesh.faces = filtered_faces
        occlusions = capture_occlusions_from_positions(mesh, view_matrices)
        mesh.faces = original_faces
        for i, occ in enumerate(occlusions):
            mesh.occluded_faces = occ
            mesh.save_occluded_faces(f'output/occlusions_{i}.txt')
        mesh.common_occluded_faces(occlusions)
        mesh.save_occluded_faces('output/combined_occluded_faces.txt')
        print("Common occluded faces saved.")

    # Print camera positions on '6' key press
    if keys[pygame.K_6]:
        print(f"Camera position: {camera_pos}")

    # Toggle visible triangles count on 'V' key press
    if keys[pygame.K_v]:
        show_visible_triangles = not show_visible_triangles
        pygame.time.wait(200)  # Add a small delay to avoid rapid toggling

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(current_shader)

    # Set up camera
    view = np.identity(4, dtype=np.float32)
    view[3, :3] = -camera_pos  # Camera position
    view_loc = glGetUniformLocation(current_shader, "view")
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    # Set up projection
    projection = np.identity(4, dtype=np.float32)
    fov = 45.0 * zoom
    aspect = WINDOW_WIDTH / WINDOW_HEIGHT
    near = 0.1
    far = 100.0
    projection[0, 0] = 1 / (aspect * np.tan(np.radians(fov) / 2))
    projection[1, 1] = 1 / np.tan(np.radians(fov) / 2)
    projection[2, 2] = -(far + near) / (far - near)
    projection[2, 3] = -1
    projection[3, 2] = -(2 * far * near) / (far - near)
    projection_loc = glGetUniformLocation(current_shader, "projection")
    glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)

    # Set up model matrix
    model = np.identity(4, dtype=np.float32)
    model = np.dot(model, np.array([
        [np.cos(np.radians(model_rot[1])), 0, np.sin(np.radians(model_rot[1])), 0],
        [0, 1, 0, 0],
        [-np.sin(np.radians(model_rot[1])), 0, np.cos(np.radians(model_rot[1])), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32))
    model = np.dot(model, np.array([
        [1, 0, 0, 0],
        [0, np.cos(np.radians(model_rot[0])), -np.sin(np.radians(model_rot[0])), 0],
        [0, np.sin(np.radians(model_rot[0])), np.cos(np.radians(model_rot[0])), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32))
    model_loc = glGetUniformLocation(current_shader, "model")
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

    # Set up lighting for lit shader
    if current_shader == shader_lit:
        light_pos = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        object_color = np.array([1.0, 0.5, 0.31], dtype=np.float32)

        glUniform3fv(glGetUniformLocation(shader_lit, "lightPos"), 1, light_pos)
        glUniform3fv(glGetUniformLocation(shader_lit, "viewPos"), 1, camera_pos)
        glUniform3fv(glGetUniformLocation(shader_lit, "lightColor"), 1, light_color)
        glUniform3fv(glGetUniformLocation(shader_lit, "objectColor"), 1, object_color)

    # Draw the object
    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

    # Render the bounding box if set
    if bounding_box_set:
        glUseProgram(shader_unlit)  # Use the unlit shader for the bounding box
        mesh.render_bounding_box(min_bound, max_bound)

    # Check for OpenGL errors
    check_gl_errors()

    # Print number of visible triangles if toggled
    if show_visible_triangles:
        print(f"Visible triangles: {len(mesh.faces)}")

    pygame.display.flip()
    clock.tick(30)

# Cleanup
glDeleteVertexArrays(1, [vao])
glDeleteProgram(shader_unlit)
glDeleteProgram(shader_lit)
pygame.quit()
