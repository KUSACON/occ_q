import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np

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
    faces = []
    normals = []
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

# Initialize Pygame and OpenGL
pygame.init()
pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("OpenGL 3D Object with Lighting")

# Load the object
vertices, normals, faces = load_obj('office_2.obj')

# Create VAO
vao, num_indices = create_vao(vertices, normals, faces)

# Load shaders
shader_unlit = create_shader_program(vertex_shader_unlit, fragment_shader_unlit)
shader_lit = create_shader_program(vertex_shader_lit, fragment_shader_lit)

current_shader = shader_unlit  # Start with unlit mode

# Camera settings
camera_pos = np.array([0, 0, 5], dtype=np.float32)
camera_rot = np.array([0, 0], dtype=np.float32)
model_rot = np.array([0, 0, 0], dtype=np.float32)
zoom = 1.0

# Main Loop
clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    # Camera movement
    if keys[pygame.K_w]:
        camera_pos[1] += 0.1
    if keys[pygame.K_s]:
        camera_pos[1] -= 0.1
    if keys[pygame.K_a]:
        camera_pos[0] -= 0.1
    if keys[pygame.K_d]:
        camera_pos[0] += 0.1

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
        current_shader = shader_lit if current_shader == shader_unlit else shader_unlit
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
    aspect = 800 / 600
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

    pygame.display.flip()
    clock.tick(60)

# Cleanup
glDeleteVertexArrays(1, [vao])
glDeleteProgram(shader_unlit)
glDeleteProgram(shader_lit)
pygame.quit()
