import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np

# Vertex Shader
vertex_shader = """
#version 330
in vec3 position;
void main()
{
    gl_Position = vec4(position, 1.0);
}
"""

# Fragment Shader
fragment_shader = """
#version 330
out vec4 FragColor;
void main()
{
    FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
"""

# Initialize Pygame and OpenGL
pygame.init()
pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("OpenGL Shader Example")

# Compile and link shaders
shader = compileProgram(
    compileShader(vertex_shader, GL_VERTEX_SHADER),
    compileShader(fragment_shader, GL_FRAGMENT_SHADER)
)

# Define a simple triangle
vertices = np.array([
    [0.0, 0.5, 0.0],
    [-0.5, -0.5, 0.0],
    [0.5, -0.5, 0.0]
], dtype=np.float32)

# Create a Vertex Buffer Object and upload the data
VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# Create a Vertex Array Object
VAO = glGenVertexArrays(1)
glBindVertexArray(VAO)

# Enable the position attribute
position = glGetAttribLocation(shader, 'position')
glEnableVertexAttribArray(position)
glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, None)

# Unbind the VAO
glBindVertexArray(0)

# Main Loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader)

    # Bind the VAO and draw the triangle
    glBindVertexArray(VAO)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    glBindVertexArray(0)

    pygame.display.flip()
    pygame.time.wait(10)

# Cleanup
glDeleteVertexArrays(1, [VAO])
glDeleteBuffers(1, [VBO])
glDeleteProgram(shader)
pygame.quit()
