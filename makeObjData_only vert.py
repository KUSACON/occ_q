def read_vertices_from_file(input_file):
    vertices = []
    with open(input_file, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(line.strip())
    return vertices


def write_obj_file(output_file, vertices):
    with open(output_file, 'w') as file:
        # Write vertices
        for vertex in vertices:
            file.write(vertex + '\n')

        # Write faces
        num_vertices = len(vertices)
        for i in range(0, num_vertices, 3):
            if i + 2 < num_vertices:
                face = f"f {i + 1} {i + 2} {i + 3}"
                file.write(face + '\n')


def main():
    input_file = 'output/3_output_without_occluded_faces.txt'
    output_file = 'output/3_output_teapot_2.obj'

    vertices = read_vertices_from_file(input_file)
    write_obj_file(output_file, vertices)


if __name__ == "__main__":
    main()
