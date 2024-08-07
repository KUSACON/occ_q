def read_occlusion_result(file_path):
    vertices = []
    faces = []
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        vertices_section = False
        for line in lines:
            line = line.strip()
            if line == "Vertices:":
                vertices_section = True
                continue
            if vertices_section:
                if line.startswith('v '):
                    vertices.append(line)
            else:
                if line.startswith('f '):
                    faces.append(line)
    """
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        vertices_section = False
        for line in lines:
            line = line.strip()
            if line.startswith('v '):
                vertices.append(line)
            if line.startswith('f '):
                faces.append(line)
    
    return vertices, faces

def write_obj_file(vertices, faces, output_path):
    with open(output_path, 'w') as file:
        file.write("# OBJ file generated from occlusion_result.txt\n")
        for vertex in vertices:
            file.write(vertex + '\n')
        for face in faces:
            file.write(face + '\n')

def main():
    
        
    filename = "combined_occluded_faces"
    input_file_path = 'output/' + filename + '.txt'
    output_file_path = 'output/out_' + filename + '.obj'
    vertices, faces = read_occlusion_result(input_file_path)
    #print(vertices)
    
    write_obj_file(vertices, faces, output_file_path)
    print(f"OBJ file saved as {output_file_path}")
    
    for x in range(6):
        filename = "occlusions_"+str(x)
        input_file_path = 'output/' + filename + '.txt'
        output_file_path = 'output/out_' + filename + '.obj'
        vertices, faces = read_occlusion_result(input_file_path)
        #print(vertices)
        
        write_obj_file(vertices, faces, output_file_path)
        print(f"OBJ file saved as {output_file_path}")

if __name__ == "__main__":
    main()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         