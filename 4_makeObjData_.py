class OBJModel:
    def __init__(self):
        self.vertices = []
        self.faces = []

    def read_from_file(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    parts = line.split()
                    vertex = (float(parts[1]), float(parts[2]), float(parts[3]))
                    self.vertices.append(vertex)
                elif line.startswith('f '):
                    parts = line.split()
                    face = tuple(int(part) - 1 for part in parts[1:])
                    self.faces.append(face)

    def save_to_file(self, filename):
        with open(filename, 'w') as file:
            for vertex in self.vertices:
                file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
            file.write('\n')
            for face in self.faces:
                file.write('f ' + ' '.join([str(v + 1) for v in face]) + '\n')

def main():
    input_file1 = 'output/8_3_occluded_faces_teapot_2.txt'
    output_file1 = 'output/8_3_occluded_faces_teapot_2.obj'

    input_file2 = 'output/8_3_output_removedData_teapot_2.txt'
    output_file2 = 'output/8_3_output_removedData_teapot_2.obj'

    obj_model = OBJModel()
    obj_model.read_from_file(input_file1)
    obj_model.save_to_file(output_file1)

    obj_model2 = OBJModel()
    obj_model2.read_from_file(input_file2)
    obj_model2.save_to_file(output_file2)

if __name__ == "__main__":
    main()
