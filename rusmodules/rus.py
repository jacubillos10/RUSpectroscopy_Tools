import numpy as np
class Material:
    def __init__(self, path, N, e_crist, encabezado = 0):
        self.constantes = np.genfromtxt(path, delimiter = ',', skip_header = encabezado, dtype = float)
        self.estructura_cristalina = e_crist
        self.N = N
    #fin fucnion
    def inicializar_phi(self):
        posibles_indices = np.array(np.meshgrid(*[range(self.N + 1)] * 3)).T.reshape(-1, 3) 
        self.phi = posibles_indices[posibles_indices.sum(axis = 1) <= self.N].T 
        self.phi = np.array(sorted(self.phi.T, key = lambda x: sum(x))).T
    #fin función
    def __str__(self):
        return str(self.phi)
    #fin funcion
#fin clase

if __name__ == '__main__':
    material_prueba = Material('constantes.csv', 2, 'monoclinica')
    material_prueba.inicializar_phi()
    print(material_prueba)
