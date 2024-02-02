import numpy as np


class Material:
    """
    Clase para representar un material colocado como muestra de RUS

    Atributos:
    N <int>: Número entero que representa el órden máximo de las potencias de las funciones base
    constantes <np.array>: Matriz de 6x6 que representa las ocnstantes elásticas del material
    estructura_cristalina <string>: Estructura cristalina del material. Algunos ejemplos son monoclínica, triclínica cúbica, etc.
    phi <np.array>: Matriz de (1/6)*(N+1)*(N+2)*(N+3) columnas y 3 filas que representa
        las funciones base para halla los desplazamientos u


    Métodos: 
    """
    def __init__(self, path, N, e_crist, encabezado=0):
        """
        Completar lo de la documentación y preguntarle a Julián si está decente!!! Luego si me pongo a pensar lo de las tuplas dentro de la matrices
        """
        self.constantes = np.genfromtxt(path, delimiter=',', skip_header=encabezado, dtype=float)
        self.estructura_cristalina = e_crist
        self.N = N
    # fin fucnion

    def inicializar_phi(self):
        posibles_indices = np.array(np.meshgrid(*[range(self.N + 1)] * 3)).T.reshape(-1, 3)
        self.phi = posibles_indices[posibles_indices.sum(axis=1) <= self.N].T
        self.phi = np.array(sorted(self.phi.T, key=lambda x: sum(x))).T
    # fin función

    def __str__(self):
        return str(self.phi)
    # fin funcion
       
# fin clase


if __name__ == '__main__':
    material_prueba = Material('constantes.csv', 2, 'monoclinica')
    material_prueba.inicializar_phi()
    print(material_prueba)
