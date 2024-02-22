import numpy as np


class Material:
    """
    Class material: DEPRECATED!!! USE rus MODULE INSTEAD!!! Representation of elastic constants, frequencies and intermediate matrices such as E and Gamma in a solid material

    Attributes:
    N <int>: Integer that represents maximum order in the base funcions (power series or Legendre polynomials)
    C_matrix <np.array>: 6x6 symetric matrix with 21 elastic constants of the material
    cry_st <string>: Crystalline structure of the material.
    phi <np.array>: Matrix with (1/6)*(N+1)*(N+2)*(N+3) columns and 3 rows representing base functions to get the displacements u


    Méthods:

    __init__: Initialization method. Constants path and crystalline structure must be specified
    get_phi: Calculate the phi matrix based on the order N and the elastic constants
    __str__: Return a string that represents this class
    """
    def __init__(self, path, N, cryst_st, header=0):
        """
        @ Input: 
        path <string>: Path in which the text archive with the elastic constants is located
        N <int>: Desired order of the power series, Legendre polynomials or other base function
        cry_st <string>: Cystalline structure in a string without upper-case characters
        
        @ Output: None
        """
        self.C_matrix = np.genfromtxt(path, delimiter=',', skip_header=header, dtype=float)
        self.cry_st = cryst_st
        self.N = N
    # fin fucnion

    def get_phi(self):
        """
        This method calculates the phi matrix. Adds the matrix to the Material's respective attribute phi

        @ Input: None
        @ Output: None
        """
        possible_index_raw = np.array(np.meshgrid(*[range(self.N + 1)] * 3)).T.reshape(-1, 3)
        self.phi = possible_index_raw[possible_index_raw.sum(axis=1) <= self.N].T
        self.phi = np.array(sorted(self.phi.T, key=lambda x: sum(x))).T
    # fin función

    def __str__(self):
        """
        @ Input: None
        @ Output: 
        str(self.phi) <string>: numpy representation of phi matrix in a string format
        """
        return str(self.phi)
    # fin funcion
       
# fin clase


if __name__ == '__main__':
    material_prueba = Material('constantes.csv', 2, 'monoclinica')
    material_prueba.get_phi()
    print(material_prueba)
