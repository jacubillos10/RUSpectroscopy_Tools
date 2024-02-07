import numpy as np

class Base:
    """
    Class Base: Abstract class containing the order of the base funcion (power series or Legendre for example) and matrix phi
    containing containing in each spot a tuple representing the order of x, y or z in each term

    Attributes: 
     N <int>: Integer that represents maximum order in the base funcions 
     phi <np.array>: Matrix with (1/6)*(N+1)*(N+2)*(N+3) columns and 3 rows representing the index of the base functions
     type <string>: Name of the base functions. For example 'Legendre'

     Methods: 
     __init__: Initialization method. N must be specified
     get_phi: Calculate the phi matrix based on the order N
     __str__: Return a string that represents this class. In this case phi will be returned as a matrix of tuples
    """

    def __init__(self, N):
        """
        @Input:
        N <int>: Order of the base function
        """
        self.N = N
    #fin init

    def get_phi(self):
        """
        This method calculates the phi matrix.

        @ Input: None
        @ Output: None
        """
        possible_index_raw = np.array(np.meshgrid(*[range(self.N + 1)] * 3)).T.reshape(-1, 3)
        self.phi = possible_index_raw[possible_index_raw.sum(axis=1) <= self.N].T
        self.phi = np.array(sorted(self.phi.T, key=lambda x: x[0])).T
    # fin función

    def __str__(self):
        """
        @ Input: None
        @ Output: 
        str(self.phi) <string>: numpy representation of phi matrix in a string format
        """
        return str(self.phi)
    # fin funcion

#fin clase

if __name__ == '__main__':
    base_test = Base(2)
    base_test.get_phi()
    print(base_test)
