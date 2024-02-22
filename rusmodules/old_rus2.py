import numpy as np

class IndexThree:
    def __init__(self,input):
        self.indices = np.array(input)

    def __getitem__(self, key):
        return self.indices[key]
    
    def __repr__(self):
        return str(tuple(self.indices))
    

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
        comb_matrix = possible_index_raw[possible_index_raw.sum(axis=1) <= self.N].T
        index_comb = np.array(list(map(lambda x: IndexThree(x), comb_matrix.T)), dtype = object)
        combinations = len(index_comb)
        v1 = np.r_[index_comb, np.zeros(2*combinations, dtype=int)]
        v2 = np.r_[np.zeros(combinations, dtype=int), index_comb, np.zeros(combinations, dtype=int)]
        v3 = np.r_[np.zeros(2*combinations, dtype=int), index_comb]
        self.phi = (np.c_[v1, v2, v3]).T
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
    print(base_test.phi[:,2][0][1])
