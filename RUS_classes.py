import numpy as np 

class base:
    """
    Class base: Abstract class containing the order of the base funcion (power series or Legendre for example) and array phi
    containing containing in each spot a tuple representing the order of x, y or z in each term.

    Attributes: 
     N <int>: Integer that represents maximum order in the base funcions.
     phi <np.array<tuples<int>>>: Numpy array with the expansion of the base functions represented with indexes in tuples.
     type <string>: Name of the base functions. For example 'Legendre'.

    Methods: 
     __init__: Initialization method. N and type must be specified.
     calc_phi: Calculate the phi array based on the order N.
     get_type: Returns the user the name of the base functions that are being used.
     get_N: Returns the user the maximum order in the base funcions.
    """

    def __init__(self,N,type):
        """
        Class initialization
        @Input:
         N <int>: Max order of the base function.
         type <str>: Name of the base functions.
        """
        self.N = N
        self.type = type 
        self.phi = self.calc_phi(self)

    def get_type(self):
        """
        Type attribute class getter.
        """
        return self.type
    
    def get_N(self):
        """
        N attribute class getter.
        """
        return self.N

    def calc_phi(self):
        """
        Method incharge of calculating the phi array made of the index combinations for the specified N.
        @Output:
         phi<np.array<tuples<int>>>:Numpy array with the expansion of the base functions represented with indexes in tuples.
        """
        nums = np.arange(0, self.N+1)
        i, j, k = np.broadcast_arrays(nums[:, None, None], nums[None, :, None], nums[None, None, :])        filtered = indices[np.sum(indices, axis=1) <= self.N]
        indices_filtrados = np.where(i + j + k <= self.N)
        phi = np.vstack([i[indices_filtrados], j[indices_filtrados], k[indices_filtrados]]).T
        return phi

    
class sample:
    """
    Class sample: Abstract class incharge of describing the nature of the sample that's being taken into account to solve the 
    forward problem.

    Attributes:
     shape <str>: A string with the name of the shape of the sample. For example "cilinder" or "rectangular paralelepiped".
     system <str>: A string with the name of the shape of the sample. For example "orthorombic" or "isotropic".
     limits <np.array<int>>: An array with all the information about the size of the sample. 

    Methods: 
     __init__: Initialization method. shape, system and limits must be specified.
     get_shape: Returns the user the name of the shape of the sample.
     get_system: Returns the user the name of the crystalline system of teh sample.
     get_limits: Returns the user the array with the limits (size) of the sample.
    """

    def __init__(self,shape,system,limits):
        """
        Class initialization
        @Input:
         shape <str>: A string with the name of the shape of the sample.
         system <str>: A string with the name of the shape of the sample.
         limits <np.array<int>>: An array with all the information about the size of the sample. 
        """
        self.shape = shape
        self.system = system
        self.limits = limits
    
    def get_shape(self):
        """
        Shape attribute class getter.
        """
        return self.shape
    
    def get_system(self):
        """
        System attribute class getter.
        """
        return self.system
    
    def get_limits(self):
        """
        Limits attribute class getter.
        """
        return self.limits