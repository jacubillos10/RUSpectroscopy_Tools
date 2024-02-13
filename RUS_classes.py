import numpy as np 

class base:
    """
    Class base: Abstract class containing the order of the base funcion (power series or Legendre for example) and array phi
    containing containing in each spot a tuple representing the order of x, y or z in each term.

    Attributes: 
     N <int>: Integer that represents maximum order in the base funcions.
     phi <np.array<tuples<int>>>: Numpy array with the expansion of the base functions represented with indexes in numpy arrays.
     type <string>: Name of the base functions. For example 'Legendre'.

    Methods: 
     __init__: Initialization method. N and type must be specified.
     calc_phi: Calculate the phi array based on the order N.
     get_type: Returns the user the name of the base functions that are being used.
     get_N: Returns the user the maximum order in the base funcions.
     get_phi: Returns the user the 
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

        return self

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
    
    def get_phi(self):
        """
        phi attribute class getter.
        """
        return self.phi

    def calc_phi(self):
        """
        Method incharge of calculating the phi array made of the index combinations for the specified N.

        @Output:
         phi<np.array<tuples<int>>>:Numpy array with the expansion of the base functions represented with indexes in tuples.
        """
        N = self.N
        nums = np.arange(0, N+1)
        i, j, k = np.broadcast_arrays(nums[:, None, None], nums[None, :, None], nums[None, None, :])        
        indices_filtrados = np.where(i + j + k <= N)
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
     rho <float>: Density of the sample.
     C <np.array<np.array<float>>>: Constant matrix of the sample.



    Methods: 
     __init__: Initialization method. shape, system, rho, C and limits must be specified.
     get_shape: Returns the user the name of the shape of the sample.
     get_system: Returns the user the name of the crystalline system of teh sample.
     get_limits: Returns the user the array with the limits (size) of the sample.
     get_rho: Returns the user the float that corresponds to the density of the sample.
     get_C: Returns the user the constant matrix that corresponds to the sample.

    """

    def __init__(self,shape,system,limits,rho,C):
        """
        Class initialization

        @Input:
         shape <str>: A string with the name of the shape of the sample.
         system <str>: A string with the name of the shape of the sample.
         limits <np.array<int>>: An array with all the information about the size of the sample. 
         rho <float>: A float that corresponds to the density of the sample.
        """
        self.shape = shape
        self.system = system
        self.limits = limits
        self.rho = rho
        self.C = C

        return self
    
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
    
    def get_rho(self):
        """
        Rho attribute class getter.
        """
        return self.rho
    
    def get_c(self):
        """
        C attribute class getter.
        """
        return self.C
    