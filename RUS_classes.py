import numpy as np
import scipy.integrate as it 
import math
from sympy import *
 
"""
def P(n):  #Rodriguez formula for legendre polynomials
    x = symbols('x')
    y = symbols('y')
    y = (x**2 - 1)**n
    pol = diff(y,x,n)/(2**n * math.factorial(n)) 
    return pol
"""


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
     get_phi: Returns the user the phi array. 
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
        self.phi = self.calc_phi()


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
        type = self.type
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
        Class initialization.

        @Input:
         shape <str>: A string with the name of the shape of the sample.
         system <str>: A string with the name of the shape of the sample.
         limits <np.array<int>>: An array with all the information about the size of the sample. 
         rho <float>: A float that corresponds to the density of the sample.
         C <np.array<np.array<float>>>: A numpy array matrix that holds the value of the elastic constants of the sample.
        """
        self.shape = shape
        self.system = system
        self.limits = limits
        self.rho = rho
        self.C = C
    
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
    
class Forward: 
    """
    Class Forward: Class that connects the other 2 classes and calculates de eigen values for the forward problem in resonant
    ultrasound spectroscopy.
    
    Attributes: 
     Sample <sample>: Object of type <sample> that holds all of the information needed from the sample to solve the problem.
     Base <base>: Object of type <base> that holds all of teh information needed form the base functions to solve the problem.

    Methods: 
     __init__: Initialization method. N, type, shape, system, limits, rho, C must be specified.
     calc_E: Method that calculates and does the volume integral for each of the values of the matrix E.
     calc_G: Method that calculates and does the volume integral for each of the values of the matrix Gamma.
     calc_eigenvals: Method that solves the generalized eigenvalue problem for the specified E, G and rho.
    """
    def __init__(self,N,type,shape,system,limits,rho,C):
        """
        Class initialization.

        @Input:
         N <int>: Max order of the base function.
         type <str>: Name of the base functions.
         shape <str>: A string with the name of the shape of the sample.
         system <str>: A string with the name of the shape of the sample.
         limits <np.array<int>>: An array with all the information about the size of the sample. 
         rho <float>: A float that corresponds to the density of the sample.
         C <np.array<np.array<float>>>: A numpy array matrix that holds the value of the elastic constants of the sample.
        """
        Sample = sample(shape,system,limits,rho,C)
        Base = base(N,type)

    def calc_E(self,Base):
        """
        Matrix E calculator. 

        @Input:
         Base <base>: Object of type <base> that holds all of teh information needed form the base functions to solve the problem.

        @Output:
         E <np.array<np.array<int>>>: A numpy array that holds the information of the volume integral for the 
         product of phi with phi trasposed (phi: the expansion of the base functions represented with indexes). 
        """
        phi = Base.get_phi()
        n1 = len(phi)
        E_red = np.array([])
        for i in range(n1):
            for j in range(n1):
                whole = np.concatenate((phi[i],phi[j]))
                if i == 0 and j == 0:
                    E_red = np.hstack((E,np.array(whole)))
                else:
                    E_red = np.vstack((E,np.array(whole)))

        n = len(E_red)
        n_i = len(E_red[0])
        E_int = np.array([])
        row = np.array([])

        for i in range(n):
            exps = []
            for j in range(n_i):
                exps.append(E[i,j])
            f = lambda x,y,z: x**(exps[0]+exps[3]) * y**(exps[1]+exps[4]) * z**(exps[2]+exps[5])
            result = it.tplquad(f,-1,1,-1,1,-1,1)
            row = np.append(row,result[0])
            if len(row) == n1 and i == n1 - 1:
                E_int = np.hstack((E_int,row))
                row = np.delete(row,[range(n1)])
            elif len(row) == n1:
                E_int = np.vstack((E_int,row))
                row = np.delete(row,[range(n1)])

        return E_int
    
    def construct_E_from_red(E_int):
        """
        Matrix E constructor. Constructs the final matrix E from the reduced version of the matrix already integrated.

        @Input:
         E_int <np.array<np.array<int>>>:  A numpy array that holds a reduced version of the information of the volume integral 
         for the product of phi with phi trasposed (phi: the expansion of the base functions represented with indexes).

        @Output:
         E <np.array<np.array<int>>>: A numpy array that holds the information of the volume integral for the 
         product of phi with phi trasposed (phi: the expansion of the base functions represented with indexes). 
        """
        N = len(E_int)
        zero_aux = np.zeros([N,N])
        #for i in range(N):




        



