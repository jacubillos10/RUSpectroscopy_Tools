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
        self.sample = sample(shape,system,limits,rho,C)
        self.base = base(N,type)

    def construct_E_from_red(self,E_int):
        """
        Matrix E constructor. Constructs the final matrix E from the reduced version of the matrix already integrated.

        @Input:
         E_int <np.array<np.array<int>>>:  A numpy array that holds a reduced version of the E matrix.

        @Output:
         E <np.array<np.array<int>>>: A numpy array that corresponds to the kinetic energy term of the Lagrangian 
         of the problem obtained with the volume integral of the product of the array phi with phi transposed 
         (phi: the expansion of the base functions represented with indexes). 
        """
        N = len(E_int)
        zero_aux = np.zeros([N])
        E = np.array([])
        for i in range(N):
            row = np.concatenate((E_int[i],zero_aux,zero_aux))
            if i == 0:
                E = np.hstack((E,row))
            else:
                E = np.vstack((E,row))
        for i in range(N):
            row = np.concatenate((zero_aux,E_int[i],zero_aux))
            E = np.vstack((E,row))
        for i in range(N):
            row = np.concatenate((zero_aux,zero_aux,E_int[i]))
            E = np.vstack((E,row))
        return E

    def calc_E(self):
        """
        Matrix E calculator. Calculates and returns the matrix E that corresponds the kinetic term of the Lagrangian of 
        the problem. 

        @Output:
         E <np.array<np.array<int>>>: A numpy array that corresponds to the kinetic energy term of the Lagrangian 
         of the problem obtained with the volume integral of the product of the array phi with phi transposed 
         (phi: the expansion of the base functions represented with indexes). 
        """
        base = self.base
        phi = base.get_phi()
        n1 = len(phi)
        E_red = np.array([])
        for i in range(n1):
            for j in range(n1):
                whole = np.concatenate((phi[i],phi[j]))
                if i == 0 and j == 0:
                    E_red = np.hstack((E_red,np.array(whole)))
                else:
                    E_red = np.vstack((E_red,np.array(whole)))

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
        
        E = self.construct_E_from_red(E_int)

        return E
    
    def calc_G(self):
        """
        Matrix G calculator. 

        @Output:
         G <np.array<np.array<int>>>: A numpy array that coresponds to the potential energy term of the Lagrangian 
         of the problem obtained with the volume integral of the 
        """
