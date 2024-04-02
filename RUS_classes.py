import numpy as np
import scipy.integrate as it 
import math
from sympy import *
import scipy.linalg as solve
from tqdm import tqdm

class base:
    """
    Class base: Abstract class containing the order of the base funcion (power series or Legendre for example) and array phi
    containing containing in each spot a tuple representing the order of x, y or z in each term.

    Attributes: 
     N <int>: Integer that represents maximum order in the base funcions.
     phi <np.array<np.array<int>>>: Numpy array with the expansion of the base functions represented with indexes in numpy arrays.
     type <string>: Name of the base functions. For example "Legendre" or "Poly" .

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
         phi<np.array<np.array<int>>>:Numpy array with the expansion of the base functions represented with indexes 
         in numpy arrays.
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
     W2 np.array<float>>: Numpy array with the eigenvalues of the problem organized in ascending order in a numpy array. This values 
     correspond phisically to the resonance frequencies multiplied by the density of the solid. 
     A np.array<np.array<float>>: Numpy array with the eigenvectors (in form of a numpy array aswell) of the problem. 
     The ith eigenvector corresponds to the vector associated to the ith eigen value in the organized w2 array. This vectors correspond 
     phisically to the ways it oscillates in terms of the directions (x,y,z).
     Amps np.array<float>: Numpy array with th amplitudes of the eigenvectors of the problem. Phisically, this values correspond to the 
     amplitude of the oscilations (amplitude of teh eigenvectors).

    Methods: 
     __init__: Initialization method. N, type, shape, system, limits, rho, C must be specified.
     calc_E: Method that calculates and does the volume integral for each of the values of the matrix E.
     calc_G: Method that calculates and does the volume integral for each of the values of the matrix Gamma.
     calc_eigenvals: Method that solves the generalized eigenvalue problem for the specified E, G and rho.
     construct_E_from_red: Constructs the final matrix E from the reduced version of the matrix already integrated.
     construct_full_phi: This function constructs the full matrix phi based on the reduced phi.
     derivation_PHI: This function derivates de phi matrix according to the derivation and cross product presented by Felipe 
     Giraldo in his thesis work.
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
        E = self.calc_E()
        G = self.calc_G()
        self.W2, self.A, self.Amps = self.calc_eigenvals(G,E)

    def construct_E_from_red(self,E_int):
        """
        Matrix E constructor. Constructs the final matrix E from the reduced version of the matrix already integrated.

        @Input:
         E_int <np.array<np.array<int>>>:  A numpy array that holds a reduced version of the E matrix.

        @Output:
         E <np.array<np.array<int>>>: A numpy array that corresponds to the kinetic energy term of the Lagrangian 
         of the problem obtained with the volume integral of the product of the array phi with phi transposed 
         (phi: The expansion of the base functions represented with indexes). 
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
    
    def construct_elem_E(self,elem1,elem2,limits):
        np.set_printoptions(linewidth=700)
        alfa = elem1[0]
        alfap = elem2[0]
        beta = elem1[1]
        betap = elem2[1]
        delta = elem1[2]
        deltap = elem2[2]

        exp1 = alfa+alfap+1
        exp2 = beta+betap+1
        exp3 = delta+deltap+1

        coef = (1 - (-1)**exp1)*(1 - (-1)**exp2)*(1 - (-1)**exp3)
        if  coef != 0 :
            element = (coef*(limits[0]**(exp1))*(limits[1]**(exp2))*(limits[2]**(exp3)))/((exp1)*(exp2)*(exp3))
        else: 
            element = 0

        return element
    
    def calc_E(self):
        """
        Matrix E calculator. Calculates and returns the matrix E that corresponds the kinetic term of the Lagrangian of 
        the problem. 

        @Output:
         E <np.array<np.array<int>>>: A numpy array that corresponds to the kinetic energy term of the Lagrangian 
         of the problem obtained with the volume integral of the product of the array phi with phi transposed 
         (phi: The expansion of the base functions represented with indexes). 
        """
        base = self.base
        sample = self.sample
        limits = sample.limits
        phi = base.phi
        R = len(phi)
        E_red = np.zeros((R,R))
        for i1 in range(3):
            for i2 in range(3):
                for p in range(R):
                    for q in range(R):
                        elem1 = phi[p]
                        elem2 = phi[q]
                        E_red[p,q] = self.construct_elem_E(elem1,elem2,limits)

        E = self.construct_E_from_red(E_red)

        return E
    
    
    def calc_G(self):
        """
        Matrix G calculator. 

        @Output:
         G <np.array<np.array<int>>>: A numpy array that coresponds to the potential energy term of the Lagrangian 
         of the problem obtained with the volume integral of the matrix product amongst Phi transposed, B transposed, 
         C, B, and Phi (Phi: The matrix with the expansion of the base functions represented with indexes, B: The 
         derivation matrix taken from Felipe Giraldo's thesis work, C: The elastic constants matrix).
        """

        np.set_printoptions(linewidth=700)
        base = self.base
        sample = self.sample
        C = sample.C
        limits = sample.limits
        phi = base.phi
        R = len(phi)  
        G = np.zeros((3*R,3*R)) 
        for i1 in tqdm(range(3)):
            for i2 in range(3):
                for p in range(R): 
                    for q in range(R):
                        elem1 = phi[p]
                        elem2 = phi[q]
                        G[i1*R+p,i2*R+q] = self.construct_elem_G(elem1,elem2,i1,i2,limits,C)
        return G
    
    def it(self,i,j):

        """
        This function changes the double index convention for single index convention in the elastic constant matrix C.
        00 is changed to 0, 11 is changed to 1, 22 is changed to 2, 
        01 or 10 is changed to 5
        02 or 20 is changed to 4
        12 or 21 is changed to 3

        @Input i <int>: first index to be changed
        @Input j <int>: second index to be changed 
        @Return index <int>: New index 
        """
        if i == j:
            index = i
        else:
            index = 6 - (i + j)
        return index

    def construct_elem_G(self, elem1, elem2, i1, i2, limits, C):
        
        element = 0
        for j1 in range(3):
            for j2 in range(3):

                elemj1 = np.zeros(3)
                elemj2 = np.zeros(3)
                elemj1[j1] = 1
                elemj2[j2] = 1

                exp1 = elem1[j1]
                exp2 = elem2[j2]

                i = self.it(i1,j1)
                j = self.it(i2,j2)

                exps = elem1 + elem2 + 1 - elemj1 - elemj2
                coef = (1-(-1)**exps[0])*(1-(-1)**exps[1])*(1-(-1)**exps[2])

                if coef != 0:

                    if j1 == j2:
                        full_exp = exp1 + exp2 - 2

                        if j1 == 0 :
                            element += (C[i,j]*(coef*exp1*exp2*(limits[0]**(full_exp+1))*(limits[1]**(elem1[1]+elem2[1]+1))*(limits[2]**(elem1[2]+elem2[2]+1)))/((full_exp+1)*(elem1[1]+elem2[1]+1)*(elem1[2]+elem2[2]+1)))
                        
                        elif j1 == 1 :
                            element += (C[i,j]*(coef*exp1*exp2*(limits[1]**(full_exp+1))*(limits[0]**(elem1[0]+elem2[0]+1))*(limits[2]**(elem1[2]+elem2[2]+1)))/((full_exp+1)*(elem1[0]+elem2[0]+1)*(elem1[2]+elem2[2]+1)))
                        
                        elif j1 == 2 : 
                            element += (C[i,j]*(coef*exp1*exp2*(limits[2]**(full_exp+1))*(limits[1]**(elem1[1]+elem2[1]+1))*(limits[0]**(elem1[0]+elem2[0]+1)))/((full_exp+1)*(elem1[1]+elem2[1]+1)*(elem1[0]+elem2[0]+1)))

                    else:

                        if j1 == 0: 

                            if j2 == 1:
                                element += (C[i,j]*(coef*exp1*exp2*(limits[0]**(exp1+elem2[0]))*(limits[1]**(elem1[1]+exp2))*(limits[2]**(elem1[2]+elem2[2]+1)))/((exp1+elem2[0])*(elem1[1]+exp2)*(elem1[2]+elem2[2]+1)))

                            else: 
                                element += (C[i,j]*(coef*exp1*exp2*(limits[0]**(exp1+elem2[0]))*(limits[2]**(elem1[2]+exp2))*(limits[1]**(elem1[1]+elem2[1]+1)))/((exp1+elem2[0])*(elem1[2]+exp2)*(elem1[1]+elem2[1]+1)))

                        elif j1 == 1:

                            if j2 == 0:
                                element += (C[i,j]*(coef*exp1*exp2*(limits[1]**(exp1+elem2[1]))*(limits[0]**(elem1[0]+exp2))*(limits[2]**(elem1[2]+elem2[2]+1)))/((exp1+elem2[1])*(elem1[0]+exp2)*(elem1[2]+elem2[2]+1)))

                            else: 
                                element += (C[i,j]*(coef*exp1*exp2*(limits[1]**(exp1+elem2[1]))*(limits[2]**(elem1[2]+exp2))*(limits[0]**(elem1[0]+elem2[0]+1)))/((exp1+elem2[1])*(elem1[2]+exp2)*(elem1[0]+elem2[0]+1)))

                        else:

                            if j2 == 0:
                                element += (C[i,j]*(coef*exp1*exp2*(limits[2]**(exp1+elem2[2]))*(limits[0]**(elem1[0]+exp2))*(limits[1]**(elem1[1]+elem2[1]+1)))/((exp1+elem2[2])*(elem1[0]+exp2)*(elem1[1]+elem2[1]+1)))

                            else: 
                                element += (C[i,j]*(coef*exp1*exp2*(limits[2]**(exp1+elem2[2]))*(limits[1]**(elem1[1]+exp2))*(limits[0]**(elem1[0]+elem2[0]+1)))/((exp1+elem2[2])*(elem1[1]+exp2)*(elem1[0]+elem2[0]+1)))

                else: 
                    element += 0

        return element

    def calc_eigenvals(self,G,E):
        """
        This function is incharge of calculating the eigenvalues, eigenvectors and amplitudes of each eigen- 
        vector for the generalized eigenvalue problem. In other words, this function calculates the resonance 
        frequencies of the solid multiplied by the density (eigenvalues), the ways it oscillates in terms of 
        the directions (x,y,z) (eigenvectors) and the amplitude of the oscilations. 

        @Input:
         G <np.array<np.array<int>>>: A numpy array that coresponds to the potential energy term of the Lagrangian 
         of the problem obtained with the volume integral of the matrix product amongst Phi transposed, B transposed, 
         C, B, and Phi (Phi: The matrix with the expansion of the base functions represented with indexes, B: The 
         derivation matrix taken from Felipe Giraldo's thesis work, C: The elastic constants matrix).
         E <np.array<np.array<int>>>: A numpy array that corresponds to the kinetic energy term of the Lagrangian 
         of the problem obtained with the volume integral of the product of the array phi with phi transposed 
         (phi: The expansion of the base functions represented with indexes).
        
         @Output:
         w2 np.array<float>>: Numpy array with the eigenvalues of the problem organized in ascending order in a numpy array. This values 
         correspond phisically to the resonance frequencies multiplied by the density of the solid. 
         a np.array<np.array<float>>: Numpy array with the eigenvectors (in form of a numpy array aswell) of the problem. 
         The ith eigenvector corresponds to the vector associated to the ith eigen value in the organized w2 array. This vectors correspond 
         phisically to the ways it oscillates in terms of the directions (x,y,z).
         amps np.array<float>: Numpy array with th amplitudes of the eigenvectors of the problem. Phisically, this values correspond to the 
         amplitude of the oscilations (amplitude of teh eigenvectors).
        """

        amps = np.array([])

        #Solving the generalized eigenvalue problem 
        w2 , a = solve.eigh(a=G,b=E)

        #Calculating the amplitudes
        for elem in a: 
            amp = solve.norm(elem)
            amps = np.append(amps,amp)
        
        return w2, a, amps