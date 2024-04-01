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

    def calc_E(self):
        """
        Matrix E calculator. Calculates and returns the matrix E that corresponds the kinetic term of the Lagrangian of 
        the problem. 

        @Output:
         E <np.array<np.array<int>>>: A numpy array that corresponds to the kinetic energy term of the Lagrangian 
         of the problem obtained with the volume integral of the product of the array phi with phi transposed 
         (phi: The expansion of the base functions represented with indexes). 
        """

        #elevant parameter
        base = self.base
        sample = self.sample
        limits = sample.limits
        phi = base.phi
        n1 = len(phi)
        E_red = np.array([])

        #Phi*Phit
        for i in tqdm(range(n1)):
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

        #Integration (Only valid for rectangular parallelepipeids)
        for i in tqdm(range(n)):
            exps = []
            for j in range(n_i):
                exps.append(E_red[i,j])

            p = exps[0]+exps[3]
            q = exps[1]+exps[4]
            r = exps[2]+exps[5]

            result = (8*(limits[0]**(p+1))*(limits[1]**(q+1))*(limits[2]**(r+1)))/((p+1)*(q+1)*(r+1))
            
            row = np.append(row,result)

            if len(row) == n1 and i == n1 - 1:
                E_int = np.hstack((E_int,row))
                row = np.delete(row,[range(n1)])

            elif len(row) == n1:
                E_int = np.vstack((E_int,row))
                row = np.delete(row,[range(n1)])
        
        #Full matrix E construction
        E = self.construct_E_from_red(E_int)

        return E
    
    def construct_full_phi(self,phi):
        """
        This function constructs the full matrix phi based on the reduced phi to make easier the calculation of the Gamma 
        matrix. It stacks a phi for each posible direction (x,y,z) and zeros for the other directions on the row. Each row and
        "block" column of the same length of phi corresponds to a single direction so the full phi matrix is diagonal by blocks
        with elements only in the directions xx, yy and zz. 

        @Input:
         phi<np.array<np.array<int>>>:Numpy array with the expansion of the base functions represented with indexes in 
         numpy arrays.
        
        @Output: 
         PHI<np.array<np.array<np.array<int>>>>:Numpy array with the expansion of the base functions represented with indexes in 
         numpy arrays. It has the expressions for the three directions (x,y,z). Matrix diagonal in blocks such as: [[phi][0][0],
         [0][phi][0],[0][0][phi]].

        """
        N = len(phi)
        n = len(phi[0])

        zero_aux = np.zeros((N,n))

        elem1 = np.row_stack((phi,zero_aux,zero_aux))
        elem2 = np.row_stack((zero_aux,phi,zero_aux))
        elem3 = np.row_stack((zero_aux,zero_aux,phi))

        PHI = np.array([elem1,elem2,elem3])
        return PHI
    
    def derivation_PHI(self,Phi,transposed):
        """
        Matrix Phi derivator. This function derivates de phi matrix according to the derivation and cross product 
        presented by Felipe Giraldo in his thesis work titled "Implementation of Machine Learning strategies in 
        Resonant Ultrasound Spectroscopy".  

        @Input: 
         Phi <np.array<np.array<np.array<int>>>>:Numpy array with the expansion of the base functions represented with indexes in 
         numpy arrays. It has the expressions for the three directions (x,y,z). Matrix diagonal in blocks such as: [[phi][0][0],
         [0][phi][0],[0][0][phi]].

        @Output:
         prod <np.array<np.array<np.array<int>>>>:Numpy array matrix with the values of thematrix product between matrix B and matrix 
         Phi.
         or 
         prodt <np.array<np.array<np.array<int>>>>: prod transposed
        """

        #Relevant paramenters
        N = len(Phi)
        n = len(Phi[0])

        #B matrix (derivation)
        B = np.zeros((6,3,2))
        B[0,0,0] = 1
        B[1,1,0] = 1
        B[2,2,0] = 1
        B[3,1,0] = 1/2
        B[3,2,0] = 1/2
        B[4,0,0] = 1/2
        B[4,2,0] = 1/2
        B[5,0,0] = 1/2
        B[5,1,0] = 1/2
        B[0,0,1] = 1
        B[1,1,1] = 1
        B[2,2,1] = 1
        B[3,1,1] = 1
        B[3,2,1] = 1
        B[4,0,1] = 1
        B[4,2,1] = 1
        B[5,0,1] = 1
        B[5,1,1] = 1

        #B*PHI 
        prod = np.zeros((6,n,4))

        for k in tqdm(range(6)):
            for i in range(n):
                for j in range(N):
                    for z in range(N):

                        phi_elem = Phi[j,i]
                        exp = phi_elem[j]

                        if k == 0 or k == 1 or k ==2:

                            if Phi[j,i,z] != 0 and z == j:
                                new = phi_elem.copy()
                                new[z] -= B[k,j,1]
                                new = np.append(new,Phi[j,i,z] * B[k,j,0])
                            else:
                                new = np.zeros(4)

                        else:

                            if Phi[j,i,z] != 0 and ((k == 3 and ((j == 1 and z == 2) or ( j == 2 and z == 1))) or (k == 5 and ((j == 1 and z == 0) or ( j == 0 and z == 1))) or (k == 4 and ((j == 2 and z == 0) or ( j == 0 and z == 2)))):
                                new = phi_elem.copy()
                                new[z] -= B[k,j,1]
                                new = np.append(new,Phi[j,i,z] * B[k,j,0])
                            else:
                                new = np.zeros(4)
                        
                        prod[k,i] += new    

        #Transposed or not       
        if transposed:
            prodt = np.transpose(prod,(1,0,2))
            return prodt
        else:
            return prod

    def calc_G(self):
        """
        Matrix G calculator. 

        @Output:
         G <np.array<np.array<int>>>: A numpy array that coresponds to the potential energy term of the Lagrangian 
         of the problem obtained with the volume integral of the matrix product amongst Phi transposed, B transposed, 
         C, B, and Phi (Phi: The matrix with the expansion of the base functions represented with indexes, B: The 
         derivation matrix taken from Felipe Giraldo's thesis work, C: The elastic constants matrix).
        """

        #Relevant parameter
        np.set_printoptions(linewidth=700)
        base = self.base
        sample = self.sample
        C = sample.C
        limits = sample.limits
        phi = base.phi
        PHI = self.construct_full_phi(phi)
        len_PHI = len(PHI)

        #B*PHI
        prod = self.derivation_PHI(PHI,False)

        #PHIt*Bt
        prod_t = self.derivation_PHI(PHI,True)

        N1 = len(prod_t)
        n1 = len(prod_t[0])
        n2 = len(prod[0])

        half_res = np.zeros((N1,n1,4))

        #PHIt*Bt*C
        for i in tqdm(range(N1)):
            for j in range(6):
                for k in range(n1):   
                    half_res[i,j,3] += prod_t[i,k,3] * C[k,j]

        G_noint = np.zeros((N1,n2,8))

        #PHIt*Bt*C*B*PHI
        for i in tqdm(range(N1)):
            row = np.array([])
            for j in range(n1):
                for k in range(n2):

                    phi_elem_1 = half_res[i,j]
                    phi_elem_2 = prod[j,k]

                    if phi_elem_1[3] == 0:
                        phi_elem_1 = np.zeros(4)

                    if phi_elem_2[3] == 0:
                        phi_elem_2 = np.zeros(4)

                    whole = np.concatenate((phi_elem_1,phi_elem_2))

                    G_noint[i,k] += whole

        N = len(G_noint)
        n = len(G_noint[0])

        G = np.zeros((N,n))

        #Integration (Only valid for rectangular parallelepipeids)
        for i in range(N):
            for j in range(n):

                exps = G_noint[i,j]
                cts = exps[3]*exps[7]
                p = exps[0]+exps[4]
                q = exps[1]+exps[5]
                r = exps[2]+exps[6]

                result = (8*cts*(limits[0]**(p+1))*(limits[1]**(q+1))*(limits[2]**(r+1)))/((p+1)*(q+1)*(r+1))
                
                G[i,j] = result
        
        return G

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