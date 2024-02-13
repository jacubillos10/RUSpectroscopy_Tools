import RUS_classes as rus
import numpy as np 

class Forward: 
    """
    Class Forward: Class incharge of combining the whole Forward RUS problem, calculate de epsilon and gamma matrixes 
    for the calculation of the eigen values and eigen vectors (solutions). 

    Attributes: 
     B <np.array<np.array<str>>>: Numpy array matrix that compacts the operations that have to be done to calculate the gamma matrix.
     sample <sample>: Object that corresponds to the sample used for the forward RUS problem. 
     base <base>: Object that corresponds to the basis functions taht are going to be used to solve the forward RUS problem.

    Methods:
     __init__: Initialization method. B, N, type, shape, system, rho, C and limits must be specified.
     get_B: Returns the user the operations matrix B.
     Calculate_eigenvals: Method incharge of calculating the eigen vectors that solve the forward RUS problem (a generalized eigen values problem).
     Calculate_gamma: Method incharge of calculating the gamma matrix in the forward RUS problem (Matrix that contains the potential energy information).
     Calculate_epsilon: Method incharge of calculating the epsioln matrix in the forward RUS problem (Matrix that contains the kinetic energy information).
    """

    def __init__(self,B,N,type,shape,system,limits,rho,C):
        """
        Class initialization

        @Input:
         B <np.array<np.array<str>>>: Operations matrix.
         N <int>: Max order of the base function.
         type <str>: Name of the base functions.
         shape <str>: A string with the name of the shape of the sample.
         system <str>: A string with the name of the shape of the sample.
         limits <np.array<int>>: An array with all the information about the size of the sample. 
         rho <float>: A float that corresponds to the density of the sample.
        """
        self.B = B
        self.sample = rus.sample(shape,system,limits,rho,C)
        self.base = rus.base(N,type)

    def get_B(self):
        """
        B attribute class getter.
        """
        return self.B
    
    def Calculate_eigenvals():
        return None

    def Calculate_gamma():
        return None
    
    def Calculate_epsilon(self):
        """
        Method incharge of calculating the matrix epsilon. In the forward RUS problem this matrix all the kinetic information of the problem. 

        @Output:
         Epsilon <np.array<np.array<np.array<int>>>>: Matrix with all the kinetic information of the forward RUS problem.
        """
        sample = self.sample
        base = self.base
