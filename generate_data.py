import numpy as np
import rusmodules
from rusmodules import rus
from rusmodules import data_generator
import scipy 

np.set_printoptions(suppress = True)
C_ranks = (0.3, 5.6)
dim_min = (0.01, 0.01, 0.01)
dim_max = (0.5, 0.5, 0.5)

def generate_eigenvalues(Dimensions, C_rank, Crystal_structure, Shape, N_frequencies, Ng, Verbose = False):
    dims = np.random.uniform(Dimensions["Min"], Dimensions["Max"])
    C = data_generator.generate_C_matrix(C_rank[0], C_rank[1], Crystal_structure)
    gamma = rus.gamma_matrix(Ng, C, dims, Shape)
    E = rus.E_matrix(Ng, Shape)
    N_freq = N_frequencies
    vals, vects = scipy.linalg.eigh(a = gamma, b = E)
    norma_gamma = np.linalg.norm(gamma - gamma.T)
    norma_E = np.linalg.norm(E - E.T)
    tol = 1e-7
    if Verbose:
        print("**** C_matrix: *****")
        print(C)
        print("*** dimensions: ***")
        print(dims)
        print("*** end verbose ***")
    #fin if 
    if abs(norma_gamma) > tol or abs(norma_E) > tol:
        print("Norma gamma: ", norma_gamma)
        print("Norma E: ", norma_E)
        raise ArithmeticError("Either gamma or E is non-symetric")
    #fin if
    if any((abs(vals[i]) > tol for i in range(6))):
        raise ArithmeticError("One of the first six eigenvalues is not zero")
    #fin if
    if N_freq == "all":
        N_freq = len(vals)
    #fin if 
    C_reshaped = np.r_[*(C[i,i:] for i in range(6))]
    return np.r_[dims, C_reshaped, vals[6:N_freq]]
#fin funcion

input_data = { 
                "Dimensions": 
                {
                    "Min": dim_min,
                    "Max": dim_max
                },
                "C_rank": C_ranks,
                "Crystal_structure": 0,
                "Shape": 0,
                "Verbose": False,
                "N_frequencies": 24,
                "Ng": 14
              }


sample1 = generate_eigenvalues(**input_data)
print(sample1)
