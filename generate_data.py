import numpy as np
import rusmodules
from rusmodules import rus
from rusmodules import data_generator
import scipy 

C_ranks = (0.3, 5.6)
dim_min = (0.01, 0.01, 0.01)
dim_max = (0.5, 0.5, 0.5)

def generate_eigenvalues(**kwargs):
    dims = np.random.uniform(kwargs["Dimensions"]["Min"], kwargs["Dimensions"]["Max"])
    C = data_generator.generate_C_matrix(kwargs["C_Rank"][0], kwargs["C_Rank"][1], kwargs["Crystal_structure"])
    gamma = rus.gamma_matrix(kwargs["Ng"], C, dims, kwargs["Shape"])
    E = rus.E_matrix(kwargs["Ng"], kwargs["Shape"])
    vals, vects = scipy.linalg.eigh(a = gamma, b = E)
    norma_gamma = np.linalg.norm(gamma - gamma.T)
    norma_E = np.linalg.norm(E - E.T)
    tol = 1e-7
    if "Verbose" in kwargs.keys() and kwargs["Verbose"]:
        print("**** C_matrix: *****")
        print(C)
        print("*** dimensions: ***")
        print(dims)
    #fin if 
    if abs(norma_gamma) > tol or abs(norma_E) > tol:
        print("Norma gamma: ", norma_gamma)
        print("Norma E: ", norma_E)
        raise ArithmeticError("Either gamma or E is non-symetric")
    #fin if 
    return vals
#fin funcion

input_data = { 
                "Dimensions": 
                {
                    "Min": dim_min,
                    "Max": dim_max
                },
                "C_Rank": C_ranks,
                "Crystal_structure": 0,
                "Shape": 0,
                "Verbose": True,
                "Ng": 14
              }


sample1 = generate_eigenvalues(**input_data)
#print(sample1)
print("**** First 6 eigenvalues:  ****")
print(sample1[0:6])
print("**** The rest of the eigenvalues: ****")
print(sample1[6:])
