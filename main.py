import numpy as np
import rusmodules
from rusmodules import rus

C_const = np.genfromtxt('constantes.csv', delimiter=',', skip_header=0, dtype=float)
geometry = np.array([1.0,1.0,1.0])

gamma = rus.gamma_matrix(1, C_const, geometry)
E = rus.E_matrix(1)
print(gamma)
print("***************")
print(E)