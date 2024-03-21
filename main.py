import numpy as np
import rusmodules
from rusmodules import rus
import scipy

C_const = np.genfromtxt('constantes2.csv', delimiter=',', skip_header=0, dtype=float)
geometry = np.array([1.0,1.0,1.0])

gamma = rus.gamma_matrix(1, C_const, geometry)
E = rus.E_matrix(1)

vals, vects = scipy.linalg.eig(a = gamma, b = E)

print(vals)
print(vals[0])
print(vals[0].imag)