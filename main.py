import numpy as np
import rusmodules
from rusmodules import rus
import scipy
import matplotlib.pyplot as plt

C_const = np.genfromtxt('constantes3.csv', delimiter=',', skip_header=0, dtype=float)
geometry = np.array([1.0,1.0,1.0])

gamma = rus.gamma_matrix(7, C_const, geometry)
E = rus.E_matrix(7)

vals, vects = scipy.linalg.eigh(a = gamma, b = E)

print("Norma: ", np.linalg.norm(gamma - gamma.T))
print("Norma: ", np.linalg.norm(E - E.T))

print(vals)
print(vals[0].real)
print(vals[0].imag)

print("N vals: ", len(vals))

plt.plot(vals, '.-')
plt.show()