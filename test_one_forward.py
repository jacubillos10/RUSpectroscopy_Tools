import numpy as np
import rusmodules
from rusmodules import rus
from rusmodules import eigenvals
import scipy
import matplotlib.pyplot as plt
import time

np.set_printoptions(suppress = True)
shape = 1 # 0: parallelepiped, 1: cilinder, 2: ellipsoid 
alphas = (1.0, np.pi/4, np.pi/6)
alpha = alphas[shape]
"""
#Datos del FeGa
Ng = 12
rho = 7.979 #g/cm^3
nombre_archivo = 'constantesFeGa.csv'
"""

"""
#Datos del SmB6
Ng = 16
rho = 4.869 #g/cm^3
nombre_archivo = 'constantesSmB6.csv' #Mbar
"""
A = 1
#Datos del URu2Si2
Ng = 14
m = (A**3) * 0.20688 #g 9.84029 #9.839 #g/cm^3 
m_p = (A**3) * m
nombre_archivo = 'constant_data/constantesURu2Si2.csv' #Mbar


C_const = np.genfromtxt(nombre_archivo, delimiter=',', skip_header=0, dtype=float)
#geometry = np.array([0.30529,0.20353,0.25334]) #cm  FeGa
#geometry = np.array([0.10872, 0.13981, 0.01757]) #cm SmB6
geometry = A * np.array([0.29605, 0.31034, 0.29138]) #cm URu2Si2
geometry_p = A * geometry
#vol = alpha*np.prod(geometry)
r = (sum(geometry**2))**0.5
Gamma = rus.gamma_matrix(Ng, C_const, geometry, shape)
E = rus.E_matrix(Ng, shape)
vals, vects = scipy.linalg.eigh(a = Gamma/r, b = E)
#print("Norma: ", np.linalg.norm(gamma - gamma.T))
#print("Norma: ", np.linalg.norm(E - E.T))
freq = (vals[6:]*(r/m))**0.5
freq_vueltas = freq*(1/(2*np.pi))
print("Original:")
print("Eigs completos:")
print(vals[6:6+12])
vals_new = vals[6:]
vals_new[1:] = vals_new[1:]/vals_new[0]
print("Eigs relativos:")
print(vals_new[:12])
print("Frecuencias en MHz:")
print(freq_vueltas[:12])

print("NEW:")
eta = 2*np.arccos(geometry[2]/r)
beta = 4*np.arctan(geometry[1]/geometry[0])
shapes = ["Parallelepiped", "Cylinder", "Ellipsoid"]
eigenvalues_test = eigenvals.get_eigenvalues(Ng, C_const, eta, beta, shapes[shape])["eig"]
print("Eigs relativos:")
print(eigenvalues_test[:12])
eigenvalues_test[1:] = eigenvalues_test[1:] * eigenvalues_test[0]
print("Eigs completos:")
print(eigenvalues_test[:12])


print("Test de generador de valores propios relativos")
const_relations = {"x_K": 4, "x_mu": 3}
eta = np.pi/2
beta = np.pi/2
vals_peq = eigenvals.get_eigenvalues_from_crystal_structure(Ng, const_relations, eta, beta, "Ellipsoid")["eig"]
print(vals_peq)
vals_big = eigenvals.get_eigenvalues_from_crystal_structure(Ng, const_relations, eta, beta, "Ellipsoid", 3)["eig"]
print(vals_big)
print(vals_big[0]/vals_peq[0])
