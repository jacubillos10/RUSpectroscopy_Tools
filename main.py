import numpy as np
import rusmodules
from rusmodules import rus
import scipy
import matplotlib.pyplot as plt
import time

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

#Datos del URu2Si2
Ng = 14
rho = 9.839 #g/cm^3
nombre_archivo = 'constantesURu2Si2.csv' #Mbar


C_const = np.genfromtxt(nombre_archivo, delimiter=',', skip_header=0, dtype=float)
#geometry = np.array([0.30529,0.20353,0.25334]) #cm  FeGa
#geometry = np.array([0.10872, 0.13981, 0.01757]) #cm SmB6
geometry = np.array([0.29605, 0.31034, 0.29138]) #cm URu2Si2
gamma = rus.gamma_matrix(Ng, C_const, geometry)
E = rus.E_matrix(Ng)

vals, vects = scipy.linalg.eigh(a = gamma, b = E)

print("Norma: ", np.linalg.norm(gamma - gamma.T))
print("Norma: ", np.linalg.norm(E - E.T))
print(vals[0])

vals = (vals/rho)**0.5
vals_vueltas = vals*(1/(2*np.pi))
np.savetxt(nombre_archivo[:-4] + '_freq_.csv', np.c_[range(len(vals)),vals, vals_vueltas], delimiter = ',')

print("N vals: ", len(vals))


#time.sleep(5)

#plt.plot(vals, '.-')
#plt.show()
