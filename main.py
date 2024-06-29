import numpy as np
import rusmodules
from rusmodules import rus
import scipy
import matplotlib.pyplot as plt
import time

np.set_printoptions(suppress = True)
shape = 1 # 0: parallelepiped, 1: cilinder, 2: ellipsoid 
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
m = (A**3)*0.20688 #g 9.84029 #9.839 #g/cm^3 
nombre_archivo = 'constantesURu2Si2.csv' #Mbar


C_const = np.genfromtxt(nombre_archivo, delimiter=',', skip_header=0, dtype=float)
#geometry = np.array([0.30529,0.20353,0.25334]) #cm  FeGa
#geometry = np.array([0.10872, 0.13981, 0.01757]) #cm SmB6
geometry = A*np.array([0.29605, 0.31034, 0.29138]) #cm URu2Si2
#Note que el doble factorial de 2 tiró un segfault. En algún momento puede estar pasando esto. 
#print(rus.fact2(-1))
gamma = rus.gamma_matrix(Ng, C_const, geometry, shape)
E = rus.E_matrix(Ng, shape)

vals, vects = scipy.linalg.eigh(a = (m**(-1/3))*gamma, b = E)

print("Norma: ", np.linalg.norm(gamma - gamma.T))
print("Norma: ", np.linalg.norm(E - E.T))
print(vals[0])

freq = (vals/(m**(2/3)))**0.5
freq_vueltas = freq*(1/(2*np.pi))
np.savetxt(nombre_archivo[:-4] + '_freq_' + str(shape) +'.csv', np.c_[range(len(vals)), freq, freq_vueltas], delimiter = ',')

print("N vals: ", len(vals))
print(freq_vueltas[6:6+12])
print(vals[6:6+12])

#time.sleep(5)

#plt.plot(vals, '.-')
#plt.show()
