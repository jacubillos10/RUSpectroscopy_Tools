import numpy as np
import rusmodules
from rusmodules import rus
#from rusmodules import rus_old
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
A = 2
#Datos del URu2Si2
Ng = 14
m = 0.2634 #g 9.84029 #9.839 #g/cm^3 
m_p = (A**3) * m
nombre_archivo = 'constantesURu2Si2.csv' #Mbar


C_const = np.genfromtxt(nombre_archivo, delimiter=',', skip_header=0, dtype=float)
#geometry = np.array([0.30529,0.20353,0.25334]) #cm  FeGa
#geometry = np.array([0.10872, 0.13981, 0.01757]) #cm SmB6
geometry = np.array([0.29605, 0.31034, 0.29138]) #cm URu2Si2
geometry_p = A * geometry
vol = alpha*np.prod(geometry)
vol_p = alpha*np.prod(geometry_p)
gamma = rus.gamma_matrix(Ng, C_const, geometry, shape)
E = rus.E_matrix(Ng, shape)

vals, vects = scipy.linalg.eigh(a = (vol**(-1/3))*gamma, b = E)

print("Norma: ", np.linalg.norm(gamma - gamma.T))
print("Norma: ", np.linalg.norm(E - E.T))

freq = (vals[6:]*(vol**(1/3))/m)**0.5
freq_vueltas = freq*(1/(2*np.pi))
#np.savetxt(nombre_archivo[:-4] + '_freq_' + str(shape) +'.csv', np.c_[range(len(vals)), freq, freq_vueltas], delimiter = ',')

gamma_p = rus.gamma_matrix(Ng, C_const, geometry_p, shape)
E_p = rus.E_matrix(Ng, shape)

vals_p, vects_p = scipy.linalg.eigh(a = (vol_p**(-1/3))*gamma_p, b = E_p)
freq_p = (vals_p[6:]*(vol_p**(1/3))/m_p)**0.5
freq_vueltas_p = freq_p*(1/(2*np.pi))

print("N vals: ", len(vals))
print(freq_vueltas[:12])
print(freq_vueltas_p[:12])
print(vals[6:6+12])
print(vals_p[6:6+12])
eigenvalue_index = np.array(range(len(vals[6:])))


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(eigenvalue_index, freq_vueltas, eigenvalue_index, freq_vueltas_p)
ax1.set_xlabel("Indice del valor propio")
ax1.set_ylabel("Frecuencia en Hz")
ax1.legend(["A = 1", "A = 2"])
plt.show()
