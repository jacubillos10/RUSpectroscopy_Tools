import numpy as np
import rusmodules
from rusmodules import rus
#from rusmodules import rus_old
import scipy
import matplotlib.pyplot as plt
import time

np.set_printoptions(suppress = True)
shape = 0 # 0: parallelepiped, 1: cilinder, 2: ellipsoid 
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
m = (A**3)*0.2634 #g 9.84029 #9.839 #g/cm^3 
nombre_archivo = 'constantesURu2Si2.csv' #Mbar


C_const = np.genfromtxt(nombre_archivo, delimiter=',', skip_header=0, dtype=float)
#geometry = np.array([0.30529,0.20353,0.25334]) #cm  FeGa
#geometry = np.array([0.10872, 0.13981, 0.01757]) #cm SmB6
geometry = A*np.array([0.29605, 0.31034, 0.29138]) #cm URu2Si2
vol = alpha*np.prod(geometry)

gamma = rus.gamma_matrix(Ng, C_const, geometry, shape)
E = rus.E_matrix(Ng, shape)

vals, vects = scipy.linalg.eigh(a = (vol**(-1/3))*gamma, b = E)

print("Norma: ", np.linalg.norm(gamma - gamma.T))
print("Norma: ", np.linalg.norm(E - E.T))

freq = (vals*(vol**(1/3))/m)**0.5
freq_vueltas = freq*(1/(2*np.pi))
#np.savetxt(nombre_archivo[:-4] + '_freq_' + str(shape) +'.csv', np.c_[range(len(vals)), freq, freq_vueltas], delimiter = ',')

print("N vals: ", len(vals))
print(freq_vueltas[6:6+12])
print(vals[6:6+12])

"""
gamma2 = rus_old.gamma_matrix(Ng, C_const, geometry, shape)
E2 = rus_old.E_matrix(Ng, shape)
vals2, vect2 = scipy.linalg.eigh(a = (vol**(-1/3)*gamma2), b = E2)
print("Norma2: ", np.linalg.norm(gamma2 - gamma2.T))
print("Norma2: ", np.linalg.norm(E2 - E2.T)) 
print("N vals: ", len(vals2))
freq2 = (vals2*(vol**(1/3))/m)**0.5
freq2_vueltas = freq2*(1/(2*np.pi))
print(freq_vueltas[6:6+12])
print(vals2[6:6+12])

#time.sleep(5)

#plt.plot(vals, '.-')
#plt.show()
"""
