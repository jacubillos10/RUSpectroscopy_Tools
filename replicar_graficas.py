import numpy as np
import rusmodules
from rusmodules import rus
import scipy
import matplotlib.pyplot as plt
import time

np.set_printoptions(suppress = True)
shape = 2 # 0: parallelepiped, 1: cilinder, 2: ellipsoid 
alphas = (1, np.pi/4, np.pi/6)
alpha = alphas[shape]
Finura = 500

A = 2
#Datos del URu2Si2
Ng = 8
m = (A**3)*0.2634 #g 9.84029 #9.839 #g/cm^3 
nombre_archivo = 'constantesURu2Si2.csv' #Mbar
C_const = np.genfromtxt(nombre_archivo, delimiter=',', skip_header=0, dtype=float)
initial_geometry = A*np.ones(3) #cm URu2Si2
vol = alpha*np.prod(initial_geometry)
b_min = 0.01
b_max = 2
bx_izq = np.linspace(b_min, b_max, int(Finura/2))
bz_izq = 2 * np.ones(int(Finura/2))
bx_der = 2 * np.ones(int(Finura/2))
bz_der = np.linspace(b_max, b_min, int(Finura/2))
count = np.array(range(Finura))
bx = np.r_[bx_izq, bx_der]
bz = np.r_[bz_izq, bz_der]
b = np.c_[bx,bx,bz]
filas = 15
freqs = np.zeros((filas, Finura))
for i in range(Finura):
    gamma = rus.gamma_matrix(Ng, C_const, b[i,:], shape)
    E = rus.E_matrix(Ng, shape)
    vals, vects = scipy.linalg.eigh(a = (m**(-1/3))*gamma, b = E)
    print("***Iteración número ", i)
    print("Norma: ", np.linalg.norm(gamma - gamma.T))
    print("Norma: ", np.linalg.norm(E - E.T))
    freq = ((vals/(2*np.pi))/(m**(2/3)))**0.5
    freqs[:,i] = freq[6:6+filas]
#fin for

fig1 = plt.figure(figsize = (20, 20))
ax1 = fig1.add_subplot(111)
ax1.plot(count, freqs.T)
plt.savefig("familia_de_resonancias_forma_" + str(shape) + ".png")
