import numpy as np
import rusmodules
from rusmodules import rus
import scipy
import matplotlib.pyplot as plt
import time

np.set_printoptions(suppress = True)
shape = 1 # 0: parallelepiped, 1: cilinder, 2: ellipsoid 
alphas = (1, np.pi/4, np.pi/6)
alpha = alphas[shape]
Finura = 100

A = 2
#Datos del URu2Si2
Ng = 8
rho = 1
mat_2 = np.zeros((6,6))
mat_3 = np.ones((3,3)) - np.identity(3)
mat_2[0:3,0:3] = mat_3
C_const = 3*np.identity(6) + mat_2
b_min = 0.01
b_max = 2
bx_izq = np.linspace(b_min, b_max, int(Finura/2))
bz_izq = 2 * np.ones(int(Finura/2))
bx_der = 2 * np.ones(int(Finura/2))
bz_der = np.linspace(b_max, b_min, int(Finura/2))
count = np.array(range(Finura))
bx = np.r_[bx_izq, bx_der]
bz = np.r_[bz_izq, bz_der]
vol = alpha*(bx**2)*bz
b = np.c_[bx,bx,bz]
filas = 15
freqs = np.zeros((filas, Finura))
for i in range(Finura):
    m = (A**3)*rho*vol[i]
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
