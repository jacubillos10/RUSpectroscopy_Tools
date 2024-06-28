import numpy as np
from rusmodules import rus_test
from rusmodules import rus

np.set_printoptions(suppress = True)
shape = 0 # 0: parallelepiped, 1: cilinder, 2: ellipsoid 

A = 1
#Datos del URu2Si2
Ng = 14
m = (A**3)*0.2634 #g 9.84029 #9.839 #g/cm^3 
nombre_archivo = 'constantesURu2Si2.csv' #Mbar


C_const = np.genfromtxt(nombre_archivo, delimiter=',', skip_header=0, dtype=float)
geometry = A*np.array([0.29605, 0.31034, 0.29138]) #cm URu2Si2

exp_index1 = np.array([4,8,2])
exp_index2 = np.array([6,6,4])
exp_index1_t = np.array(exp_index1, dtype = np.int32)
exp_index2_t = np.array(exp_index2, dtype = np.int32)
i1 = 0
i2 = 0
j1 = 0
j2 = 0
gamma_element_1 = rus.generate_term_in_gamma_matrix_element(exp_index1, exp_index2, i1, i2, j1, j2, C_const, geometry, shape)
gamma_element_2 = rus_test.generate_term_in_gamma_matrix_element(exp_index1_t, exp_index2_t, i1, i2, j1, j2, C_const, geometry, shape)
print(gamma_element_1)
print(gamma_element_2)
