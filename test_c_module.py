import numpy as np
from rusmodules import rus_test
from rusmodules import rus

np.set_printoptions(suppress = True)
shape = 1 # 0: parallelepiped, 1: cilinder, 2: ellipsoid 

A = 8
#Datos del URu2Si2
Ng = 14
m = (A**3)*0.2634 #g 9.84029 #9.839 #g/cm^3 
#nombre_archivo = 'constantesURu2Si2.csv' #Mbar
nombre_archivo = 'constantes.csv'


C_const = np.genfromtxt(nombre_archivo, delimiter=',', skip_header=0, dtype=float)
geometry = A*np.array([0.29605, 0.31034, 0.29138]) #cm URu2Si2

exp_index1 = np.array([2,6,6])
exp_index2 = np.array([4,2,8])
exp_index1_t = np.array(exp_index1, dtype = np.int32)
exp_index2_t = np.array(exp_index2, dtype = np.int32)
i1 = 0 
i2 = 0
gamma_element_1 = rus.generate_matrix_element_gamma(i1, i2, exp_index1, exp_index2, C_const, geometry, shape)
gamma_element_2 = rus_test.generate_gamma_matrix_element(i1, i2, exp_index1_t, exp_index2_t, C_const, geometry, shape)
print("Elemento gamma old: ", gamma_element_1)
print("Elemento gamma new: ", gamma_element_2)

E_element_1 = rus.generate_matrix_element_E(i1, i2,exp_index1, exp_index2, shape)
E_element_2 = rus_test.generate_E_matrix_element(i1, i2, exp_index1_t, exp_index2_t, shape)

print("Elemento E old: ", E_element_1)
print("Elemento E new: ", E_element_2)
