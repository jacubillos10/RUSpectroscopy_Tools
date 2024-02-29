import numpy as np
import rusmodules
from rusmodules import rus

C_const = np.genfromtxt('constantes.csv', delimiter=',', skip_header=0, dtype=float)
example_indices = rus.it_c(0,1)
#example_term = rus.generate_term_in_gamma_matrix(np.array([1, 0, 0]), np.array([0, 0, 1]), 0, 1, 0, 2, C_const, np.array([1, 1, 1]))
example_element = rus.generate_matrix_element_gamma(0, 1, np.array([1,0,0]), np.array([0,0,1]), C_const, np.array([1.0,1.0,1.0]))
print(example_indices)
#print(example_term)
print(example_element)