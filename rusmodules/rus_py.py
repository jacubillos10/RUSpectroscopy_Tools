import numpy as np
from numba import jit

@jit(nopython = True)
def it_c(i,j):
    """
    This function changes the double index convention for single index convention in the elastic constant matrix C.
    00 is changed to 0, 11 is changed to 1, 22 is changed to 2, 
    01 or 10 is changed to 5
    02 or 20 is changed to 4
    12 or 21 is changed to 3

    @Input i <int>: first index to be changed
    @Input j <int>: second index to be changed 
    @Return index <int>: New index 
    """
    if i == j:
        index = i
    else:
        index = 6 - (i + j)
    #fin if
    return index
#fin función

@jit(nopython = True)
def generate_term_in_ſ(exp_index1, exp_index2, i1, i2, j1, j2, C, geo_par):
    """
    This function generates one term of the sum composing an element of the matrix ſ. See equation 11 in Leisure 1997
    @Input exp_index1 <np.array>: Exponents lambda, mu and nu of the p element in the matrix and i1 index
    @Input exp_index2 <np.array>: Exponents lambda, mu and nu of the q element in the matrix and i2 index
    @Input i1 <int>: index i of the expresion in eq 11. 0 to x, 1 to y, 2 to z
    @Input i2 <int>: index i' of the expresion in eq 11. 0 to x, 1 to y, 2 to z
    @Input j1 <int>: index j of the sum in equation 11. 0 to x, 1 to y, 2 to z
    @Input j2 <int>: index j' of the sum in equation 11. 0 to x, 1 to y, 2 to z
    @Input C <np.array>: Matrix with the elastic constants
    @Input geo_par <np.array>: Geometric parameters of the sample, such as Lx, Ly, Lz
    @Return <float>: Returns a term in the equation 11 of Leisure 1997
    """
    array_j1 = np.zeros(3)
    array_j2 = np.zeros(3)
    array_j1[j1] = 1
    array_j2[j2] = 1
    coeff = exp_index1 + exp_index2 + 1 - array_j1 - array_j2
    Q = (1 - (-1)**coeff[0])*(1 - (-1)**coeff[1])*(1 - (-1)**coeff[2])
    if Q == 0:
        return 0
    #fin if 
    P = (C[it_c(i1,j1),it_c(i2,j2)]/geo_par[j1]*geo_par[j2])
    R = (exp_index1[j1]*exp_index2[j2])/(coeff[0]*coeff[1]*coeff[2])
    return P*Q*R
#fin función    

@jit(nopython = True)
def generate_matrix_element_ſ(index1, index2, C, geo_par, geometry = 'parallelpiped'):
    exp_index1, exp_index2 = index1[1:], index2[1:]
    i1, i2 = index1[0], index2[0]
    acum = 0
    for j1 in range(3):
        for j2 in range(3):
            acum += generate_term_in_ſ(exp_index1, exp_index2, i1, i2, j1, j2, C, geo_par)
        # fin for 
    #fin for
    return acum
#fin función

if __name__ == "__main__":
    C_const = np.genfromtxt('constantes.csv', delimiter=',', skip_header=0, dtype=float)
    #aa = generate_term_in_s(np.array([1, 0, 0]), np.array([0, 0, 1]), 0, 1, 0, 2, C_const, np.array([1, 1, 1]))
    #print(aa/8)
    aac = generate_matrix_element_ſ(np.array([0,1,0,0]), np.array([1,0,0,1]), C_const, np.array([1,1,1]))
    print(aac/8)