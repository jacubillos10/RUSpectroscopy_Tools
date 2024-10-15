import numpy as np
import pandas as pd
import scipy
import os
import itertools
from datamodules import preproc
from rusmodules import rus

def fill_eigenvalues(Ng, independent_constants, gamma, beta, eta, shape):
    """
    @input Ng <int>: Grado máximos de las funciones base en el forward problem
    @input independent_constants <dict>: Valor de las contantes elásticas independientes. 
        Por ejemplo: independent_constants = {"K": 0.45, "mu": 0.25}
    @input gamma <double>: Relación entre las dimensiones a y b: b/a
    @input beta <double>: Relación entre las dimensiones c y a: c/a
    @input eta <double>: Relación entre las dimensiones b y c: c/b
    @input shape <string>: Forma de la muestra
    @output vals <np.array>: Valores propios del forward problem en cuestión
    """
    C = np.zeros((6,6))
    alphas = {"Parallelepiped": 1, "Cylinder": np.pi/4, "Ellipsoid": np.pi/6}
    alpha = alphas[shape]
    C_prim = np.array(tuple(map(lambda i: tuple(map(lambda j: 1 if i == j and i<3 else 0, range(6))), range(6)))) #Valores de C00, C11, C22
    C_sec = np.array(tuple(map(lambda i: tuple(map(lambda j: 1 if i == j and i >= 3 else 0, range(6))), range(6)))) #Valores de C33, C44, C55
    C_shear_prim = np.array(tuple(map(lambda i: tuple(map(lambda j: 1 if i != j and i<3 and j<3 else 0, range(6))), range(6)))) #Valores de C01, C02, C12
    C_shear_sec = np.array(tuple(map(lambda i: tuple(map(lambda j: 1 if i != j and i>=3 and j>=3 else 0, range(6))), range(6)))) #Valores de C34, C35, C45
    if len(independent_constants) == 2:
        C_prim = C_prim * (independent_constants["K"] + (4/3)*independent_constants["mu"])
        C_sec = C_sec * independent_constants["mu"]
        C_shear_prim = C_shear_prim * (independent_constants["K"] - (2/3)*independent_constants["mu"])
        C = C_prim + C_sec + C_shear_prim
        a = 1/((alpha*gamma*beta)**(1/3))
        b = gamma*a
        c = beta*a
        geometry = np.array([a,b,c])
        vol = alpha*np.prod(geometry)
        Gamma = rus.gamma_matrix(Ng, C, geometry, shape)
        E = rus.E_matrix(Ng, shape)
        vals = scipy.linalg.eigvalsh(Gamma, b = E)
    #fin if
    return vals
#fin función

def fill_combinatorial(Ng, C_rank, Np_dim, shape, N_freq):
    """
    @input: C_rank <dict>: Diccionario que indica el mínimo, el máximo y la cantidad de puntos de cada uno de los valores a generar. 
        Por ejemplo, para el caso isotrópico C_rank = {"K": {"min": 0, "max": 1, "Finura": 10}, "mu": {"min": 0, "max": 1, "Finura": 10}}
    @input: Np_dim <int>: Cantidad de puntos a usar en cada una de las dimensiones. 
    @input: Ng <int>: Grado máximo de las funciones base en el forward problem
    @input: shape <string>: "Parallelepiped", "Cylinder" o "Ellipsoid" según el caso
    @output: datos <pd.DataFrame>: salida de los datos combinatoriales. 
    """
    N_dir = {"Parallelepiped": 2, "Cylinder": 3, "Ellipsoid": 2}
    keys_dims = {"Parallelepiped": ["gamma", "beta"], "Cylinder": ["gamma", "beta", "eta"], "Ellipsoid": ["gamma", "beta"]}
    combinations_param = np.array(tuple(itertools.product(*(np.linspace(C_rank[key]["min"] + (1/C_rank[key]["Finura"]), C_rank[key]["max"], C_rank[key]["Finura"]) for key in C_rank.keys()))))
    combinations_dims = np.array(tuple(itertools.combinations_with_replacement(np.linspace((1/Np_dim), 1, Np_dim), N_dir[shape])))
    index_total_combinations = np.array(tuple(itertools.product(range(len(combinations_param)), range(len(combinations_dims)))))
    total_combinations = np.array(tuple(map(lambda x: (*combinations_param[x[0]], *combinations_dims[x[1]]), index_total_combinations)))
    keys_ini = list(C_rank.keys()) + keys_dims[shape]
    total_combinations = pd.DataFrame(dict(map(lambda x, y: (x, y), keys_ini, total_combinations.T)))
    return total_combinations
#fin funcion

if __name__ == "__main__":
    a = fill_combinatorial(8, {"K": {"min": 0, "max": 1, "Finura": 10}, "mu": {"min": 0, "max": 1, "Finura": 10}}, 5, "Ellipsoid")
    print(a)
 
