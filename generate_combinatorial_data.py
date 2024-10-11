import numpy as np
import pandas as pd
from datamodules import preproc
from rusmodules import rus

def fill_eigenvalues(independent_constants, gamma, beta, shape):
    C = np.zeros((6,6))
    alphas = {"Parallelepiped": 1, "Cylinder": np.pi/4, "Ellipsoid": np.pi/6}
    alpha = alphas[shape]
    C_prim = np.array(tuple(map(lambda i: tuple(map(lambda j: 1 if i == j and i<3 else 0, range(6))), range(6)))) #Valores de C00, C11, C22
    C_sec = np.array(tuple(map(lambda i: tuple(map(lambda j: 1 if i == j and i >= 3 else 0, range(6))), range(6)))) #Valores de C33, C44, C55
    C_shear_prim = np.array(tuple(map(lambda i: tuple(map(lambda j: 1 if i != j and i<3 and j<3 else 0, range(6))), range(6)))) #Valores de C01, C02, C12
    C_shear_sec = np.array(tuple(map(lambda i: tuple(map(lambda j: 1 if i != j and i>=3 and j>=3 else 0, range(6))), range(6)))) #Valores de C34, C35, C45
    if len(independent_constants) == 2:
        C_prim = Cprim * (independent_constants["K"] + (4/3)*independent_constants["mu"])
        C_sec = C_sec * independent_constants["mu"]
        C_shear_prim = C_shear_prim * (independent_constants["mu"])
        C = C_prim + C_sec + C_shear_prim
        a = 1/((alpha*gamma*beta)**(1/3))
        b = gamma*a
        c = beta*a
