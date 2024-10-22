import numpy as np
import pandas as pd
import scipy
import os
import itertools
from datamodules import preproc
from rusmodules import rus, eigenvals, geometry
import os

def gen_combinatorial_parameters(Ng, C_rank, Np_dim, shape):
    """
    @input: C_rank <dict>: Diccionario que indica el mínimo, el máximo y la cantidad de puntos de cada uno de los valores a generar. 
        Por ejemplo, para el caso isotrópico C_rank = {"K": {"min": 0, "max": 1, "Finura": 10}, "mu": {"min": 0, "max": 1, "Finura": 10}}
    @input: Np_dim <int>: Cantidad de puntos a usar en las relaciones de dimensiones gamma y beta. Estas siempre estarán de 0 a 1.  
    @input: Ng <int>: Grado máximo de las funciones base en el forward problem
    @input: shape <string>: "Parallelepiped", "Cylinder" o "Ellipsoid" según el caso
    @output: datos <pd.DataFrame>: salida de los datos combinatoriales. 
    """
    max_eta = {"Parallelepiped": 0.5*np.pi, "Cylinder": np.pi, "Ellipsoid": 0.5*np.pi}
    max_beta = {"Parallelepiped": np.pi, "Cylinder": np.pi, "Ellipsoid": np.pi}
    geometry_options = {"Parallelepiped": {"theta": True, "phi": True}, 
                        "Cylinder": {"theta": False, "phi": True},
                        "Ellipsoid": {"theta": True, "phi": True}}

    N_dir = 2
    keys_dims = ("eta", "beta")
    combinations_param = np.array(tuple(itertools.product(*(np.linspace(C_rank[key]["min"] 
                        + (1/C_rank[key]["Finura"]), C_rank[key]["max"]*(1 - (1/C_rank[key]["Finura"])), 
                        C_rank[key]["Finura"]) for key in C_rank.keys()))))
    #combinations_dims = np.array(tuple(itertools.combinations_with_replacement(np.linspace((1/Np_dim), 1, Np_dim), N_dir)))
    combinations_dims = geometry.generate_sphere_surface_points(Np_dim, max_eta[shape], max_beta[shape], geometry_options[shape])
    index_total_combinations = np.array(tuple(itertools.product(range(len(combinations_param)), range(len(combinations_dims)))))
    C_dir = lambda C_keys, combi: dict(zip(C_keys, combi))
    total_combinations = tuple(map(lambda x: {**C_dir(C_rank.keys(), combinations_param[x[0]]), **C_dir(keys_dims, combinations_dims[x[1]])}, index_total_combinations))
    return total_combinations
#fin funcion

def generate_combinatorial_data_isotropic(path, Ng, Np_const, Np_geo, shape, mode = "Magnitude", N_vals = 100):
    pars = gen_combinatorial_parameters(Ng, {"phi_K": {"min": 0, "max": np.pi/2, "Finura": Np_const}}, Np_geo, shape)
    exponents = {"Magnitude": 1, "Sum": 2}
    for a, param in enumerate(pars):
        param["x_K"] = (np.cos(param["phi_K"]))**exponents[mode]
        param["x_mu"] = (np.sin(param["phi_K"]))**exponents[mode]
        constant_relations = {"x_K": param["x_K"], "x_mu": param["x_mu"]}
        data_vals = eigenvals.get_eigenvalues_from_crystal_structure(Ng, constant_relations, param["eta"], param["beta"], shape)
        vals = data_vals["eig"]
        keys_eigen = tuple(map(lambda x: "eig_" + str(x), range(N_vals)))
        for i in range(N_vals):
            param[keys_eigen[i]] = vals[i]
        #fin for 
        with open(path, "a+t") as f:
            if a == 0:
                f.write(",".join(list(param.keys())) + "\n")
            #fin if 
            f.write(",".join(list(map(lambda x: str(x), param.values()))) + "\n")
        #fin with
    #fin for 
    #return pd.DataFrame(pars)
#fin función

if __name__ == "__main__":
    """
    """
    ruta_archivo = "input_data/" + "combi_" + str(os.getpid()) + ".csv"
    generate_combinatorial_data_isotropic(ruta_archivo, 6, 10, 4, "Parallelepiped")  
