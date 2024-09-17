import numpy as np
import pandas as pd
import itertools

def generate_combinations(cry_st, mode,  par_const, par_dim, shape = "Parallelepiped"): 
    dict_shape = {"Parallelepiped": 0, "Cylinder": 1, "Ellipsoid": 2}
    get_vals = lambda minimum, maximum, N: np.linspace(minimum, maximum, N)

    if cry_st == "Isotropic":
        keys_bas = ["K", "mu", "l"]
        min_val = [par_const[0], par_const[0], par_dim[0]]
        max_val = [par_const[1], par_const[1], par_dim[1]]
        if mode == "Omega":
            N = [10, 10, 5]
            dict_generation = dict(map(lambda x, y, z, w: (x, get_vals(y, z, w)), keys_bas, min_val, max_val, N))
            dict_generation["rho"] = get_vals(2.0, 10, 10)
            combi_perm = ["K", "mu", "rho"]
        else:
            N = [30, 30, 5]
            dict_generation = dict(map(lambda x, y, z, w: (x, get_vals(y, z, w)), keys_bas, min_val, max_val, N))
            combi_perm = ["K", "mu"]
        #fin if
        combinations_param = np.array(tuple(itertools.product(*(dict_generation[key] for key in combi_perm))))
    #fin if 

    #OJO!! LA SIGUIENTE LÍNEA SOLO APLICA PARA PERALELEPÍPEDO Y ELIPSOIDE!!
    combinations_dims = np.array(tuple(itertools.combinations_with_replacement(dict_generation["l"], 3)))
    index_total_combinations = np.array(tuple(itertools.product(range(len(combinations_param)), range(len(combinations_dims)))))
    total_combinations = np.array(tuple(map(lambda x: (*combinations_param[x[0]], *combinations_dims[x[1]]), index_total_combinations)))
    keys_ini = combi_perm + ["lx", "ly", "lz"]

    total_combinations = pd.DataFrame(dict(map(lambda x, y: (x, y), keys_ini, total_combinations.T)))
    return total_combinations
#fin función

if __name__ == "__main__":
    print(generate_combinations("Isotropic", "Eigen", (0, 1), (0.01, 1)))
