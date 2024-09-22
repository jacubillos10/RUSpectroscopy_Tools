import numpy as np
import pandas as pd
import itertools

def generate_combinations(cry_st, mode,  par_const, par_dim, N_freq=100, options = "default"):
    defaults = {"Eigen": (30, 5), "Omega": (10, 5)}
    get_vals = lambda minimum, maximum, N: np.linspace(minimum, maximum, N)
    if cry_st == "Isotropic":
        opts_dat = defaults[mode] if options == "default" else options
        keys_bas = ["K", "mu", "l"]
        min_val = [par_const[0], par_const[0], par_dim[0]]
        max_val = [par_const[1], par_const[1], par_dim[1]]
        if mode == "Omega":
            N = (opts_dat[0], opts_dat[0], opts_dat[1])
            dict_generation = dict(map(lambda x, y, z, w: (x, get_vals(y, z, w)), keys_bas, min_val, max_val, N))
            dict_generation["rho"] = get_vals(2.0, 10, opts_dat[0])
            combi_perm = ["K", "mu", "rho"]
        else:
            N = (opts_dat[0], opts_dat[0], opts_dat[1])
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
    name_key_eigen = "(omega^2)_" if "rho" in total_combinations.keys() else "eig"
    keys_obj = list(map(lambda x: name_key_eigen + str(x), range(N_freq)))
    nones_feasibility = pd.DataFrame({"Feasible": [None] * len(total_combinations)})
    nones_eigen = pd.DataFrame(dict(map(lambda x: (x, [None] * len(total_combinations)), keys_obj))) 
    total_combinations = pd.concat((total_combinations, nones_feasibility, nones_eigen), axis = 1) 
    return total_combinations
#fin función

if __name__ == "__main__":
    #print(generate_combinations("Isotropic", "Eigen", (0, 1), (0.01, 1)))
    a = generate_combinations("Isotropic", "Eigen", (0, 1), (1, 2), options = (3, 2))
    b = generate_combinations("Isotropic", "Eigen", (5,6), (7,8), options = (6,3))
    c = pd.concat((a, b), axis = 0, ignore_index = True)
    print(c.iloc[30:90])
