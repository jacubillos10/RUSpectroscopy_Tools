import numpy as np
import sys
import pandas as pd
import itertools

if len(sys.argv) >= 5:
    print("Coloque tres argumentos")
    raise IndexError("Wrong number of arguments")
#fin if 

cry_st = sys.argv[1]
mode = sys.argv[2]
shape = sys.argv[3]

dict_shape = {"Parallelepiped": 0, "Cylinder": 1, "Ellipsoid": 2}

get_vals = lambda minimum, maximum, N: np.linspace(minimum, maximum, N)

if cry_st == "Isotropic":
    keys_bas = ["K", "mu", "lx", "ly", "lz"]
    if mode == "Omega":
        N = [10, 10, 5, 5, 5]
        min_val = [0.01, 0.01, 0.01, 0.01, 0.01]
        max_val = [1, 1, 0.5, 0.5, 0.5, 0.5]
        dict_generation = dict(map(lambda x, y, z, w: (x, get_vals(y, z, w)), keys_bas, min_val, max_val, N))
        dict_generation["rho"] = get_vals(2.0, 10, 10)
        combi_perm = ["K", "mu", "rho"]
    else:
        N = [30, 30, 5, 5, 5]
        min_val = [0.01, 0.01, 0.01, 0.01, 0.01]
        max_val = [1, 1, 1, 1, 1, 1]
        dict_generation = dict(map(lambda x, y, z, w: (x, get_vals(y, z, w)), keys_bas, min_val, max_val, N))
        combi_perm = ["K", "mu"]
    #fin if
    combinations_param = np.array(tuple(itertools.product(*(dict_generation[key] for key in combi_perm))))
    combinations_param = np.c_[range(len(combinations_param)), combinations_param]
#fin if 

print(combinations_param)
