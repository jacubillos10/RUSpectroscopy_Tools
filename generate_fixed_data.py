import numpy as np
from datamodules import parameter_combinator
from rusmodules import rus
from datamodules import preproc
import scipy
import sys
import os 

targets_default = preproc.targets_default

def generate_data_frame(row_indexes, d_frame, cry_st, Ng=8, N_freq = 100, shape = "Parallelepiped"):
    tol = 1e-7
    name_key_eigen = "(omega^2)_" if "rho" in d_frame.keys() else "eig"
    keys_obj = list(map(lambda x: name_key_eigen + str(x), range(N_freq)))
    dict_shape = {"Parallelepiped": 0, "Cylinder": 1, "Ellipsoid": 2}
    alphas = (1, np.pi/4, np.pi/6)
    alpha = alphas[dict_shape[shape]]
    iterator = range(len(d_frame)) if row_indexes == "full" else range(row_indexes[0], row_indexes[1])
    for i in iterator:
        row = d_frame.iloc[i]
        C = np.zeros((6,6))
        vol = alpha*row["lx"]*row["ly"]*row["lz"]
        dims_adim = np.array([row["lx"], row["ly"], row["lz"]])/(vol**(1/3))
        if cry_st == "Isotropic":
            C_prim = np.array(list(map(lambda x: list(map(lambda y: 1 if x == y and x < 3 else 0, range(6))), range(6))))
            C_lambda = np.array(list(map(lambda x: list(map(lambda y: 1 if x != y and x < 3 and y < 3 else 0, range(6))), range(6))))
            C_mu = np.array(list(map(lambda x: list(map(lambda y: 1 if x == y and x >= 3 else 0, range(6))), range(6))))
            C_prim = (row["K"] + (4/3)*row["mu"])*C_prim
            C_lambda = (row["K"] - (2/3)*row["mu"])*C_lambda
            C_mu = row["mu"]*C_mu
            C = C + C_prim + C_lambda + C_mu
        #fin if
        gamma = rus.gamma_matrix(Ng, C, dims_adim, dict_shape[shape])
        E = rus.E_matrix(Ng, dict_shape[shape])
        vals, vects = scipy.linalg.eigh(a = gamma, b = E)
        norma_gamma = np.linalg.norm(gamma - gamma.T)
        norma_E = np.linalg.norm(E - E.T)
        if any((abs(vals[i]) > tol for i in range(6))):
            #print("WARNING: One of the first sixth eigenvalues is not zero. Recomputing...")
            #print("Iteration state: ", tries, " of ", maxTry)
            feasible = 0
        elif norma_gamma > tol or norma_E > tol:
            print("Norma gamma: ", norma_gamma)
            print("Norma E: ", norma_E)
            print("Warning: Either gamma or E is non-symetric")
            print("Iteration state: ", tries, " of ", maxTry)
            feasible = 0
        else:
            feasible = 1
        #fin if
        l_feasible = ["No", "Yes", "OJO!!"]
        d_frame.loc[i, "Feasible"] = l_feasible[feasible]
        if N_freq == "all":
            N_freq = len(vals) - 6
        #fin if
        eigenvals = vals[6:N_freq+6]
        if "rho" in d_frame.keys():
            freqs_2 = eigenvals/(row["rho"] * vol**(2/3))
            d_frame.loc[i, keys_obj] = freqs_2
        else:
            d_frame.loc[i, keys_obj] = eigenvals
        #fin if 
    #fin for 
#fin función

d_frame = parameter_combinator.generate_combinations("Isotropic", sys.argv[3], (0,1), (0.01, 1))
try:
    generate_data_frame((int(sys.argv[1]), int(sys.argv[2])), d_frame, "Isotropic", shape = sys.argv[4])
    d_frame.iloc[int(sys.argv[1]):int(sys.argv[2])].to_csv("input_data/Combinatoriales_"+str(os.getpid()) + "_" + sys.argv[3] + "_" + sys.argv[4] + ".csv")
except ValueError:
    generate_data_frame(sys.argv[1], d_frame, "Isotropic", shape = sys.argv[4]) 
    d_frame.to_csv("input_data/Combinatoriales_"+str(os.getpid()) + "_" + sys.argv[3] + "_" + sys.argv[4] + ".csv")
