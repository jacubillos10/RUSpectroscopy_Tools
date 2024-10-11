import numpy as np
import rusmodules
from rusmodules import rus
import data_generator
import scipy
from scipy import linalg 
import os

np.set_printoptions(suppress = True)
C_ranks = (0.3, 5.6) #Al usar distribución uniforme estos son los rangos de los C principales, al usar Gaussiana estos son la media y desviación respectivamente-
dim_min = (0.01, 0.01, 0.01)
dim_max = (0.5, 0.5, 0.5)
Density = (2.0, 10)
write_header = False

input_data = { 
                "Dimensions": 
                {
                    "Min": dim_min,
                    "Max": dim_max
                },
                "C_rank": C_ranks,
                "Density": Density,
                "Crystal_structure": 0,
                "Shape": 0,
                "Verbose": False,
                "N_freq": 500,
                "distribution": 0,   #Cambiar esta linea al cambiar de distribución
                "Ng": 14
              }
distt = ("Unif", "Gauss")
pid = os.getpid()
nombre_archivo = "input_data/d_" + distt[input_data["distribution"]] + "_" + str(pid) + ".csv"
nombre_archivo_adim = "input_data/b_" + distt[input_data["distribution"]] + "_" + str(pid) +".csv" 

def generate_eigenvalues(Dimensions, C_rank, Density, Crystal_structure, Shape, N_freq, Ng, distribution, Verbose = False):
    alpha = (1, np.pi/4, np.pi/6)
    dims = np.random.uniform(Dimensions["Min"], Dimensions["Max"])
    vol = alpha[Shape]*np.prod(dims)
    dims_adim = dims/(vol**(1/3))
    C = data_generator.generate_C_matrix(C_rank[0], C_rank[1], Crystal_structure, distribution)
    rho = np.random.uniform(Density[0], Density[1])
    gamma = rus.gamma_matrix(Ng, C, dims_adim, Shape)
    E = rus.E_matrix(Ng, Shape)
    vals, vects = scipy.linalg.eigh(a = gamma, b = E)
    norma_gamma = np.linalg.norm(gamma - gamma.T)
    norma_E = np.linalg.norm(E - E.T)
    tol = 1e-7
    tries = 0
    maxTry = 100
    if Verbose:
        print("**** C_matrix: *****")
        print(C)
        print("*** dimensions: ***")
        print(dims)
        print("*** end verbose ***")
    #fin if 
    if abs(norma_gamma) > tol or abs(norma_E) > tol:
        print("Norma gamma: ", norma_gamma)
        print("Norma E: ", norma_E)
        raise ArithmeticError("Either gamma or E is non-symetric")
    #fin if
    while any((abs(vals[i]) > tol for i in range(6))) and tries < maxTry:
        print("WARNING: Any of the fisrt sixth eigenvalues is not near zero. Recomputing...")
        dims = np.random.uniform(Dimensions["Min"], Dimensions["Max"])
        vol = alpha[Shape]*np.prod(dims)
        dims_adim = dims/(vol**(1/3))
        C = data_generator.generate_C_matrix(C_rank[0], C_rank[1], Crystal_structure)
        rho = np.random.uniform(Density[0], Density[1])
        gamma = rus.gamma_matrix(Ng, C, dims_adim, Shape)
        E = rus.E_matrix(Ng, Shape)
        vals, vects = scipy.linalg.eigh(a = gamma, b = E)
        tries = tries + 1
    #fin if
    if N_freq == "all":
        N_freq = len(vals) - 6
    #fin if
    C_reshaped = np.concatenate([C[i, i:] for i in range(6)])
    #print(C_reshaped)
    eigenvals = vals[6:N_freq+6]
    freqs_2 = eigenvals/(rho * vol**(2/3))
    return [np.array([np.r_[Shape, Crystal_structure, dims_adim, C_reshaped, eigenvals]]),
            np.array([np.r_[Shape, Crystal_structure, rho, dims, C_reshaped, freqs_2]])]
#fin funcion

def generate_keys(N_vals):
    keys_ini = ["Shape", "Cry_st", "Density", "Lx", "Ly", "Lz"]
    keys_ini_adim = ["Shape", "Cry_st", "bx", "by", "bz"]
    keys_C = sum(map(lambda x: list(map(lambda y: "C" + str(x) + str(y) , range(x,6))), range(6)), [])
    keys_eigenvals = list(map(lambda x: "eig_" + str(x), range(N_vals)))
    keys_freq = list(map(lambda x: "(omega^2)_" + str(x), range(N_vals)))
    return [keys_ini_adim + keys_C + keys_eigenvals,
            keys_ini + keys_C + keys_freq]
#fin función

keys = generate_keys(input_data["N_freq"])
keys_adim = ",".join(keys[0])
keys_str = ",".join(keys[1])

if write_header:
    datos = generate_eigenvalues(**input_data)
    with open(nombre_archivo, "w+t") as f:
        print(len(keys[1]), len(datos[1][0]))
        np.savetxt(f, datos[1], header = keys_str, delimiter = ",")
    with open(nombre_archivo_adim, "w+t") as f:
        print(len(keys[0]), len(datos[0][0]))
        np.savetxt(f, datos[0], header = keys_adim, delimiter = ",")
else:
    for i in range(8):
        input_data["Shape"] = np.random.randint(0, 3)
        input_data["Crystal_structure"] = np.random.randint(0,4) 
        datos = generate_eigenvalues(**input_data)
        with open(nombre_archivo, "a+t") as f:
            np.savetxt(f, datos[1], delimiter = ",")
        with open(nombre_archivo_adim, "a+t") as f:
            np.savetxt(f, datos[0], delimiter = ",")
#fin if

