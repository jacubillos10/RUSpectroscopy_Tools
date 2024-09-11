import numpy as np
import rusmodules
from rusmodules import rus
from datamodules import constant_generator
import scipy
from scipy import linalg 
import os
from csv import writer
import sys

if len(sys.argv) != 3:
    raise IndexError("Coloque dos argumentos")
#fin if 
N_datos_generar = int(sys.argv[1])
np.set_printoptions(suppress = True)
C_ranks = (0, 1) #Al usar distribución uniforme estos son los rangos de los C principales, al usar Gaussiana estos son la media y desviación respectivamente-
dim_min = (0.01, 0.01, 0.01)
dim_max = (0.5, 0.5, 0.5)
Density = (2.0, 10)
write_header = False
opcion_gen = sys.argv[2]
lista_cryst = ["Orthorombic", "Tetragonal", "Cubic", "Isotropic"]
Shape_Names = ["Parallelepiped", "Cylinder", "Ellipsoid"]
feasibility_Names = ["No", "Yes"]

input_data = { 
                "Dimensions": 
                {
                    "Min": dim_min,
                    "Max": dim_max
                },
                "C_rank": C_ranks,
                "Density": Density,
                "Crystal_structure": 3,
                "Shape": 0,
                "Verbose": False,
                "N_freq": 100,
                "distribution": 0,   #Cambiar esta linea al cambiar de distribución
                "Ng": 8, 
                "options": opcion_gen
              }
distt = ("Unif", "Gauss")
pid = os.getpid()
if write_header:
    name_h = "Header"
else:
    name_h = str(pid)
#fin if
nombre_archivo = "input_data/f_" + opcion_gen  + "_" + name_h + ".csv"
#nombre_archivo = "input_data/bf_" + distt[input_data["distribution"]] + ".csv" 

def generate_eigenvalues(Dimensions, C_rank, Density, Crystal_structure, Shape, N_freq, Ng, distribution, options, Verbose = False):
    alpha = (1, np.pi/4, np.pi/6)
    tol = 1e-7
    dims = np.random.uniform(Dimensions["Min"], Dimensions["Max"])
    vol = alpha[Shape]*np.prod(dims)
    C = constant_generator.generate_C_matrix(C_rank[0], C_rank[1], Crystal_structure, distribution)
    rho = np.random.uniform(Density[0], Density[1])
    dims_adim = dims/(vol**(1/3))
    gamma = rus.gamma_matrix(Ng, C, dims_adim, Shape)
    E = rus.E_matrix(Ng, Shape)
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

    if N_freq == "all":
        N_freq = len(vals) - 6
    #fin if
    C_reshaped = np.concatenate([C[i, i:] for i in range(6)])
    eigenvals = vals[6:N_freq+6]
    freqs_2 = eigenvals/(rho * vol**(2/3)) 
    str_Shape = Shape_Names[Shape]
    str_Crystal_structure = lista_cryst[Crystal_structure]
    if options == "Omega":
        resp = [Shape, Crystal_structure, feasible, rho, *dims, *C_reshaped, *freqs_2] 
    else: 
        resp = [Shape, Crystal_structure, feasible, *dims_adim, *C_reshaped, *eigenvals]
    return resp
#fin función

def generate_keys(N_vals, options):
    keys_ini = ["Shape", "Cry_st", "Feasibility", "Density", "Lx", "Ly", "Lz"]
    keys_ini_adim = ["Shape", "Cry_st", "Feasibility", "bx", "by", "bz"]
    keys_C = sum(map(lambda x: list(map(lambda y: "C" + str(x) + str(y) , range(x,6))), range(6)), [])
    keys_eigenvals = list(map(lambda x: "eig_" + str(x), range(N_vals)))
    keys_freq = list(map(lambda x: "(omega^2)_" + str(x), range(N_vals)))
    if options == "Omega":
        keys_finale = keys_ini + keys_C + keys_freq
    else: 
        keys_finale = keys_ini_adim + keys_C + keys_eigenvals
    #fin if 
    return keys_finale
#fin función

keys = generate_keys(input_data["N_freq"], opcion_gen)
keys_str = ",".join(keys)

if write_header:
    datos = generate_eigenvalues(**input_data)
    datos[0] = Shape_Names[int(datos[0])]
    datos[1] = lista_cryst[int(datos[1])]
    datos[2] = feasibility_Names[int(datos[2])] 
    with open(nombre_archivo, "w+t") as f:
        print(len(keys), len(datos))
        f.write(keys_str + "\n")
        #writer_object = writer(f)
        #writer_object.writerow(datos)
else:
    for i in range(N_datos_generar):
        input_data["Shape"] = np.random.randint(0, 3)
        #input_data["Crystal_structure"] = np.random.randint(0,4)
        datos = generate_eigenvalues(**input_data)
        datos[0] = Shape_Names[int(datos[0])]
        datos[1] = lista_cryst[int(datos[1])]
        datos[2] = feasibility_Names[int(datos[2])] 
        with open(nombre_archivo, "a+t") as f:
            writer_object = writer(f)
            writer_object.writerow(datos)

 
