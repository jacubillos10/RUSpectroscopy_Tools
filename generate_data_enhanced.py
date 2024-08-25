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
    tol = 1e-7
    tries = 0
    maxTry = 100
    vals = np.ones(7)
    norma_gamma = 1; norma_E = 1;
    while any((abs(vals[i]) > tol for i in range(6))) and tries < maxTry and norma_gamma > tol and norma_E > tol:

