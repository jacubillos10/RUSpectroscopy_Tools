from datamodules import preproc
from datamodules import mutual_info
from datamodules import linear_reg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

nombre_personalizado = "eigen_unif"
cols_disc = ["Shape"]
#cols_disc = ["# Shape", "Cry_st"]

if len(sys.argv) != 3:
    print("Uso del programa: python3 test_datamodules.py [Nombre del archivo dentro de input_data] [Estructura Cristalina] ")
    print("Coloque la estructura cristalina o la palabra 'full' para usar todos los datos")
    raise IndexError("El programa se debe correr con solo dos argumentos")
else:
    nombre_archivo = sys.argv[1]
    datos = pd.read_csv(nombre_archivo, delimiter = ",")
    #datos_test = pd.read_csv("input_data/f_Eigen_Header.csv", delimiter = ",")
    #datos_test["K"] = datos_test["C01"] + (2/3)*datos_test["C44"]
    #for CXX in ['C00', 'C01', 'C02', 'C11', 'C12', 'C22', 'C33', 'C55'] + preproc.terciary_targets:
    #    del datos_test[CXX]
    #fin for}
    datos["Shape"] = datos["shape"]
    del datos["shape"]
    #print(datos_test.head())
    dict_graficos = linear_reg.generar_MSE_multiples_frecuencias(4, 65, datos, sysarg = sys.argv[2], cols_discretas=cols_disc, targets = ["K", "C44"], ignore_cols = ["Cry_st"])

linear_reg.generar_graficas_freq_multiples(dict_graficos, "MCEP", sys.argv[2], nombre_personalizado)
