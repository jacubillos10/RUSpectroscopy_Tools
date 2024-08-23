from datamodules import preproc
from datamodules import mutual_info
from datamodules import linear_reg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    print("Uso del programa: python3 test_datamodules.py [Estructura Cristalina]")
    print("Coloque un número entero para estructura cristalina o la palabra 'full' para usar todos los datos")
    raise IndexError("El programa se debe correr con un solo argumento")
elif sys.argv[1] == "full":
    datos_nueva_full = pd.read_csv("output_data/a_Unif_30k.csv", delimiter=",", on_bad_lines='skip')
    datos_antigua_full = pd.read_csv("output_data/l_Unif_30k.csv", delimiter=",", on_bad_lines='skip')
    casillas_variables = 0
    datos_antigua = datos_antigua_full
    datos_nueva = datos_nueva_full
else:
    datos_nueva_full = pd.read_csv("output_data/a_Unif_30k.csv", delimiter=",", on_bad_lines='skip')
    datos_antigua_full = pd.read_csv("output_data/l_Unif_30k.csv", delimiter=",", on_bad_lines='skip')
    estructura_cristalina = int(sys.argv[1])
    casillas_variables = 4
    datos_antigua = datos_antigua_full[datos_antigua_full["Cry_st"] == estructura_cristalina]
    datos_nueva = datos_nueva_full[datos_nueva_full["Cry_st"] == estructura_cristalina]
#fin if

N_datos = len(datos_nueva)
preproc.normalizar(datos_nueva, datos_nueva.keys()[2:], modo="min-max")
preproc.one_hottear(datos_nueva, ["# Shape", "Cry_st"], casillas_variables)
MI_nuevos = mutual_info.MI(datos_nueva)
mutual_info.graficar_info_mutua(MI_nuevos, "Eigen_" + sys.argv[1], N_datos)

