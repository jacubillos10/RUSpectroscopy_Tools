from datamodules import preproc
from datamodules import mutual_info
from datamodules import linear_reg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

path_datos = sys.argv[1]

if len(sys.argv) != 3:
    print("Uso del programa: python3 test_datamodules.py [Path archivo] [Estructura Cristalina]")
    print("Coloque un número entero para estructura cristalina o la palabra 'full' para usar todos los datos")
    raise IndexError("El programa se debe correr con 3 argumentos")
elif sys.argv[2] == "full":
    datos_full = pd.read_csv(path_datos, delimiter=",", on_bad_lines='skip')
    casillas_variables = 0
    datos = datos_full
else:
    datos_full = pd.read_csv(path_datos, delimiter=",", on_bad_lines='skip')
    estructura_cristalina = sys.argv[2]
    casillas_variables = 4
    datos = datos_full[datos_full["Cry_st"] == estructura_cristalina]
#fin if

try:
    del datos["Feasibility"]
except KeyError:
    print("No se encontró el feature de Factibilidad. Procediendo...")
#fin exception

print(datos.keys())
lista_one_hot = list(filter(lambda x: any((y in x for y in ["Shape", "Cry_st"])), datos.keys()))
print(lista_one_hot)
N_datos = len(datos)
preproc.normalizar(datos, datos.keys()[2:], modo="min-max")
preproc.one_hottear(datos, lista_one_hot, casillas_variables)
MI_data = mutual_info.MI(datos)
mutual_info.graficar_info_mutua(MI_data, estructura_cristalina+ "_", N_datos)

