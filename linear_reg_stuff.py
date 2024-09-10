from datamodules import preproc
from datamodules import mutual_info
from datamodules import linear_reg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

nombre_personalizado = "Iso"
cols_disc = ["Shape", "Cry_st"]
#cols_disc = ["# Shape", "Cry_st"]
if len(sys.argv) != 3:
    print("Uso del programa: python3 test_datamodules.py [Nombre del archivo dentro de input_data] [Estructura Cristalina] ")
    print("Coloque la estructura cristalina o la palabra 'full' para usar todos los datos")
    raise IndexError("El programa se debe correr con solo dos argumentos")
else:
    nombre_archivo = sys.argv[1]
    dict_graficos = linear_reg.generar_MSE_multiples_frecuencias(4, 65, nombre_archivo, sysarg = sys.argv[2], cols_discretas=cols_disc)
#fin if

linear_reg.generar_graficas_freq_multiples(dict_graficos, "MCEP", sys.argv[2], nombre_personalizado)

