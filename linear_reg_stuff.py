from datamodules import preproc
from datamodules import mutual_info
from datamodules import linear_reg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 3:
    print("Uso del programa: python3 test_datamodules.py [Estructura Cristalina] [Nombre del archivo dentro de input_data]")
    print("Coloque un número entero para estructura cristalina o la palabra 'full' para usar todos los datos")
    raise IndexError("El programa se debe correr con solo dos argumentos")
else:
    nombre_archivo = "output_data/" + sys.argv[2]
    dict_graficos = linear_reg.generar_MSE_multiples_frecuencias(4, 65, nombre_archivo, sysarg = sys.argv[1])
#fin if

linear_reg.generar_graficas_freq_multiples(dict_graficos, "MCEP", sys.argv[1], "eigen")

