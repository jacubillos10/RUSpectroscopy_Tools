import numpy as np
import os
import pandas as pd 
path_repo = "../QuantumMaterialsUniandesRUS/5 Experimental Data/Test Samples/Iso/"

materiales =(dr for dr in os.listdir(path_repo) if os.path.isdir(os.path.join(path_repo, dr)))
lista_data_frame = []
for material in materiales:
    path_material = path_repo + material
    lista_archivos = os.listdir(path_material)
    archivos_freq = tuple(filter(lambda x: material.lower() + "_freq" in x
                                 or material + "_freq" in x, lista_archivos))
    archivo_freq = archivos_freq[0]
    freqs = np.genfromtxt(path_material + "/" + archivo_freq)
    dict_data = dict(map(lambda x: ("(omega^2)_" + str(x), (2*np.pi*freqs[x])**2), range(len(freqs))))
    dict_data["Cry_st"] = "Isotropic"
    archivos_dims = tuple(filter(lambda x: material.lower() + "_prop" in x
                                 or material + "_prop" in x, lista_archivos))
    archivo_dim = archivos_dims[0]
    dims_rho = np.genfromtxt(path_material + "/" + archivo_dim)
    rho = dims_rho[0]
    dims = dims_rho[1:]
    dict_data["rho"] = rho
    dict_dims = dict(map(lambda x, y: ("d" + y, x), dims, ("x", "y", "z")))
    dict_data = {**dict_data, **dict_dims}
    archivos_out = tuple(filter(lambda x: material.lower() + "_out" in x
                                 or material + "_out" in x, lista_archivos))
    lista_data_frame.append(dict_data)
#fin for 

df_data = pd.DataFrame(lista_data_frame)
df_data.to_csv("data_exp.csv")
