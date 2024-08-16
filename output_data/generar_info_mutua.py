# %%
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_selection import mutual_info_regression
import pandas as pd

# %%
datos_antigua_full = pd.read_csv("l_Unif_30k.csv", delimiter=",", on_bad_lines='skip')
datos_nueva_full = pd.read_csv("a_Unif_30k.csv", delimiter=",", on_bad_lines='skip')

# %%
#datos_antigua = datos_antigua_full.sample(n =100)
#datos_nueva = datos_nueva_full.sample(n = 100)
datos_antigua = datos_antigua_full
datos_nueva = datos_nueva_full
N_datos = len(datos_nueva)

# %%
datos_nueva.tail()

# %%
def normalizar(d_frame):
    for column in d_frame.keys()[2:]:
        if not sum(d_frame[column]) == 0:
            the_minimus = min(d_frame[column])
            the_maximus = max(d_frame[column])
            d_frame[column] = (d_frame[column] - the_minimus)/(the_maximus - the_minimus)
    #fin for
#fin función

# %%
def one_hottear(d_frame, cols_discretas):
    cols_nuevas = []
    for column in cols_discretas:
        posibles_valores = set(d_frame[column])
        for i in posibles_valores:
            d_frame[column + str(i)] = 0
            d_frame.loc[d_frame[column] == i, column + str(i)] = 1
            cols_nuevas.append(column + str(i))
        #fin for 
        del d_frame[column]
    #fin for 
    d_frame.loc[:, cols_nuevas + [col for col in d_frame.columns if col not in cols_nuevas]]
#fin función

# %%
normalizar(datos_antigua)
normalizar(datos_nueva)
datos_nueva.head()

# %%
one_hottear(datos_antigua, ["# Shape", "Cry_st"])
one_hottear(datos_nueva, ["# Shape", "Cry_st"])
datos_nueva.head()

# %%
# Para X tomaré N frecuencias (dadas po el usuario). Para "y" tomaré un CXX dado por el usuario
def info_mutua(N_freq_disp, Cobj, d_frame, opt = "Lineal"):
    if any("(omega^2)" in x for x in d_frame.keys()):
        key_str = "(omega^2)"
        N_col = 4
    elif any("eig" in x for x in d_frame.keys()):
        key_str = "eig"
        N_col = 3
    else:
        raise KeyError("No hay columnas llamadas eig_X o (omega^2)_X")
    #fin if 
    lista_ini = list(d_frame.keys()[:N_col]) + list(d_frame.keys()[-7:])
    lista_eig_raw = list(filter(lambda x: key_str in x, d_frame.keys()))
    if opt == "Log":
        indices_elegidos = np.logspace(0, np.log10(len(lista_eig_raw) - 1), N_freq_disp, dtype = int)
        indices_elegidos[0] = 0
        lista_eig = list(map(lambda x: key_str + "_" + str(x), indices_elegidos))
    else:
        lista_eig = lista_eig_raw[:N_freq_disp]
    #fin if 
    lista_X = lista_ini + lista_eig
    X = d_frame[lista_X]
    y = d_frame[Cobj]
    resp = mutual_info_regression(X, y, discrete_features=tuple(range(N_col, N_col+7)))
    return dict(map(lambda p: (lista_X[p], resp[p]), range(len(lista_X))))
#fin función

# %% 
N_disp = 10
options = "Lineal"
targets = ["C00", "C11", "C22", "C33", "C44", "C55", "C01", "C02", "C12"]
MI_antiguas = dict()
MI_nuevas = dict()
for i in range(len(targets)):
    target = targets[i]
    MI_antigua_info = info_mutua(N_disp, target, datos_antigua, options)
    MI_nueva_info = info_mutua(N_disp, target, datos_nueva, options)
    MI_antiguas[target] =  MI_antigua_info
    MI_nuevas[target] = MI_nueva_info
#fin for 


MI_antiguas = pd.DataFrame(MI_antiguas)
MI_nuevas = pd.DataFrame(MI_nuevas)

MI_antiguas.to_csv("mutual_info_l.csv")
MI_nuevas.to_csv("mutual_info_a.csv")


"""
Coloque esto de las gráficas en un script por separado que solo puede ser ejecutado en local 


# %%
fig1 = plt.figure(figsize=(30,30))
for i in range(len(targets)):
    num = int(str(33) + str(i+1))
    target = targets[i]
    ax = fig1.add_subplot(num)
    ax.barh(listas_antiguas[i], MI_antiguas[i])
    ax.set_title(target)
plt.savefig("Omega_" + str(N_datos) + "datos.png")

# %%
fig2 = plt.figure(figsize = (30,30))
for i in range(len(targets)):
    num = int(str(33) + str(i+1))
    target = targets[i]
    ax = fig2.add_subplot(num)
    ax.barh(listas_nuevas[i], MI_nuevas[i])
    ax.set_title(target)
plt.savefig("Eigen_" + str(N_datos) + "datos.png")

# %%


"""
#Esto es todo folks!!
