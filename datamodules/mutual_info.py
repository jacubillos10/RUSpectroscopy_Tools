import numpy as np 
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from . import preproc


def info_mutua(N_freq_disp, Cobj, d_frame, opt = "Lineal"):
    """
    Esta función determina la información mutua entre los datos de una constante especificada en Cobj y los demás features. 
    @input N_freq_disp <int>: Número de frecuencias o valores propios que van a entrar en el cálculo de información mutua
    @input Cobj <string>: Nombre de la constante a la que se va a calcular la información mutua
    @input d_frame <pd.DataFrame>: Tabla de los datos
    @input opt <string>: Opciones de selección de los valores pripios, Lineal: Escoge los primeros N_freq_disp, Log: Escoge logaritmicamente 
    """
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

N_d = preproc.N_disp_default
o_d = preproc.options_default
t_d = preproc.targets_default 
def MI(datos, N_disp=N_d, targets=t_d, options=o_d):
    """
    Esta función aplica info_mutua a cada una de las constantes espeficicadas en tagets, con los mismos features
    @input datos <pd.DataFrame>: Tabla de datos generados
    @input N_disp <int>: Número de frecuencias o valores propios a los que se les sacará mutual information
    @input targets <iterable strings>: Constantes a los que se les sacará la información mutua
    @input options <string>: "Lineal" para distribuir frecuencias linealmente, "Log" para hacerlo logarítmicamente
    """
    return pd.DataFrame(dict(map(lambda x: (x, info_mutua(N_disp, x, datos, options)), targets)))
#fin función

def graficar_info_mutua(data_MI, nombre, N_datos = "FULL"):
    """
    Esta función grafica la información mutua de 9 constantes con sus respestivos features especificados en la función MI
    @input data_MI <pd.DataFrame>: Data frame que contiene los datos de información mutua
    @input nombre <string>: nombre que queremos que aparezca justo después del 'MI' en el archivo png
    @input N_datos <int>: Número de datos que tiene el dataframe original 
    """
    fig2 = plt.figure(figsize = (20,20))
    targets = tuple(data_MI.keys())
    len_cuadro = int(np.ceil(len(targets)**(1/2)))
    for i in range(len(targets)):
        par_plot = (len_cuadro, len_cuadro, i+1)     
        target = targets[i]
        ax = fig2.add_subplot(*par_plot)
        ax.barh(data_MI.index, data_MI[target])
        ax.set_title(target, fontsize=24)
        ax.tick_params(axis='x', labelsize=21)  # Larger x-labels
        ax.tick_params(axis='y', labelsize=14)  # Slightly larger y-labels
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig("MI_" + nombre + "_" + str(N_datos) + ".png")
    plt.close()
#fin función de graficar


