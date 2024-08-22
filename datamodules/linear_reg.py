import numpy as np 
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from . import preproc

N_d = preproc.N_disp_default
o_d = preproc.options_default
t_d = preproc.targets_default 

def error_porcentual(y, y_gorro, estadisticos, target):
    media = estadisticos[target]["media"]
    desv = estadisticos[target]["desviacion"]
    nu = (media + desv*y_gorro)/(media + desv*y)
    ep = (1/(len(y)))*(1 - nu).dot(1 - nu)
    return ep
#fin función

def regresion_lineal(Datos_train, Datos_test, targets, Estadisticos_train):
    features = list(filter(lambda x: x not in targets, Datos_train.keys()))
    X_train = Datos_train[features]
    X_test = Datos_test[features]
    W = dict()
    features_con_b = ["intercepto"] + features
    for target in targets:
        y_train = Datos_train[target]
        y_test = Datos_test[target]
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        w = [modelo.intercept_, *modelo.coef_]
        y_gorro_test = w[0] + X_test.dot(w[1:])
        W[target] = dict(map(lambda x: (features_con_b[x], w[x]), range(len(features_con_b))))
        W[target]["MSE"] = (1/(len(y_test)))*(y_test - y_gorro_test).dot(y_test - y_gorro_test)
        W[target]["MCEP"] = error_porcentual(y_test, y_gorro_test, Estadisticos_train, target)
    #fin for 
    W = pd.DataFrame(W)
    return W
#fin función

def explorar_N_frecuencias(N, archivo, targets = t_d, cols_discretas = ["# Shape", "Cry_st"], test_p = 0.4, sysarg = "full"):
    datos_full = pd.read_csv(archivo, delimiter = ",", on_bad_lines="skip", usecols=tuple(range(26+N)))
    if sysarg == "full":
        casillas_variables = 0
        datos = datos_full
    else:
        estructura_cristalina = int(sysarg)
        casillas_variables = 4
        datos = datos_full[datos_full["Cry_st"] == estructura_cristalina]
    #fin if 
    N_datos = len(datos)
    cols_normalizar = list(filter(lambda x: x not in cols_discretas, datos.keys())) 
    preproc.one_hottear(datos, cols_discretas)
    Datos_train, Datos_test = train_test_split(datos, test_size = test_p)
    Estadisticos_train = dict(map(lambda x: (x, {"media": np.mean(Datos_train[x]), "desviacion": np.std(Datos_train[x])}), Datos_train.keys()))
    preproc.normalizar(Datos_train, cols_normalizar)
    preproc.normalizar(Datos_test, cols_normalizar, Estadisticos_train)
    return regresion_lineal(Datos_train, Datos_test, targets, Estadisticos_train)
#fin función

def generar_MSE_multiples_frecuencias(freq_min, freq_max, archivo, targets = t_d, cols_discretas = ["# Shape", "Cry_st"], test_p = 0.4, sysarg = "full"):
    resp_MSE = pd.DataFrame(); resp_MCEP = pd.DataFrame();
    for i in range(freq_min, freq_max):
        print("Sacando errores para " + str(i) + " frecuencias")
        datos_W = explorar_N_frecuencias(i, archivo, targets, cols_discretas, test_p, sysarg)
        resp_MSE[i] = datos_W.T["MSE"]
        resp_MCEP[i] = datos_W.T["MCEP"]
    #fin for 
    return {"MSE": resp_MSE.T, "MCEP": resp_MCEP.T}
#fin funcion

def generar_graficas_freq_multiples(dict_Errores, key, sysarg = "full", nombre = "eigen"):
    fig = plt.figure(figsize = (20,20))
    for i, CXX in enumerate(dict_Errores[key].keys()):
        num = int(str(33) + str(i+1))
        ax = fig.add_subplot(num)
        ax.scatter(dict_Errores[key][CXX].index, dict_Errores[key][CXX])
        ax.set_title(CXX, fontsize = 24)
        ax.tick_params(axis='x', labelsize=13)  # Larger x-labels
        ax.tick_params(axis='y', labelsize=12)  # Slightly larger y-labels
        ax.set_xlabel("Número de frecuencias usadas para entrenar")
        ax.set_ylabel(key)
    #fin for 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig(nombre + "_" + key + "Cry_st:" + str(sysarg) + "_scatter.png")
    plt.close()
#fin función
