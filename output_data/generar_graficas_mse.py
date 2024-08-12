## %%
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split

## %% 
freq_min = 4
freq_max = 65
nombre_archivo_a = "a_Unif.csv"
nombre_archivo_l = "l_Unif.csv"
## %%

#Definimos las funciones de normalizar y de one-hot:
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

def normalizar(dataSet, features, parametros = []):
    """
    Esta función resta la media y luego divide entre la desviación estandar de cada dato en cada columna. Si se le especifica la media y la desviación en los parámetros
    Usa esa media y desviación dada, de lo contrario, calcula la media y la desviación de cada columna especificada en "features"
    @input: dataSet <pd.DataFrame>: Tabla de datos a la que se le va a dar normalización a la columans dadas
    @input: features <list>: lista de "features" o columnas que se van a normalizar
    @input: (opcional) parametros <dict>: En caso de que se quiera normalizar con una media y una desviación dada se especifican en un diccionario así: {'mileage':{'media': 2.9, 'desviacion': 7}}
    """
    if len(parametros) == 0:
        for feature in features:
            media = np.mean(dataSet[feature])
            desviacion = np.std(dataSet[feature])
            if desviacion != 0:
                dataSet[feature] = (dataSet[feature] - media)/desviacion
            #fin if
        #fin for
    #fin if
    else:
        for feature in features:
            media = parametros[feature]['media']
            desviacion = parametros[feature]['desviacion']
            if desviacion != 0:
                dataSet[feature] = (dataSet[feature] - media)/desviacion
            #fin if
        #fin for
#fin función

## %%

#Creamos una función que reciba el número de valores propios a usar y devuelva las métricas MSE

def MSE(archivo_eigen, archivo_omega, N):
    datos_antigua = pd.read_csv(archivo_omega, delimiter=",", on_bad_lines='skip', usecols=tuple(range(26 + N)))
    datos_nueva = pd.read_csv(archivo_eigen, delimiter=",", on_bad_lines='skip', usecols=tuple(range(26 + N)))
    N_datos = len(datos_nueva)
    columnas_normalizar_a = list(datos_nueva.keys()[2:])
    columnas_normalizar_l = list(datos_antigua.keys()[2:])
    one_hottear(datos_antigua, ["# Shape", "Cry_st"])
    one_hottear(datos_nueva, ["# Shape", "Cry_st"])
    Datos_train_l, Datos_test_l = train_test_split(datos_antigua, test_size = 0.4)
    Datos_train_a, Datos_test_a = train_test_split(datos_nueva, test_size = 0.4)
    Estadisticos_train_l= dict(map(lambda x: (x, {'media': np.mean(Datos_train_l[x]), 'desviacion': np.std(Datos_train_l[x])}), Datos_train_l.keys()))
    Estadisticos_train_a= dict(map(lambda x: (x, {'media': np.mean(Datos_train_a[x]), 'desviacion': np.std(Datos_train_a[x])}), Datos_train_a.keys()))
    normalizar(Datos_train_l, columnas_normalizar_l)
    normalizar(Datos_train_a, columnas_normalizar_a)
    normalizar(Datos_test_l, columnas_normalizar_l, Estadisticos_train_l)
    normalizar(Datos_test_a, columnas_normalizar_a, Estadisticos_train_a)
    features_discretos = list(filter(lambda x: "Shape" in x or "CrySt" in x, Datos_train_a.keys()))
    features_X_a = list(filter(lambda x: "eig" in x, Datos_train_a.keys()))
    features_X_l = list(filter(lambda x: "omega" in x, Datos_train_l.keys()))
    features_geo_l = ["Density", "Lx", "Ly", "Lz"]
    features_geo_a = ["bx", "by", "bz"]
    features_a = features_discretos + features_geo_a + features_X_a
    features_l = features_discretos + features_geo_l + features_X_l
    X_train_a = Datos_train_a[features_a]
    X_train_l = Datos_test_l[features_l]
    X_test_a = Datos_test_a[features_a]
    X_test_l = Datos_test_l[features_l]
    targets = ["C00", "C11", "C22", "C33", "C44", "C55", "C01", "C02", "C12"]
    W_a = dict()
    W_l = dict()
    for target in targets:
        y_train_a = Datos_train_a[target]
        y_train_l = Datos_test_l[target]
        y_test_a = Datos_test_a[target]
        y_test_l = Datos_test_l[target]
        modelo_a = LinearRegression()
        modelo_l = LinearRegression()
        modelo_a.fit(X_train_a, y_train_a)
        modelo_l.fit(X_train_l, y_train_l)
        w_a = [modelo_a.intercept_, *modelo_a.coef_]
        w_l = [modelo_l.intercept_, *modelo_l.coef_]
        y_gorro_test_a = X_test_a.dot(w_a[1:]) + w_a[0]
        y_gorro_test_l = X_test_l.dot(w_l[1:]) + w_l[0]
        W_a[target] = dict(map(lambda x: (features_a[x], w_a[1:][x]), range(len(features_a))))
        W_l[target] = dict(map(lambda x: (features_l[x], w_l[1:][x]), range(len(features_l))))
        W_a[target]["intercepto"] = w_a[0]
        W_l[target]["intercepto"] = w_l[0]
        W_a[target]["MSE"] = (1/(len(y_test_a)))*(y_test_a - y_gorro_test_a).dot(y_test_a - y_gorro_test_a)
        W_l[target]["MSE"] = (1/(len(y_test_l)))*(y_test_l - y_gorro_test_l).dot(y_test_l - y_gorro_test_l)
        media_a = Estadisticos_train_a[target]["media"]
        media_l = Estadisticos_train_l[target]["media"]
        desv_a = Estadisticos_train_a[target]["desviacion"]
        desv_l = Estadisticos_train_l[target]["desviacion"]
        nu_test_a = (media_a + desv_a*y_gorro_test_a)/(media_a + desv_a*y_test_a)
        nu_test_l = (media_l + desv_l*y_gorro_test_l)/(media_l + desv_l*y_test_l)
        W_a[target]["MSE_p"] = (1/(len(y_test_a)))*(1 - nu_test_a).dot(1 - nu_test_a)
        W_l[target]["MSE_p"] = (1/(len(y_test_l)))*(1 - nu_test_l).dot(1 - nu_test_l)

    #fin for
    W_l = pd.DataFrame(W_l)
    W_a = pd.DataFrame(W_a)
    return {"eigen": W_a, "antigua": W_a}
#fin función

## %%
def sacar_datos_MSE(freq_min, freq_max, archivo_eigen, archivo_omega):
    datos_fin = MSE(archivo_eigen, archivo_omega, freq_min)
    resp_a = pd.DataFrame(); resp_ap = pd.DataFrame(); resp_l = pd.DataFrame(); resp_lp = pd.DataFrame();
    for i in range(freq_min, freq_max):
        print("Sacando MSE para " + str(i) +" frecuencias")
        datos_fin = MSE(archivo_eigen, archivo_omega, i)
        resp_a[i] = datos_fin["eigen"].T["MSE"]
        resp_ap[i] = datos_fin["eigen"].T["MSE_p"]
        resp_l[i] = datos_fin["antigua"].T["MSE"]
        resp_lp[i] = datos_fin["antigua"].T["MSE_p"]
    #fin for 
    return {"eigen": {"MSE": resp_a.T, "MSE_p": resp_ap.T}, "antigua": {"MSE": resp_l.T, "MSE_p": resp_lp.T}}
#fin función

##%%

def generar_graficas(dataFrame):
    for key1 in dataFrame.keys():
        for key2 in dataFrame[key1].keys():
            fig = plt.figure(figsize=(30,30))
            for i, CXX in enumerate(dataFrame[key1][key2].keys()):
                num = int(str(33) + str(i+1))
                ax = fig.add_subplot(num)
                ax.scatter(dataFrame[key1][key2][CXX].index, dataFrame[key1][key2][CXX])
                ax.set_title(CXX)
                ax.set_xlabel("Número de frecuencias usadas para entrenar")
                ax.set_ylabel("MSE")
            #fin for
            plt.savefig(key1 + "_" + key2 + "_scatter.png")
        #fin for 
    #fin for 
#fin función
                

datos_MSE = sacar_datos_MSE(freq_min, freq_max, nombre_archivo_a, nombre_archivo_l)
generar_graficas(datos_MSE)
