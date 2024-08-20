# %%
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

# %%
datos_antigua_full = pd.read_csv("output_data/l_Unif_30k.csv", delimiter=",", on_bad_lines='skip')
datos_nueva_full = pd.read_csv("output_data/a_Unif_30k.csv", delimiter=",", on_bad_lines='skip')
if len(sys.argv) != 2:
    print("Uso del programa: python3 regresion_lineal.py [Estructura Cristalina]")
    print("Coloque un número entero para estructura cristalina o la palabra 'full' para usar todos los datos")
    raise IndexError("El programa se debe correr con un solo argumento")
elif sys.argv[1] == "full":
    casillas_variables = True
    datos_antigua = datos_antigua_full
    datos_nueva = datos_nueva_full
else: 
    estructura_cristalina = int(sys.argv[1])
    casillas_variables = False
    datos_antigua = datos_antigua_full[datos_antigua_full["Cry_st"] == estructura_cristalina]
    datos_nueva = datos_nueva_full[datos_nueva_full["Cry_st"] == estructura_cristalina]
#fin if 
N_datos = len(datos_nueva)
columnas_normalizar_a = list(datos_nueva.keys()[2:])
columnas_normalizar_l = list(datos_antigua.keys()[2:])
#print(columnas_normalizar_a)

def one_hottear(d_frame, cols_discretas, fijo = True):
    cols_nuevas = []
    for column in cols_discretas:
        posibles_valores = set(d_frame[column]) if fijo else range(4)
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
one_hottear(datos_antigua, ["# Shape", "Cry_st"])
one_hottear(datos_nueva, ["# Shape", "Cry_st"])

# %%
Datos_train_l, Datos_test_l = train_test_split(datos_antigua, test_size = 0.4)
Datos_train_a, Datos_test_a = train_test_split(datos_nueva, test_size = 0.4)

# %%
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

# %%
Estadisticos_train_l= dict(map(lambda x: (x, {'media': np.mean(Datos_train_l[x]), 'desviacion': np.std(Datos_train_l[x])}), Datos_train_l.keys()))
Estadisticos_train_a= dict(map(lambda x: (x, {'media': np.mean(Datos_train_a[x]), 'desviacion': np.std(Datos_train_a[x])}), Datos_train_a.keys()))

# %%
normalizar(Datos_train_l, columnas_normalizar_l)
normalizar(Datos_train_a, columnas_normalizar_a)
normalizar(Datos_test_l, columnas_normalizar_l, Estadisticos_train_l)
normalizar(Datos_test_a, columnas_normalizar_a, Estadisticos_train_a)

# %%
"""
TODO: 
1) Hacer una lista con los features que serán nuestros X. Escoger cualquiera de los CXX y ese será nuestro 'y' (luego haremos un loop para cubrir todos los CXX)
2) Hacer una regresión lineal con cada uno de los CXX y reportar un MSE por cada uno de los CXX
"""
features_discretos = list(filter(lambda x: "Shape" in x or "CrySt" in x, Datos_train_a.keys()))
features_X_a = list(filter(lambda x: "eig" in x, Datos_train_a.keys()))
features_X_l = list(filter(lambda x: "omega" in x, Datos_train_l.keys()))
features_geo_l = ["Density", "Lx", "Ly", "Lz"]
features_geo_a = ["bx", "by", "bz"]
features_a = features_discretos + features_geo_a + features_X_a
features_l = features_discretos + features_geo_l + features_X_l
#print(features_a)
#print(features_l)

# %%
X_train_a = Datos_train_a[features_a]
X_train_l = Datos_test_l[features_l]
X_test_a = Datos_test_a[features_a]
X_test_l = Datos_test_l[features_l]

# %%
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
#fin for 
W_l = pd.DataFrame(W_l)
W_a = pd.DataFrame(W_a)

# %%
print("b y pesos para el problema adimensionalizado: ")
print(W_a)
nombre_archivo_a = "R_lineal_a_" + "Cry_st:" + sys.argv[1] + "_" + str(N_datos) + ".csv"
W_a.to_csv(nombre_archivo_a)

# %%
print("b y pesos para el problema a la antigua")
print(W_l)
nombre_archivo_l = "R_lineal_l_" + "Cry_st:" + sys.argv[1] + "_" + str(N_datos) + ".csv"
W_l.to_csv(nombre_archivo_l)

# %%
