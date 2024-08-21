import numpy as np 
import pandas as pd

def normalizar(dataSet, features, parametros = [], modo = "media-desv"):
    """
    Esta función resta la media y luego divide entre la desviación estandar de cada dato en cada columna. Si se le especifica la media y la desviación en los parámetros
    Usa esa media y desviación dada, de lo contrario, calcula la media y la desviación de cada columna especificada en "features"
    @input: dataSet <pd.DataFrame>: Tabla de datos a la que se le va a dar normalización a la columans dadas
    @input: features <list>: lista de "features" o columnas que se van a normalizar
    @input: (opcional) parametros <dict>: En caso de que se quiera normalizar con una media y una desviación dada se especifican en un diccionario así: {'mileage':{'media': 2.9, 'desviacion': 7}}
    """
    for feature in features:
        if (modo == "min-max") and (not sum(dataSet[feature]) == 0):
            param1 = min(dataSet[feature]) if len(parametros) == 0 else parametros[feature]["minimo"]
            param2 = max(dataSet[feature]) if len(parametros) == 0 else parametros[feature]["maximo"]
            dataSet[feature] = (dataSet[feature] - param1)/(param2 - param1)
        else:
            media = np.mean(dataSet[feature]) if len(parametros) == 0 else parametros[feature]["media"]
            desviacion = np.std(dataSet[feature]) if len(parametros) == 0 else parametros[feature]["desviacion"]
            if desviacion != 0:
                dataSet[feature] = (dataSet[feature] - media)/desviacion
            #fin if 
        #fin if 
#fin función

def one_hottear(d_frame, cols_discretas, valores_fijos = 0):
    """
    Esta función aplica la codificación one-hot a un data frame
    @input: d_frame <pd.DataFrame>: Data frame al que se le va a aplicar codificación one hot
    @input: cols_discretas <iterable>: Lista u otro iterable que contiene los nombred e las columnas con datos discretos. 
    @input: valores_fijos <int>: Si esto es cero los posibles valores de una columna discreta serán el set de los mismos. Si no será el número especificado 
    """
    cols_nuevas = []
    for column in cols_discretas:
        posibles_valores = set(d_frame[column]) if valores_fijos == 0 else range(valores_fijos)
        for i in posibles_valores:
            d_frame[column + str(i)] = 0
            d_frame.loc[d_frame[column] == i, column + str(i)] = 1
            cols_nuevas.append(column + str(i))
        #fin for
        del d_frame[column]
    #fin for
    d_frame.loc[:, cols_nuevas + [col for col in d_frame.columns if col not in cols_nuevas]]
#fin función
