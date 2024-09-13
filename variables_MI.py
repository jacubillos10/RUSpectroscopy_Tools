import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import re

# Utility functions to calculate Mutual Information scores
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores_multiple_targets(mi_scores_dict, save_path=None):
    num_targets = len(mi_scores_dict)
    columns_per_row = 2
    num_rows = (num_targets + columns_per_row - 1) // columns_per_row  # Redondear hacia arriba
    
    fig, axes = plt.subplots(num_rows, columns_per_row, figsize=(5*columns_per_row, 5*num_rows))
    axes = axes.flatten()  # Aplanar la matriz de ejes
    
    for i, (target, mi_scores) in enumerate(mi_scores_dict.items()):
        mi_scores = mi_scores.sort_values(ascending=True)
        width = np.arange(len(mi_scores))
        ticks = list(mi_scores.index)
        axes[i].barh(width, mi_scores)
        axes[i].set_yticks(width)
        axes[i].set_yticklabels(ticks)
        axes[i].set_title(f"Mutual Information with Target: {target}")
    
    # Si hay más subplots de los necesarios, ocultar los sobrantes
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()

    if save_path:
        # Asegurarse de que el directorio de guardado existe
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Directorio creado: {save_dir}")
        plt.savefig(save_path)
        print(f"Gráfico guardado en: {save_path}")
    else:
        plt.show()

def main():

    # 
    variables_MI = ["Shape", "Cry_st"]

    # Configurar argparse para aceptar el CSV, múltiples targets y variables
    parser = argparse.ArgumentParser(description="Mostrar la información mutua entre varios targets y varias variables en un único gráfico.")
    parser.add_argument('--csv', type=str, help="Ruta del archivo CSV.", default='data/f_Eigen_Header.csv')
    parser.add_argument('--variables', nargs='+', required=False, help="Columnas del dataset a comparar con los targets", default=None)
    parser.add_argument('--targets', nargs='+', required=True, help="Columnas objetivo para calcular la información mutua")
    parser.add_argument('--save', type=str, help="Ruta donde guardar el gráfico generado.", default='/tmp/MI_multiple_targets.png')

    args = parser.parse_args()
    csv_file = args.csv
    selected_columns = args.variables
    target_columns = args.targets
    save_path = args.save
    
    # Cargar dataset desde el archivo CSV proporcionado por el usuario
    try:
        dataset_df = pd.read_csv(csv_file)
        print(f"Archivo CSV cargado exitosamente desde: {csv_file}")
    except FileNotFoundError:
        print(f"Error: El archivo '{csv_file}' no se encontró.")
        return
    
    if selected_columns is None:
        selected_columns = list(dataset_df.columns[:37])
        regex = r'C\d{2}'
        coeficientes = [col for col in dataset_df.columns if re.match(regex, col)]
        selected_columns = list( set(selected_columns) - set(coeficientes) )
        #selected_columns.remove()

    #print(selected_columns)
    #print(target_columns)

    # Asegurarse de que las columnas seleccionadas y los targets existen en el DataFrame
    all_columns = selected_columns + target_columns
    for col in all_columns:
        if col not in dataset_df.columns:
            print(f"Error: La columna '{col}' no existe en el DataFrame.")
            return

    # Diccionario para almacenar los MI scores de cada target
    mi_scores_dict = {}

    # Iterar sobre cada target, calcular los MI scores y agregarlos al diccionario
    for target_column in target_columns:
        # Preparar las variables (X) y el target (y) para calcular la información mutua
        X = dataset_df[selected_columns]
        y = dataset_df[target_column]

        if target_column in X.columns:
            X = X.drop(target_column, axis=1)

        # Calcular los scores de información mutua
        mi_scores = make_mi_scores(X, y)
        mi_scores_dict[target_column] = mi_scores

    # Graficar los resultados para múltiples targets en un único gráfico
    plot_mi_scores_multiple_targets(mi_scores_dict, save_path)

if __name__ == "__main__":
    main()
