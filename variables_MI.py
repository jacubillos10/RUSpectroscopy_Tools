import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.feature_selection import mutual_info_regression

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

def plot_mi_scores(scores, target, save_path=None):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title(f"Mutual Information Scores with Target: {target}")
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
    # Configurar argparse para aceptar el CSV, target y variables
    parser = argparse.ArgumentParser(description="Mostrar la información mutua entre un target y varias variables.")
    parser.add_argument('--csv', type=str, help="Ruta del archivo CSV.", default='data/lf_Isotropic.csv')
    parser.add_argument('--variables', nargs='+', required=True, help="Columnas del dataset a comparar con el target")
    parser.add_argument('--target', type=str, required=True, help="Columna objetivo para calcular la información mutua")
    parser.add_argument('--save', type=str, help="Ruta donde guardar el gráfico generado.", default='/tmp/variables_MI.png')

    args = parser.parse_args()
    csv_file = args.csv
    selected_columns = args.variables
    target_column = args.target
    save_path = args.save
    
    # Cargar dataset desde el archivo CSV proporcionado por el usuario
    try:
        dataset_df = pd.read_csv(csv_file)
        print(f"Archivo CSV cargado exitosamente desde: {csv_file}")
    except FileNotFoundError:
        print(f"Error: El archivo '{csv_file}' no se encontró.")
        return

    # Asegurarse de que las columnas seleccionadas y el target existen en el DataFrame
    all_columns = selected_columns + [target_column]
    for col in all_columns:
        if col not in dataset_df.columns:
            print(f"Error: La columna '{col}' no existe en el DataFrame.")
            return

    # Preparar las variables (X) y el target (y) para calcular la información mutua
    X = dataset_df[selected_columns]
    y = dataset_df[target_column]

    # Calcular los scores de información mutua
    mi_scores = make_mi_scores(X, y)

    # Graficar los resultados
    plot_mi_scores(mi_scores, target_column, save_path)


if __name__ == "__main__":
    main()
