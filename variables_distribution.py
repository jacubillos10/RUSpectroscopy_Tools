import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_distribution_columns(df, columns, save_path=None):

    num_columns = len(columns)
    columns_per_row = 2
    num_rows = (num_columns + columns_per_row - 1) // columns_per_row  # Redondear hacia arriba
    
    fig, axes = plt.subplots(num_rows, columns_per_row, figsize=(5*columns_per_row, 5*num_rows))
    axes = axes.flatten()  # Aplanar la matriz de ejes
    
    for i, col in enumerate(columns):
        if col in df.columns:
            axes[i].hist(df[col], bins=20, color='blue', alpha=0.7)
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        else:
            print(f"Warning: La columna '{col}' no existe en el DataFrame.")
    
    # Si hay más subplots de los necesarios, ocultar los sobrantes
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()

    # Si se especifica una ruta para guardar el gráfico
    if save_path:
        # Asegurarse de que el directorio existe
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Directorio creado: {save_dir}")
        plt.savefig(save_path)
        print(f"Gráfico guardado en: {save_path}")
    else:
        plt.show()

def main():
    # Configurar argparse para aceptar el CSV, columnas y la opción de guardar el gráfico
    parser = argparse.ArgumentParser(description="Generar histogramas de columnas del dataset.")
    parser.add_argument('--csv', type=str, help="Ruta del archivo CSV.", default='data/lf_Isotropic.csv')
    parser.add_argument('--variables', nargs='+', required=True, help="Columnas del dataset a graficar")
    parser.add_argument('--save', type=str, help="Ruta donde guardar el gráfico generado.", default='/tmp/variables_distribution.png')

    args = parser.parse_args()
    csv_file = args.csv
    selected_columns = args.variables
    save_path = args.save
    
    # Cargar dataset desde el archivo CSV proporcionado por el usuario
    try:
        dataset_df = pd.read_csv(csv_file)
        print(f"Archivo CSV cargado exitosamente desde: {csv_file}")
    except FileNotFoundError:
        print(f"Error: El archivo '{csv_file}' no se encontró.")
        return

    # Graficar las columnas seleccionadas y guardar si se proporciona una ruta
    plot_distribution_columns(dataset_df, selected_columns, save_path)

if __name__ == "__main__":
    main()
