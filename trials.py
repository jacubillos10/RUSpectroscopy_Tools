import numpy as np
def generar_tuplas(N):
    nums = np.arange(0, N+1)
    # Generar todas las posibles combinaciones de índices usando broadcasting
    i, j, k = np.broadcast_arrays(nums[:, None, None], nums[None, :, None], nums[None, None, :])
    # Filtrar combinaciones donde la suma sea menor o igual a N
    indices_filtrados = np.where(i + j + k <= N)
    # Obtener las tuplas correspondientes a los índices filtrados
    tuplas = np.vstack([i[indices_filtrados], j[indices_filtrados], k[indices_filtrados]]).T
    return tuplas

# Ejemplo de uso
N = int(input("Ingrese un entero N: "))
tuplas = generar_tuplas(N)
print("Tuplas cuya suma es menor o igual a", N, ":")
print(tuplas)