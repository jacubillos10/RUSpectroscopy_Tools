import numpy as np
import scipy.integrate as it 

def generar_tuplas(N):
    nums = np.arange(0, N+1)
    # Generar todas las posibles combinaciones de índices usando broadcasting
    i, j, k = np.broadcast_arrays(nums[:, None, None], nums[None, :, None], nums[None, None, :])
    # Filtrar combinaciones donde la suma sea menor o igual a N
    indices_filtrados = np.where(i + j + k <= N)
    # Obtener las tuplas correspondientes a los índices filtrados
    phi = np.vstack([i[indices_filtrados], j[indices_filtrados], k[indices_filtrados]]).T 
    n1 = len(phi)
    E = np.array([])
    for i in range(n1):
        for j in range(n1):
            whole = np.concatenate((phi[i],phi[j]))
            if i == 0 and j == 0:
                E = np.hstack((E,np.array(whole)))
            else:
                E = np.vstack((E,np.array(whole)))
    n = len(E)
    n_i = len(E[0])
    E_int = np.array([])
    row = np.array([])
    for i in range(n):
        exps = []
        l=0
        for j in range(n_i):
            exps.append(E[i,j])
        f = lambda x,y,z: x**(exps[0]+exps[3]) * y**(exps[1]+exps[4]) * z**(exps[2]+exps[5])
        result = it.tplquad(f,-1,1,-1,1,-1,1)
        row = np.append(row,result[0])
        if len(row) == n1 and i == n1- 1:
            E_int = np.hstack((E_int,row))
            row = np.delete(row,[range(n1)])
        elif len(row) == n1:
            E_int = np.vstack((E_int,row))
            row = np.delete(row,[range(n1)])
    print(len(E_int[0]))
    return E_int

# Ejemplo de uso
N = int(input("Ingrese un entero N: "))
tuplas = generar_tuplas(N)
print("Tuplas cuya suma es menor o igual a", N, ":")
print(tuplas)

