import numpy as np
import scipy.integrate as it 
from tqdm import tqdm 

def construct_E_from_int(E_int):
        """
        Matrix E constructor. Constructs the final matrix E from the reduced version of the matrix already integrated.

        @Input:
         E_int <np.array<np.array<int>>>:  A numpy array that holds a reduced version of the information of the volume integral 
         for the product of phi with phi trasposed (phi: the expansion of the base functions represented with indexes).

        @Output:
         E <np.array<np.array<int>>>: A numpy array that holds the information of the volume integral for the 
         product of phi with phi trasposed (phi: the expansion of the base functions represented with indexes). 
        """
        N = len(E_int)
        zero_aux = np.zeros([N])
        E = np.array([])
        for i in tqdm(range(N)):
            row = np.concatenate((E_int[i],zero_aux,zero_aux))
            if i == 0:
                E = np.hstack((E,row))
            else:
                E = np.vstack((E,row))
        for i in tqdm(range(N)):
            row = np.concatenate((zero_aux,E_int[i],zero_aux))
            E = np.vstack((E,row))
        for i in tqdm(range(N)):
            row = np.concatenate((zero_aux,zero_aux,E_int[i]))
            E = np.vstack((E,row))
        return E

def generar_tuplas(N,limits):
    nums = np.arange(0, N+1)
    # Generar todas las posibles combinaciones de índices usando broadcasting
    i, j, k = np.broadcast_arrays(nums[:, None, None], nums[None, :, None], nums[None, None, :])
    # Filtrar combinaciones donde la suma sea menor o igual a N
    indices_filtrados = np.where(i + j + k <= N)
    # Obtener las tuplas correspondientes a los índices filtrados
    phi = np.vstack([i[indices_filtrados], j[indices_filtrados], k[indices_filtrados]]).T 
    n1 = len(phi)
    E = np.array([])
    for i in tqdm(range(n1)):
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
    for i in tqdm(range(n)):
        exps = []
        l=0
        for j in range(n_i):
            exps.append(E[i,j])
        p = exps[0]+exps[3]
        q = exps[1]+exps[4]
        r = exps[2]+exps[5]

        result = (8*(limits[0]**(p+1))*(limits[1]**(q+1))*(limits[2]**(r+1)))/((p+1)*(q+1)*(r+1))
            
        row = np.append(row,result)
        if len(row) == n1 and i == n1- 1:
            E_int = np.hstack((E_int,row))
            row = np.delete(row,[range(n1)])
        elif len(row) == n1:
            E_int = np.vstack((E_int,row))
            row = np.delete(row,[range(n1)])

    E = construct_E_from_int(E_int)
    return E

# Ejemplo de uso
N = int(input("Ingrese un entero N: "))
l = list(map(float,input("Ingrese las medidas del paralelepipedo separadas por comas: ").strip().split(",")))
limits = np.array(l)
tuplas = generar_tuplas(N,limits)
print(tuplas)
"""
f = lambda x,y,z: x**(exps[0]+exps[3]) * y**(exps[1]+exps[4]) * z**(exps[2]+exps[5])
        result = it.tplquad(f,-1,1,-1,1,-1,1,epsabs=1.49e-2,epsrel=1.49e-2)
        row = np.append(row,result[0])
"""
