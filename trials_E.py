import numpy as np
import scipy.integrate as it 
from tqdm import tqdm 

def construct_elem_E(elem1,elem2,limits):
    np.set_printoptions(linewidth=700)
    alfa = elem1[0]
    alfap = elem2[0]
    beta = elem1[1]
    betap = elem2[1]
    delta = elem1[2]
    deltap = elem2[2]

    exp1 = alfa+alfap+1
    exp2 = beta+betap+1
    exp3 = delta+deltap+1

    coef = (1 - (-1)**exp1)*(1 - (-1)**exp2)*(1 - (-1)**exp3)
    if  coef != 0 :
        element = (coef*(limits[0]**(exp1))*(limits[1]**(exp2))*(limits[2]**(exp3)))/((exp1)*(exp2)*(exp3))
    else: 
        element = 0

    return element

def construct_E_from_red(E_int):
        
        N = len(E_int)
        zero_aux = np.zeros([N])
        E = np.array([])

        for i in range(N):

            row = np.concatenate((E_int[i],zero_aux,zero_aux))

            if i == 0:
                E = np.hstack((E,row))
            else:
                E = np.vstack((E,row))
        
        for i in range(N):

            row = np.concatenate((zero_aux,E_int[i],zero_aux))
            E = np.vstack((E,row))
        
        for i in range(N):

            row = np.concatenate((zero_aux,zero_aux,E_int[i]))
            E = np.vstack((E,row))

        return E

def generar_tuplas(N,limits):
    np.set_printoptions(precision=2)
    nums = np.arange(0, N+1)
    # Generar todas las posibles combinaciones de índices usando broadcasting
    i, j, k = np.broadcast_arrays(nums[:, None, None], nums[None, :, None], nums[None, None, :])
    # Filtrar combinaciones donde la suma sea menor o igual a N
    indices_filtrados = np.where(i + j + k <= N)
    # Obtener las tuplas correspondientes a los índices filtrados
    phi = np.vstack([i[indices_filtrados], j[indices_filtrados], k[indices_filtrados]]).T 
    R = len(phi)
    E_red = np.zeros((R,R))
    for i1 in range(3):
        for i2 in range(3):
            for p in range(R):
                for q in range(R):
                    elem1 = phi[p]
                    elem2 = phi[q]
                    E_red[p,q] = construct_elem_E(elem1,elem2,limits)

    W = np.linalg.eigvalsh(E_red)
    print(np.all(W>=0))
    print(np.linalg.norm(E_red-E_red.T))
    print(W)
    E = construct_E_from_red(E_red)
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
