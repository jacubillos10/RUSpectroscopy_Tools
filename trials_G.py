import numpy as np
from tqdm import tqdm
def construct_full_phi(phi):
    N = len(phi)
    n = len(phi[0])
    zero_aux = np.zeros((N,n))
    elem1 = np.row_stack((phi,zero_aux,zero_aux))
    elem2 = np.row_stack((zero_aux,phi,zero_aux))
    elem3 = np.row_stack((zero_aux,zero_aux,phi))
    PHI = np.array([elem1,elem2,elem3])
    return PHI

def construct_elem_G(elem1, elem2):
    for i in range(3):
        for j in range(3):
            if i == j:
                if (elem1[i] + elem2[j] - 2) // 2 == 0:
                    
                else: 
                    return np.zeros(3)    



def derivation_PHI(PHI,transposed):
    prod=None    
            
    if transposed:
        prodt = np.transpose(prod,(1,0,2))
        return prodt
    else:
        return prod

def generar_tuplas(N):
    nums = np.arange(0, N+1)
    i, j, k = np.broadcast_arrays(nums[:, None, None], nums[None, :, None], nums[None, None, :])
    indices_filtrados = np.where(i + j + k <= N)
    phi = np.vstack([i[indices_filtrados], j[indices_filtrados], k[indices_filtrados]]).T 
    return phi

def calc_G(C,limits,N):
    np.set_printoptions(linewidth=700)
    phi = generar_tuplas(N)
    PHI = construct_full_phi(phi)
    prod = derivation_PHI(PHI,False)
    prod_t = derivation_PHI(PHI,True)
    len_PHI = len(PHI)

    N1 = len(prod_t)
    n1 = len(prod_t[0])
    
    half_res = np.zeros((N1,n1,4))

    #PHIt*Bt*C
    for i in tqdm(range(N1)):
        for j in range(6):
            for k in range(n1):   
                half_res[i,j,3] += prod_t[i,k,3] * C[k,j]

    n2 = len(prod[0])
    G_noint = np.zeros((N1,n2,8))

    #PHIt*Bt*C*B*PHI
    for i in tqdm(range(N1)):
        row = np.array([])
        for j in range(n1):
            for k in range(n2):

                phi_elem_1 = half_res[i,j]
                phi_elem_2 = prod[j,k]
                if phi_elem_1[3] == 0:
                    phi_elem_1 = np.zeros(4)
                if phi_elem_2[3] == 0:
                    phi_elem_2 = np.zeros(4)
                whole = np.concatenate((phi_elem_1,phi_elem_2))
                G_noint[i,k] += whole

    N = len(G_noint)
    n = len(G_noint[0])
    G = np.zeros((N,n))


    #Integration
    for i in tqdm(range(N)):
        for j in range(n):
            exps = G_noint[i,j]
            cts = exps[3]*exps[7]
            p = exps[0]+exps[4]
            q = exps[1]+exps[5]
            r = exps[2]+exps[6]

            result = (8*cts*(limits[0]**(p+1))*(limits[1]**(q+1))*(limits[2]**(r+1)))/((p+1)*(q+1)*(r+1))
            
            G[i,j] = result
    
    return G
    
   

N = int(input("Ingrese N: "))
l = list(map(float,input("Ingrese las medidas del paralelepipedo separadas por comas: ").strip().split(",")))
limits = np.array(l)
c = list(map(float,input("Ingrese las 21 constantes en orden: ").strip().split(",")))
C = np.array([[c[0],c[1],c[2],c[3],c[4],c[5]],
              [c[1],c[6],c[7],c[8],c[9],c[10]],
              [c[2],c[7],c[11],c[12],c[13],c[14]],
              [c[3],c[8],c[12],c[15],c[16],c[17]],
              [c[4],c[9],c[13],c[16],c[18],c[19]],
              [c[5],c[10],c[14],c[17],c[19],c[20]]])

G = calc_G(C,limits,N)
print(G)

