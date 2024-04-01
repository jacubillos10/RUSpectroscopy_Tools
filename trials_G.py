import numpy as np
from tqdm import tqdm
def it_c(i,j):
    """
    This function changes the double index convention for single index convention in the elastic constant matrix C.
    00 is changed to 0, 11 is changed to 1, 22 is changed to 2, 
    01 or 10 is changed to 5
    02 or 20 is changed to 4
    12 or 21 is changed to 3

    @Input i <int>: first index to be changed
    @Input j <int>: second index to be changed 
    @Return index <int>: New index 
    """
    if i == j:
        index = i
    else:
        index = 6 - (i + j)
        
    return index

def construct_full_phi(phi):
    N = len(phi)
    n = len(phi[0])
    zero_aux = np.zeros((N,n))
    elem1 = np.row_stack((phi,zero_aux,zero_aux))
    elem2 = np.row_stack((zero_aux,phi,zero_aux))
    elem3 = np.row_stack((zero_aux,zero_aux,phi))
    PHI = np.array([elem1,elem2,elem3])
    return PHI

def construct_elem_G(elem1, elem2, i1, i2, limits, C):
    element = 0
    for j1 in range(3):
        for j2 in range(3):
            exp1 = elem1[j1]
            exp2 = elem2[j2]
            if j1 == j2:
                full_exp = exp1 + exp2 - 2
                if full_exp // 2 == 0:
                    if j1 == 0 and ((elem1[1]+elem2[1]) // 2 == 0) and ((elem1[2]+elem2[2]) // 2 == 0):
                        element += (C[it_c(i1,j1),it_c(i2,j2)]*(8*exp1*exp2*(limits[0]**(full_exp+1))*(limits[1]**(elem1[1]+elem2[1]+1))*(limits[2]**(elem1[2]+elem2[2]+1)))/((full_exp+1)*(elem1[1]+elem2[1]+1)*(elem1[2]+elem2[2]+1)))
                    elif j1 == 1 and ((elem1[0]+elem2[0]) // 2 == 0) and ((elem1[2]+elem2[2]) // 2 == 0):
                        element += (C[it_c(i1,j1),it_c(i2,j2)]*(8*exp1*exp2*(limits[1]**(full_exp+1))*(limits[0]**(elem1[0]+elem2[0]+1))*(limits[2]**(elem1[2]+elem2[2]+1)))/((full_exp+1)*(elem1[0]+elem2[0]+1)*(elem1[2]+elem2[2]+1)))
                    elif j1 == 2  and ((elem1[1]+elem2[1]) // 2 == 0) and ((elem1[0]+elem2[0]) // 2 == 0): 
                        element += (C[it_c(i1,j1),it_c(i2,j2)]*(8*exp1*exp2*(limits[2]**(full_exp+1))*(limits[1]**(elem1[1]+elem2[1]+1))*(limits[0]**(elem1[0]+elem2[0]+1)))/((full_exp+1)*(elem1[1]+elem2[1]+1)*(elem1[0]+elem2[0]+1)))
                    else:
                        element += 0
                else: 
                    element += 0
            else:
                if j1 == 0: 
                    if j2 == 1:
                        if (exp1+elem2[0]-1) // 2 == 0 and ((elem1[1]+exp2-1) // 2 == 0) and ((elem1[2]+elem2[2]) // 2 == 0):
                            element += (C[it_c(i1,j1),it_c(i2,j2)]*(8*exp1*exp2*(limits[0]**(exp1+elem2[0]))*(limits[1]**(elem1[1]+exp2))*(limits[2]**(elem1[2]+elem2[2]+1)))/((exp1+elem2[0])*(elem1[1]+exp2)*(elem1[2]+elem2[2]+1)))
                        else: 
                            element += 0
                    else: 
                        if (exp1+elem2[0]-1) // 2 == 0 and ((elem1[2]+exp2-1) // 2 == 0) and ((elem1[1]+elem2[1]) // 2 == 0):
                            element += (C[it_c(i1,j1),it_c(i2,j2)]*(8*exp1*exp2*(limits[0]**(exp1+elem2[0]))*(limits[2]**(elem1[2]+exp2))*(limits[1]**(elem1[1]+elem2[1]+1)))/((exp1+elem2[0])*(elem1[2]+exp2)*(elem1[1]+elem2[1]+1)))
                        else: 
                            element += 0
                elif j1 == 1:
                    if j2 == 0:
                        if (exp1+elem2[1]-1) // 2 == 0 and ((elem1[0]+exp2-1) // 2 == 0) and ((elem1[2]+elem2[2]) // 2 == 0):
                            element += (C[it_c(i1,j1),it_c(i2,j2)]*(8*exp1*exp2*(limits[1]**(exp1+elem2[1]))*(limits[0]**(elem1[0]+exp2))*(limits[2]**(elem1[2]+elem2[2]+1)))/((exp1+elem2[1])*(elem1[0]+exp2)*(elem1[2]+elem2[2]+1)))
                        else: 
                            element += 0
                    else: 
                        if (exp1+elem2[1]-1) // 2 == 0 and ((elem1[2]+exp2-1) // 2 == 0) and ((elem1[0]+elem2[0]) // 2 == 0):
                            element += (C[it_c(i1,j1),it_c(i2,j2)]*(8*exp1*exp2*(limits[1]**(exp1+elem2[1]))*(limits[2]**(elem1[2]+exp2))*(limits[0]**(elem1[0]+elem2[0]+1)))/((exp1+elem2[1])*(elem1[2]+exp2)*(elem1[0]+elem2[0]+1)))
                        else: 
                            element += 0 
                else:
                    if j2 == 0:
                        if (exp1+elem2[2]-1) // 2 == 0 and ((elem1[0]+exp2-1) // 2 == 0 )and ((elem1[1]+elem2[1]) // 2 == 0):
                            element += (C[it_c(i1,j1),it_c(i2,j2)]*(8*exp1*exp2*(limits[2]**(exp1+elem2[2]))*(limits[0]**(elem1[0]+exp2))*(limits[1]**(elem1[1]+elem2[1]+1)))/((exp1+elem2[2])*(elem1[0]+exp2)*(elem1[1]+elem2[1]+1)))
                        else: 
                            element += 0
                    else: 
                        if (exp1+elem2[2]-1) // 2 == 0 and ((elem1[1]+exp2-1) // 2 == 0) and ((elem1[0]+elem2[0]) // 2 == 0):
                            element += (C[it_c(i1,j1),it_c(i2,j2)]*(8*exp1*exp2*(limits[2]**(exp1+elem2[2]))*(limits[1]**(elem1[1]+exp2))*(limits[0]**(elem1[0]+elem2[0]+1)))/((exp1+elem2[2])*(elem1[1]+exp2)*(elem1[0]+elem2[0]+1)))
                        else: 
                            element += 0 
    return element



def generar_tuplas(N):
    nums = np.arange(0, N+1)
    i, j, k = np.broadcast_arrays(nums[:, None, None], nums[None, :, None], nums[None, None, :])
    indices_filtrados = np.where(i + j + k <= N)
    phi = np.vstack([i[indices_filtrados], j[indices_filtrados], k[indices_filtrados]]).T 
    return phi

def calc_G(C,limits,N):
    np.set_printoptions(linewidth=700)
    phi = generar_tuplas(N)
    R = len(phi)  
    G = np.zeros((3*R,3*R)) 
    for i1 in range(3):
        for i2 in range(3):
            for p in range(R): 
                for q in range(R):
                    elem1 = phi[p]
                    elem2 = phi[q]
                    G[i1*R+p,i2*R+q] = construct_elem_G(elem1,elem2,i1,i2,limits,C)
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

"""
def calc_G_alt(self):
        

        #Relevant parameter
        np.set_printoptions(linewidth=700)
        base = self.base
        sample = self.sample
        C = sample.C
        limits = sample.limits
        phi = base.phi
        PHI = self.construct_full_phi(phi)
        len_PHI = len(PHI)

        #B*PHI
        prod = self.derivation_PHI(PHI,False)

        #PHIt*Bt
        prod_t = self.derivation_PHI(PHI,True)

        N1 = len(prod_t)
        n1 = len(prod_t[0])
        n2 = len(prod[0])

        half_res = np.zeros((N1,n1,4))

        #PHIt*Bt*C
        for i in tqdm(range(N1)):
            for j in range(6):
                for k in range(n1):   
                    half_res[i,j,3] += prod_t[i,k,3] * C[k,j]

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

        #Integration (Only valid for rectangular parallelepipeids)
        for i in range(N):
            for j in range(n):

                exps = G_noint[i,j]
                cts = exps[3]*exps[7]
                p = exps[0]+exps[4]
                q = exps[1]+exps[5]
                r = exps[2]+exps[6]

                result = (8*cts*(limits[0]**(p+1))*(limits[1]**(q+1))*(limits[2]**(r+1)))/((p+1)*(q+1)*(r+1))
                
                G[i,j] = result
        
        return G

def derivation_PHI(self,Phi,transposed):

        #Relevant paramenters
        N = len(Phi)
        n = len(Phi[0])

        #B matrix (derivation)
        B = np.zeros((6,3,2))
        B[0,0,0] = 1
        B[1,1,0] = 1
        B[2,2,0] = 1
        B[3,1,0] = 1/2
        B[3,2,0] = 1/2
        B[4,0,0] = 1/2
        B[4,2,0] = 1/2
        B[5,0,0] = 1/2
        B[5,1,0] = 1/2
        B[0,0,1] = 1
        B[1,1,1] = 1
        B[2,2,1] = 1
        B[3,1,1] = 1
        B[3,2,1] = 1
        B[4,0,1] = 1
        B[4,2,1] = 1
        B[5,0,1] = 1
        B[5,1,1] = 1

        #B*PHI 
        prod = np.zeros((6,n,4))

        for k in tqdm(range(6)):
            for i in range(n):
                for j in range(N):
                    for z in range(N):

                        phi_elem = Phi[j,i]
                        exp = phi_elem[j]

                        if k == 0 or k == 1 or k ==2:

                            if Phi[j,i,z] != 0 and z == j:
                                new = phi_elem.copy()
                                new[z] -= B[k,j,1]
                                new = np.append(new,Phi[j,i,z] * B[k,j,0])
                            else:
                                new = np.zeros(4)

                        else:

                            if Phi[j,i,z] != 0 and ((k == 3 and ((j == 1 and z == 2) or ( j == 2 and z == 1))) or (k == 5 and ((j == 1 and z == 0) or ( j == 0 and z == 1))) or (k == 4 and ((j == 2 and z == 0) or ( j == 0 and z == 2)))):
                                new = phi_elem.copy()
                                new[z] -= B[k,j,1]
                                new = np.append(new,Phi[j,i,z] * B[k,j,0])
                            else:
                                new = np.zeros(4)
                        
                        prod[k,i] += new    

        #Transposed or not       
        if transposed:
            prodt = np.transpose(prod,(1,0,2))
            return prodt
        else:
            return prod

"""