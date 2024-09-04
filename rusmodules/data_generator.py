import numpy as np

def generate_C_matrix(C_min, C_max, crystal_structure = 0, distribution = 0):
    C = np.zeros((6,6))
    C_prc = np.zeros(3)
    C_sec = np.zeros(6)
    C_ter = np.zeros(12)
    d_struc = np.array([[3,6],[2,4],[1,2],[1,1]])
    n_p = d_struc[crystal_structure, 0]
    n_s = d_struc[crystal_structure, 1]
    if distribution == 1:
        C_prc_raw = np.random.normal(C_min, C_max, n_p)
        C_sec_raw = np.random.normal(0.5*C_min, 0.5*C_max, n_s)
    else:
        C_prc_raw = np.random.uniform(C_min, C_max, n_p)
        C_sec_raw = np.random.uniform(0, C_max, n_s)
    if crystal_structure == 1:
        C_prc[0:2] = C_prc_raw[0]
        C_prc[2] = C_prc_raw[1]
        C_sec[0] = C_sec_raw[0]
        C_sec[1:3] = C_sec_raw[1]
        C_sec[3:5] = C_sec_raw[2]
        C_sec[5] = C_sec_raw[3]
    elif crystal_structure == 2:
        C_prc[:] = C_prc_raw[0]
        C_sec[0:3] = C_sec_raw[0]
        C_sec[3:] = C_sec_raw[1]
    elif crystal_structure == 3:
        C_prc[:] = C_prc_raw[0]
        C_sec[0:3] = C_prc_raw[0] - 2*C_sec_raw[0]
        C_sec[3:] = C_sec_raw[0]
    else: 
        C_prc = C_prc_raw
        C_sec = C_sec_raw
    #end if 
    for i in range(3):
        for j in range(3):
            C[i,j] = C_prc[i] if i == j else C_sec[i+j-1]
            C[i+3,j+3] = C_sec[i+3] if i == j else 0
        #fin for 
    #fin if
    
    return C
#end function

