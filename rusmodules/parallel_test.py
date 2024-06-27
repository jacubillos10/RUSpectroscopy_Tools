import numpy as np
from numba import njit, types
from numba.pycc import CC
from numba import prange

cc = CC("test")

@njit("i8[:,:](i8,i8)", parallel = True)
def funcion_prueba(a, b):
    resp = np.zeros((a,b), dtype = np.int64)
    for i in prange(a):
        for j in prange(b):
            resp[i,j] = i*j
        #fin for 
    #fin for 
    return resp
#fin función

@cc.export("funcion_prueba", "i8[:,:](i8,i8)")
def function_prueba_total(a,b):
    return funcion_prueba(a,b)

if __name__ == "__main__":
    cc.compile() 
