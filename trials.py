import numpy as np
import scipy.integrate as it 
from tqdm import tqdm 
C = np.zeros((6,6))
for r in range(6):
    row = [1,2,3,4,5,6]
    for l in range(len(row)):
        C[r,l] = row[l]
print(C)