import numpy as np
import rusmodules
from rusmodules import rus
from rusmodules import data_generator

C_ranks = (0.3, 5.6)
dim_min = (0.01, 0.01, 0.01)
dim_max = (0.5, 0.5, 0.5)

dims = np.random.uniform(dim_min, dim_max)
C = data_generator(C_ranks[0], C_ranks[1], 0)

