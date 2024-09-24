import numpy as np 
import pandas as pd

data_parallelepiped_raw = pd.read_csv("input_data/Combinatoriales_214825_Eigen_Parallelepiped.csv", delimiter = ",", index_col = 0)
data_cylinder_raw = pd.read_csv("input_data/Combinatoriales_1962981_Eigen_Cylinder.csv", delimiter = ",", index_col = 0)
data_ellipsoid_raw = pd.read_csv("input_data/Combinatoriales_1101790_Eigen_Ellipsoid.csv", delimiter = ",", index_col = 0)

parallelepipeds = pd.DataFrame({"Shape": ["Parallelepipied"] * len(data_parallelepiped_raw)})
cylinders = pd.DataFrame({"Shape": ["Cylinder"] * len(data_cylinder_raw)})
ellipsoids = pd.DataFrame({"Shape": ["Ellipsoid"] * len(data_ellipsoid_raw)})

data_parallelepiped = pd.concat((data_parallelepiped_raw, parallelepipeds), axis = 1)
data_cylinder = pd.concat((data_cylinder_raw, cylinders), axis = 1)
data_ellipsoid = pd.concat((data_ellipsoid_raw, ellipsoids), axis = 1)

data_total = pd.concat((data_parallelepiped, data_cylinder, data_ellipsoid), axis = 0, ignore_index = True)
cols_iso = pd.DataFrame({"Cry_st": ["Isotropic"] * len(data_total)})
data_total = pd.concat((data_total, cols_iso), axis = 1)

print(len(data_total))
data_total.to_csv("input_data/Combinatoriales_3_formas.csv")
