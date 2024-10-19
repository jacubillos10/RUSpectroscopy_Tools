from . import rus
import numpy as np
import scipy

def get_eigenvalues(Ng, C, gamma, beta, shape):
    """
    Get the normalized eigenvalues given the shape of the sample and the
    elastic constants.

    Arguments:
    Ng -- <int> Maximum degree of the basis function in the forward problem.
    C -- <np.array> 6x6 elastic constant matrix. Units are arbitrary. The first
         lambda will have the same units as the constants.
    gamma -- <float> First relation in the dimensions of the sample:
             gamma = ly / (lx + ly).
    beta -- <float> Second relation in the dimensions of the sample:
            beta = lz / (lx + lz).
    shape -- <string> Shape of the sample. Currently only supports one of these
             values: "Parallelepiped", "Cylinder", and "Ellipsoid".

    Returns:
    A dictionary containing the following:
    key: "eig" -- <np.array> Normalized eigenvalues of the rus forward problem. 
            Thefirst element `vals[0]` is the first eigenvalue (lambda_0) and 
            has the same units and order of magnitude as the given elastic
            constants. The rest of the elements (vals[1:]) are the relation
            between the i-th eigenvalue and the first eigenvalue (lambda_i /
            lambda_0). Each eigenvalue is:
            lambda_i = (m (omega_i)^2) / (volume^(1/3)).
    """
    alphas = {"Parallelepiped": 1, "Cylinder": np.pi/4, "Ellipsoid": np.pi/6}
    alpha = alphas[shape]
    a = (( (1 - gamma) * (1 - beta) )/( alpha * gamma *beta ))**(1/3)
    b = (( (gamma**2) * (1 - beta) )/( alpha * ((1 - gamma)**2) * beta ))**(1/3)
    c = (( (beta**2) * (1 - gamma) )/( alpha * ((1 - beta)**2) * gamma ))**(1/3)
    geometry = np.array([a,b,c])
    #vol = alpha*np.prod(geometry) este siempre debe ser 1. Esta variable nos sirve para depurar
    Gamma = rus.gamma_matrix(Ng, C, geometry, shape)
    E = rus.E_matrix(Ng, shape)
    vals = scipy.linalg.eigvalsh(Gamma, b = E)
    #Aquí podemos estudiar la factibilidad de los daots dados
    vals_fin = vals[6:]
    vals_fin[1:] = vals_fin[1:]/vals_fin[0]
    return {"eig": vals_fin}
#fin función

def get_elastic_constants(independent_constants):
    """
    Get the elastic constants given the crystaline structure and relevant 
    parameters.

    Arguments:
    independent_constants -- <dict> Names of independent constants in the 
    keys their respective values. 

    Returns:
    C -- <np.array> The 6x6 matrix of the elastic constants. 
    """
    C = np.zeros((6,6))
    C_prim = np.array(tuple(map(lambda i: tuple(map(lambda j: 1 if i == j and i<3 else 0, range(6))), range(6)))) #Valores de C00, C11, C22
    C_sec = np.array(tuple(map(lambda i: tuple(map(lambda j: 1 if i == j and i >= 3 else 0, range(6))), range(6)))) #Valores de C33, C44, C55
    C_shear_prim = np.array(tuple(map(lambda i: tuple(map(lambda j: 1 if i != j and i<3 and j<3 else 0, range(6))), range(6)))) #Valores de C01, C02, C12
    C_shear_sec = np.array(tuple(map(lambda i: tuple(map(lambda j: 1 if i != j and i>=3 and j>=3 else 0, range(6))), range(6)))) #Valores de C34, C35, C45
    if len(independent_constants) == 2:
        C_prim = C_prim * (independent_constants["K"] + (4/3)*independent_constants["mu"])
        C_sec = C_sec * independent_constants["mu"]
        C_shear_prim = C_shear_prim * (independent_constants["K"] - (2/3)*independent_constants["mu"])
        C = C_prim + C_sec + C_shear_prim
    #fin if 
    return C
#fin función

