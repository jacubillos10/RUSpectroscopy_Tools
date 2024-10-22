from . import rus
import numpy as np
import scipy

def get_eigenvalues(Ng, C, eta, beta, shape):
    """
    Get the normalized eigenvalues given the shape of the sample and the
    elastic constants.

    Arguments:
    Ng -- <int> Maximum degree of the basis function in the forward problem.
    C -- <np.array> 6x6 elastic constant matrix. Units are arbitrary. The first
         lambda will have the same units as the constants.
    eta -- <float> First relation in the dimensions of the sample:
             cos(2*eta) = lz/(lx^2 + ly^2 + lz^2).
    beta -- <float> Second relation in the dimensions of the sample:
            cos(beta) = lx/(lx^2 + ly^2).
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
            lambda_i = (m (omega_i)^2) / r) where r is: r = (lx^2 + ly^2 + 
            lz^2)^(1/2).
    """
    alphas = {"Parallelepiped": 1, "Cylinder": np.pi/4, "Ellipsoid": np.pi/6}
    shapes = {"Parallelepiped": 0, "Cylinder": 1, "Ellipsoid": 2}
    alpha = alphas[shape]
    rc = np.sin(0.5*eta)
    a = rc*np.cos(0.25*beta)
    b = rc*np.sin(0.25*beta)
    c = np.cos(0.5*eta)
    geometry = np.array([a,b,c])
    Gamma = rus.gamma_matrix(Ng, C, geometry, shapes[shape])
    E = rus.E_matrix(Ng, shapes[shape])
    vals = scipy.linalg.eigvalsh(Gamma, b = E)
    #Aquí podemos estudiar la factibilidad de los datos dados
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
    keys and the constants itself in the values. 

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
#fin funcióN

def get_eigenvalues_from_crystal_structure(Ng, const_relations, eta, beta, shape, mult = 1):
    """
    Get the normalized eigenvalues given 
    crystal estructure and the relation between independen constants.
    This function normalizes the given constants by default 

    Arguments: 
    Ng -- <int> Maximum degree of the basis function in the forward problem.
    const_relations -- <dict> A dictionary containing the values of the 
             independent contants according to a crystal structure. All keys
             MUST start with "x_". For example, in a isotropic material a 
             dictionary like this must be the argument: {"x_K": 3/5, "x_mu": 4/5}
    eta -- <float> First relation in the dimensions of the sample:
             cos(2*eta) = lz/(lx^2 + ly^2 + lz^2).
    beta -- <float> Second relation in the dimensions of the sample:
            cos(4*beta) = lx/(lx^2 + ly^2).
    shape -- <string> Shape of the sample. Currently only supports one of these
             values: "Parallelepiped", "Cylinder", and "Ellipsoid".
    mutl -- <float> Multiplier of the elastic constants 
    
    Returns:
    A dictionary containing the following:
    key: "eig" -- <np.array> Normalized eigenvalues of the rus forward problem. 
            Thefirst element `vals[0]` is the first eigenvalue (lambda_0) and 
            has the same units and order of magnitude as the given elastic
            constants. The rest of the elements (vals[1:]) are the relation
            between the i-th eigenvalue and the first eigenvalue (lambda_i /
            lambda_0). Each eigenvalue is:
            lambda_i = (m (omega_i)^2) / r) where r is: r = (lx^2 + ly^2 + 
            lz^2)^(1/2).
    """
    if not all((x[:2] == "x_" for x in const_relations.keys())):
        raise KeyError("All the keys within const_relations MUST start with 'x_'")
    #fin if
    #C_vals = np.array(tuple(const_relations.values()))
    constants = dict(map(lambda x: (x[2:], mult*const_relations[x]), const_relations.keys()))
    C = get_elastic_constants(constants)
    relative_eig = get_eigenvalues(Ng, C, eta, beta, shape)
    return relative_eig
#fin función

if __name__ == "__main__":
    C_test = get_elastic_constants({"K": 3, "mu":1})
    print(C_test)

