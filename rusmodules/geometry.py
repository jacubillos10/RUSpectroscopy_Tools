import numpy as np
import matplotlib.pyplot as plt

def generate_sphere_surface_points(N, max_theta, max_phi, options = {"theta": False, "phi": False}):
    Omega = (max_phi * (1 - np.cos(max_theta))) / N
    d_med = Omega**(1/2)
    N_latitudes = int(max_theta/d_med)
    delta_theta = max_theta/N_latitudes
    delta_phi = Omega/delta_theta
    max_theta_space = max_theta if options["theta"] else max_theta*(1 - 1/N_latitudes)
    rango_theta = np.linspace(max_theta/N_latitudes, max_theta_space, N_latitudes)
    resp = []
    for i, theta in enumerate(rango_theta):
        N_longitudes = int(max_phi*np.sin(theta)/delta_phi)
        max_phi_space = max_phi if options["phi"] else max_phi*(1 - 1/N_longitudes)
        rango_phi = np.linspace(max_phi/N_longitudes, max_phi_space, N_longitudes)
        for j, phi in enumerate(rango_phi):
            resp.append([theta, phi])
        #fin for 
    #fin for
    return np.array(resp)
#fin función

if __name__ == "__main__":
    combi = generate_sphere_surface_points(100, 0.5*np.pi, np.pi, options = {"theta": True, "phi": True})
    print(combi)
    print("Número de puntos")
    print(len(combi))
    x = np.sin(combi[:,0])*np.cos(combi[:,1])
    y = np.sin(combi[:,0])*np.sin(combi[:,1])
    z = np.cos(combi[:,0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    ax.scatter(x,y,z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
