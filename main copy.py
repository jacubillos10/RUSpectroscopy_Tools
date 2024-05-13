import RUS_classes as rus
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
from random import uniform, random, randint 
from tqdm import tqdm


if __name__ == "__main__":
    with open('data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for c in tqdm(range(100)):
            names = ['Ortho','Tetra','Cubic','Iso']
            system = names[randint(0,3)]
            shape = 'cylinder'
            limits = [uniform(0.1,1),uniform(0.1,1),uniform(0.1,1)]
            rho = uniform(2.0,10.0)
            centinela = True
            while centinela:
                c11 = uniform(0.3,5.6)
                c22 = uniform(0.3,5.6)
                c33 = uniform(0.3,5.6)
                c44 = uniform(0.3,5.6)
                c55 = uniform(0.3,5.6)
                c66 = uniform(0.3,5.6)
                c12 = uniform(0.3,5.6)
                c13 = uniform(0.3,5.6)
                c23 = uniform(0.3,5.6)
                if min(c11,c33,c22) > 2*max(c12,c13,c44,c66,c55,c23):
                    systems = {'Ortho': np.array([np.array([c11, c12, c13, 0, 0, 0]),
                                np.array([c12, c22, c23, 0, 0, 0,]),
                                np.array([c13, c23, c33, 0, 0, 0]),
                                np.array([0, 0, 0, c44, 0, 0]),
                                np.array([0, 0, 0, 0, c55, 0]),
                                np.array([0, 0, 0, 0, 0, c66])]), 
                                'Tetra': np.array([np.array([c11, c12, c13, 0, 0, 0]),
                                np.array([c12, c11, c13, 0, 0, 0,]),
                                np.array([c13, c13, c33, 0, 0, 0]),
                                np.array([0, 0, 0, c44, 0, 0]),
                                np.array([0, 0, 0, 0, c44, 0]),
                                np.array([0, 0, 0, 0, 0, c66])]), 
                                'Cubic' : np.array([np.array([c11, c12, c12, 0, 0, 0]),
                                np.array([c12, c11, c12, 0, 0, 0,]),
                                np.array([c12, c12, c11, 0, 0, 0]),
                                np.array([0, 0, 0, c44, 0, 0]),
                                np.array([0, 0, 0, 0, c44, 0]),
                                np.array([0, 0, 0, 0, 0, c44])]), 
                                'Iso': np.array([np.array([c11, c11-2*c44, c11-2*c44, 0, 0, 0]),
                                np.array([c11-2*c44, c11, c11-2*c44, 0, 0, 0,]),
                                np.array([c11-2*c44, c11-2*c44, c11, 0, 0, 0]),
                                np.array([0, 0, 0, c44, 0, 0]),
                                np.array([0, 0, 0, 0, c44, 0]),
                                np.array([0, 0, 0, 0, 0, c44])])}
                    C = systems[system]
                    centinela = False
            Forward = rus.Forward(10,'poly',shape,system,limits,rho,C)
            w2 = Forward.W2
            fs = [((w/rho)**(1/2))/(2*np.pi) for w in w2]
            data = [ c+1, 10, 'poly', shape, system, limits, rho, c11, c22, c33, c44, c55, c66, c12, c13, c23, str(fs)]
            writer.writerow(data)


        