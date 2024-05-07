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
        for c in tqdm(range(50)):
            shape = 'rectangular_p'
            limits = [random(),random(),random()]
            rho = uniform(2.0,10.0)
            centinela = True
            while centinela:
                c11 = uniform(0.3,5.6)
                c12 = uniform(0.3,5.6)
                c13 = uniform(0.3,5.6)
                c33 = uniform(0.3,5.6)
                c44 = uniform(0.3,5.6)
                c66 = uniform(0.3,5.6)
                if c11 > c12 and c11 > c13 and c11>c44 and c11>c66:
                    C = np.array([np.array([c11, c12, c13, 0, 0, 0]),
                                np.array([c12, c11, c13, 0, 0, 0,]),
                                np.array([c13, c13, c33, 0, 0, 0]),
                                np.array([0, 0, 0, c44, 0, 0]),
                                np.array([0, 0, 0, 0, c44, 0]),
                                np.array([0, 0, 0, 0, 0, c66])])
                    centinela = False
            Forward = rus.Forward(10,'poly',shape,'Tetra',limits,rho,C)
            w2 = Forward.W2
            amps = Forward.Amps
            fs = [((w/rho)**(1/2))/(2*np.pi) for w in w2]
            data = [ ' ', 10, 'poly', shape, 'Tetra', limits, rho, c11, c12, c13, c33, c44, c66, str(fs)]
            writer.writerow(data)
        


        