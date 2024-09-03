import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def graficar_histogramas(data, targets):
    fig = plt.figure(figsize = (20,20))
    N_datos = len(data)
    for i in range(len(targets)):
        num = int(str(33) + str(i+1))
        target = targets[i]
        ax = fig.add_subplot(num)
        ax.hist(data[target], bins = 100)
        ax.set_title(target, fontsize=24)
        ax.tick_params(axis='x', labelsize=21)  # Larger x-labels
        ax.tick_params(axis='y', labelsize=14)  # Slightly larger y-labels
    #fin for 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig("Distribución_CXX" "_" + str(N_datos) + ".png")
    plt.close()
#fin función de graficar
