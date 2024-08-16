import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

MI_nuevas = pd.read_csv("mutual_info_a.csv", index_col=0)
MI_antiguas = pd.read_csv("mutual_info_l.csv", index_col=0)
targets = MI_nuevas.keys()
features_nuevas = MI_nuevas.index
features_antiguas = MI_antiguas.index
N_datos = 30000

fig1 = plt.figure(figsize=(20,20))
for i in range(len(targets)):
    num = int(str(33) + str(i+1))
    target = targets[i]
    ax = fig1.add_subplot(num)
    ax.barh(features_antiguas, MI_antiguas[target])
    ax.set_title(target, fontsize=24)
    ax.tick_params(axis='x', labelsize=21)  # Larger x-labels
    ax.tick_params(axis='y', labelsize=14)  # Slightly larger y-labels
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig("Omega_" + str(N_datos) + "datos.png")

# %%
fig2 = plt.figure(figsize = (20,20))
for i in range(len(targets)):
    num = int(str(33) + str(i+1))
    target = targets[i]
    ax = fig2.add_subplot(num)
    ax.barh(features_nuevas, MI_nuevas[target])
    ax.set_title(target, fontsize=24)
    ax.tick_params(axis='x', labelsize=21)  # Larger x-labels
    ax.tick_params(axis='y', labelsize=14)  # Slightly larger y-labels
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig("Eigen_" + str(N_datos) + "datos.png")


