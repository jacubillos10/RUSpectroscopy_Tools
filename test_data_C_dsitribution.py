import numpy as np
import pandas as pd
from datamodules import preproc
from datamodules import generate_histograms

CXX = preproc.targets_default
datos = pd.read_csv("input_data/l_Unif_try2.csv", delimiter=",")#, on_bad_lines='skip') 
generate_histograms.graficar_histogramas(datos, CXX)
