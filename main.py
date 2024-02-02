import numpy as np
import rusmodules
from rusmodules import rus

Material1 = rus.Material('./constantes.csv', 2, 'Monoclinica')
Material1.inicializar_phi()
print(Material1)