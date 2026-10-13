#%%



# Agrego la raiz del proyecto
import sys
import os

os.environ["PYTHONPATH"] = r"C:\BackUp\Facultad\Labo 7\env_labo7"

sys.path.insert(0, os.environ["PYTHONPATH"])

from pathlib import Path
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt

import time
from datetime import datetime

from metricas_enfoque import *



# %%

# "QQVGA",
# "QVGA",
# "VGA",
# "HD",
# "FHD",
# "WQXGA2"

carpetas = [
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.09.25 preparado - barrido a un cv de gls\barrido cerca del plano de enfoque\Pasada 8.5 xy 150_150\Pixformat RGB565- resolucion QVGA"
]

for carpeta in carpetas:

    print(f"\n\n\n\n{carpeta = }")

    resultados = analizar_carpeta_completo(carpeta)

    # print(resultados[0])
    grafico_del_maximo_de_la_metrica_con_imagenes(resultados, carpeta)

    time.sleep(2)



# %%
