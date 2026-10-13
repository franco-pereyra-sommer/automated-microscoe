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



carpetas = [
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 1\Pasada 1",
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 1\Pasada 2",
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 4\Pasada 1",
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 4\Pasada 2 menos luz",
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 4\Pasada 3 menos luz (lo mismo porque la metrica esta rara)",
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 4\Pasada 4 mas luz",
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 4\Pasada 5 cambio la region xy",
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 5\Pasada 1",
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 5\Pasada 2 menos luz",
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 5\Pasada 3 mucha luz",
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 6\Pasada 1",
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 6\Pasada 2 menos luz",
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 7\Pasada 1",
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 7\Pasada 2 menos luz",
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 8\Pasada 1",
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 8\Pasada 2 menos luz",
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 9\Pasada 1",
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 10\Pasada 1"
]

for carpeta in carpetas:
    # carpeta = r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 1\Pasada 1"

    print(f"\n\n\n\n{carpeta = }")

    resultados = analizar_carpeta(carpeta)

    # No haria que se printeen los resultados, con el grafico esta bien
    # mostrar_maximos(resultados) 

    graficar_metricas_mas_imagenes(resultados, carpeta)

    time.sleep(8)



# %%

## Viendo las fotos del 26.09.06


carpetas = [
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.09.03 fotos sv\ desplazamiento 0\Pasada QVGA",
    # r"C:\BackUp\Facultad\Labo 7\Fotos\26.09.03 fotos sv\ desplazamiento 0\Pasada FHD",
    # r"C:\BackUp\Facultad\Labo 7\Fotos\26.09.03 fotos sv\ desplazamiento 0\Pasada WQXGA2"
]

for carpeta in carpetas:
    # carpeta = r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 1\Pasada 1"

    print(f"\n\n\n\n{carpeta = }")

    resultados = analizar_carpeta(carpeta)

    # No haria que se printeen los resultados, con el grafico esta bien
    # mostrar_maximos(resultados) 

    # print(resultados)

    # Esto es para graficar el tenengrad del barrido

    zs = []
    metricas =[]

    for i in range(len(resultados)):
        zs.append(resultados[i]["z"])
        metricas.append(resultados[i]["tenengrad"])

    fig, ax = plt.subplots()
    
    ax.plot(
        zs,
        metricas,
        marker="o"
    )

    ax.scatter(
        zs[np.argmax(np.array(metricas))],
        metricas[np.argmax(np.array(metricas))],
        s=80,
        label="Máximo"
    )

    ax.set_xlabel("Z [mm]")
    ax.set_ylabel("tenengrad")
    ax.set_title("Metricas para el autofocus")
    ax.grid()
    ax.legend()

    plt.show()

    graficar_metricas_mas_imagenes_vieja(resultados, carpeta)

    time.sleep(2)


# %%
