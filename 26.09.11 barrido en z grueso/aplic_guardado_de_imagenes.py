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

from Main import (
    OpenMVCamera,
    TrimakerCOSMOS
    )

from metricas_enfoque import (
    tenengrad, 
    varianza_intensidad,
    barrido_z_mas_guardar_imagenes
    )



#%%



grua = TrimakerCOSMOS("COM11")

grua.escuchar(tiempo_maximo=3)

grua.home(devolveme_timepos=True)

# grua.escuchar(tiempo_maximo=1)



#%%



cam = OpenMVCamera("COM7")

cam.conectar()

cam.set_framesize("HD") 

cam.set_pixformat("RGB565") 

cam.set_pixformat("JPEG") 

cam.set_exposure_auto()

cam.set_gain_auto()
# cam.set_gain_auto()
# cam.set_whitebal_auto()



#%%



z = 28.3

z_limites = [28.05, 27.09, 26.875, 26.84, 26.74]

pasos_lista = [0.1, 0.025, 0.005, 0.0025, 0.005]

resolucion_z = 0.0025

numero = 0

for z_min, paso in zip(z_limites, pasos_lista):

    print(f"{paso = }")

    zs = np.arange(z, z_min, -paso)

    if numero > 0:
        zs = zs[1:]

    print(f"{z_min = }")

    print(f"{zs = }")

    print(f"{np.round(zs[-1], 4) = }")

    z = resolucion_z * np.floor((np.round(zs[-1], 4) / resolucion_z)+0)

    print(f"{z = }\n\n")

    numero = numero + 1



#%%



# carpeta_guardado = rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.11 barrido gruso z en disitintos cv"

# tiempo_espera = 0.3

# x = 150 # Fijo

# y = 150 # Fijo

# z_max = 30

# z_limites = [28] # Porque existe una lista que contiene z's? -> Para poder decir por ejemplo: desde z=30 hasta z=29 movete con un paso de 100 micrones; desde z=29 hasta z=28 movete de a 50 micrones ...

# paso_lista = [0.05]

grua.home()

# time.sleep(15)


for xy in [[150,150]]:

    x = xy[0]
    y = xy[1]


    barrido_z_mas_guardar_imagenes(
        cam,
        grua,
        x,
        y,
        z_max = 29,
        z_limites = [28.1, 27.95, 27.91, 27.805, 27.78, 27.73], #[28.05, 27.09, 27.875, 27.84, 27.74], #ESTO TIENE QUE SER DECRECIENTE
        pasos_lista = [0.1, 0.025, 0.01, 0.0025, 0.005, 0.01],
        tiempo_espera_lista = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        cam_pixformat = ["RGB565"], #JPEG
        cam_framesize = ["QVGA", "HD", "FHD"], #["WQXGA2"],
        carpeta_guardado = rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 4 xy {x}_{y}"
    )


# %% Vidrio 1 (vidrio que esta por arriba de la muestra)



grua.home()

# time.sleep(15)


for xy in [[160,150]]:

    x = xy[0]
    y = xy[1]


    # barrido_z_mas_guardar_imagenes(
    #     cam,
    #     grua,
    #     x,
    #     y,
    #     z_max = 29,
    #     z_limites = [27.87, 27.73],#[28.1, 27.975, 27.90, 27.84, 27.8, 27.78, 27.6], #[28.05, 27.09, 27.875, 27.84, 27.74], #ESTO TIENE QUE SER DECRECIENTE
    #     pasos_lista = [0.1, 0.025, 0.01, 0.005, 0.0025, 0.005, 0.01],
    #     tiempo_espera_lista = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    #     cam_pixformat = ["RGB565"], #JPEG
    #     cam_framesize = ["QVGA", "HD", "FHD"], #["WQXGA2"],
    #     carpeta_guardado = rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 0 xy {x}_{y}"
    # )

barrido_z_mas_guardar_imagenes(
        cam,
        grua,
        x,
        y,
        z_max = 27.87,
        z_limites = [27.73], #ESTO TIENE QUE SER DECRECIENTE
        pasos_lista = [0.0025],
        tiempo_espera_lista = [0.3],
        cam_pixformat = ["RGB565"], #JPEG
        cam_framesize = ["QVGA", "HD", "FHD"], #["WQXGA2"],
        carpeta_guardado = rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 1 xy {x}_{y}"
    )



# %% Vidrio 1 (vidrio que esta por arriba de la muestra EL BORDE)



grua.home()

# time.sleep(15)


for xy in [[161,150]]:

    x = xy[0]
    y = xy[1]


    barrido_z_mas_guardar_imagenes(
        cam,
        grua,
        x,
        y,
        z_max = 29,
        z_limites = [28.1, 28.05, 27.7], #[28.05, 27.09, 27.875, 27.84, 27.74], #ESTO TIENE QUE SER DECRECIENTE
        pasos_lista = [0.1, 0.025, 0.0025],
        tiempo_espera_lista = [0.3, 0.3, 0.3],
        cam_pixformat = ["RGB565"], #JPEG
        cam_framesize = ["QVGA", "HD", "FHD"], #["WQXGA2"],
        carpeta_guardado = rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 1 xy {x}_{y}"
    )




# %% Vidrio 2 (vidrio que esta por abajo)



grua.home()

# time.sleep(15)


for xy in [[165,150]]:

    x = xy[0]
    y = xy[1]


    barrido_z_mas_guardar_imagenes(
        cam,
        grua,
        x,
        y,
        z_max = 29,
        z_limites = [28.15, 27.55, 27.15], #[28.05, 27.09, 27.875, 27.84, 27.74], #ESTO TIENE QUE SER DECRECIENTE
        pasos_lista = [0.05, 0.005, 0.0025],
        tiempo_espera_lista = [0.3, 0.3, 0.3],
        cam_pixformat = ["RGB565"], #JPEG
        cam_framesize = ["QVGA", "HD", "FHD"], #["WQXGA2"],
        carpeta_guardado = rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 0 xy {x}_{y}"
    )

