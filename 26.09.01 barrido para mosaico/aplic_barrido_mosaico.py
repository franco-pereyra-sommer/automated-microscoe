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

from Main import OpenMVCamera
from Main import TrimakerCOSMOS


from metricas_enfoque import *



#%%



grua = TrimakerCOSMOS("COM11")

grua.home()

print(f"{grua.pregunta_posicion_actual() = }")



#%%



cam = OpenMVCamera("COM7")

cam.conectar()



#%%

grua.desp_lin_rel(0,0.06,0)

for n in range(3, 10):

    # cam.mostrar(cam.capturar())
    grua.desp_lin_rel(0,-0.06,0)
    # cam.mostrar(cam.capturar())
    # grua.desp_lin_rel(0.4,0,0)

    for resolucion in ["QVGA", "FHD", "WQXGA2"]:
        cam.set_framesize(resolucion)

        # set_framesize
        #             "QQVGA",
        #             "FHD",
        #             "WQXGA2"


        mejor_z, posiciones_z, valores_metrica = autofocus(
            cam,
            grua,
            tenengrad,
            pasos_abajo=10,
            pasos_arriba=10,
            paso_z=0.0025,
            espera_movimiento=0.3,
            graficamos=True,
            guardamos=True,
            carpeta_guardado=rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.03 fotos sv\ desplazamiento {n}\Pasada {resolucion}"
            # carpeta_guardado=rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.01 fotos para metricas\Campo densidad {grua.posicion_x}\Pasada {resolucion}"
            )



# %%

## PASASDDA DE PARAMETROSZ DE LA CAMARA


mejor_z, posiciones_z, valores_metrica = autofocus(
    cam,
    grua,
    tenengrad,
    pasos_abajo=8,
    pasos_arriba=8,
    paso_z=0.0025,
    espera_movimiento=0.5,
    graficamos=True,
    guardamos=False,
    carpeta_guardado=r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.25 nombre del dia\barrido z 3"
    )


cam.mostrar(cam.capturar())


pixformats = ["RGB565"] #,"GRAYSCALE"]

exposure_times = [
    250,
    500,
    1000,
    # 2500,
    5000,
    10000,
    20000,
    30000,
    50000,
    80000,
    120000,
    # 150000,
    160000,
    # 170000,
    180000,
    # 190000,
    200000,

]

gains = [
    0, 
    3, 
    6, 
    9, 
    # 12, 
    15, 
    # 20, 
    30] # Esta bien? No tengo idea de esto

cam.set_framesize("WQXGA2")

for pixformat in pixformats:
    for gain in gains:

        mejor_z, posiciones_z, valores_metrica = autofocus(
            cam,
            grua,
            tenengrad,
            pasos_abajo=3,
            pasos_arriba=3,
            paso_z=0.0025,
            espera_movimiento=0.5,
            graficamos=False,
            guardamos=False,
            carpeta_guardado=r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.25 nombre del dia\barrido z 3"
            )

        for exposure_time in exposure_times:

            # mejor_z, posiciones_z, valores_metrica = autofocus(
            #     cam,
            #     grua,
            #     tenengrad,
            #     pasos_abajo=4,
            #     pasos_arriba=4,
            #     paso_z=0.0025,
            #     espera_movimiento=0.3,
            #     graficamos=True,
            #     guardamos=True,
            #     carpeta_guardado=rf"C:\BackUp\Facultad\Labo 7\Fotos\26.08.25 nombre del dia\prueba 1\conf {pixformat}-{exposure_time}us-{gain}"
            #     )

            # time.sleep(1)

            carpeta_guardado = r"C:\BackUp\Facultad\Labo 7\Fotos\26.09.03 fotos sv\parametros 1"
            # C:\BackUp\Facultad\Labo 7\Fotos\26.09.03 fotos sv

            hora_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            image_data = cam.capturar()
            
            cam.guardar_imagen(
                image_data = image_data,
                carpeta=carpeta_guardado,
                nombre = rf"conf {pixformat}-gain {gain}-{exposure_time}us. {hora_actual} ({grua.posicion_x}, {grua.posicion_y}, {grua.posicion_z})",
                mostrar=False
            )
            
            time.sleep(1)
# %%
