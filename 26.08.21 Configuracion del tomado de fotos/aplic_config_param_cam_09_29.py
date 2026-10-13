#%%



# Agrego la raiz del proyecto
import sys
import os

os.environ["PYTHONPATH"] = r"C:\BackUp\Facultad\Labo 7\env_labo7"

sys.path.insert(0, os.environ["PYTHONPATH"])


# from pathlib import Path
# import re

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

import time
from datetime import datetime

from Main import OpenMVCamera
from Main import TrimakerCOSMOS


from busqueda_muestra import *



#%%



grua = TrimakerCOSMOS("COM11")

grua.home()

print(f"{grua.pregunta_posicion_actual() = }")

# grua.desp_lin_abs(150,150,27.72)

grua.desp_lin_abs(150,140,28)



#%%



cam = OpenMVCamera("COM7")

cam.conectar()

cam.mostrar(cam.capturar())



#%%



cam.set_pixformat("JPEG")

cam.mostrar(cam.capturar())

time.sleep(1)

cam.set_pixformat("GRAYSCALE")

cam.mostrar(cam.capturar())

time.sleep(1)

cam.set_pixformat("RGB565")

cam.mostrar(cam.capturar())



#%%



cam.config_autofoco_local(
)

mejor_z, posiciones_z, valores_metrica = autofocus(
    cam,
    grua,
    varianza_intensidad,
    pasos_abajo=15,
    pasos_arriba=15,
    paso_z=0.0025,
    espera_movimiento=0.3,
    graficamos=True,
    guardamos=False,
    carpeta_guardado=r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.25 nombre del dia\barrido z 3"
    )

cam.config_de_una()
# grua.pregunta_posicion_actual()
cam.mostrar(cam.capturar())
grua.pregunta_posicion_actual()

# cam.config_de_una()

# mejor_z, posiciones_z, valores_metrica = autofocus(
#     cam,
#     grua,
#     tenengrad,
#     pasos_abajo=4,
#     pasos_arriba=4,
#     paso_z=0.0025,
#     espera_movimiento=0.3,
#     graficamos=True,
#     guardamos=False,
#     carpeta_guardado=r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.25 nombre del dia\barrido z 3"
#     )

# cam.config_de_una()
# # grua.pregunta_posicion_actual()
# cam.mostrar(cam.capturar())
# grua.pregunta_posicion_actual()




pixformats = ["RGB565","GRAYSCALE"] #["JPEG", "RGB565","GRAYSCALE"]

# exposure_times = [
#     250,
#     # 500,
#     1000,
#     # 2500,
#     5000,
#     # 10000,
#     20000,
#     # 30000,
#     50000,
#     # 80000,
#     # 120000,
#     100000,
#     150000,
#     # 160000,
#     # 170000,
#     # 180000,
#     # 190000,
#     200000,
# ]

exposure_times = [
    150000,
    175000,
    200000,
    225000,
    250000,
    275000,
    300000    
]

gains = [0,
         3,
         6,
         9,
        #  12,
         15,
        #  20,
         30
         ] # Esta bien? No tengo idea de esto

framesizes = ["WQXGA2"] #["QVGA", "VGA", "HD", "FHD", "WQXGA2"]

# cam.set_framesize("WQXGA2")

for framesize in framesizes:

    for pixformat in pixformats:

        for gain in gains:

            cam.config_autofoco_local()

            mejor_z, posiciones_z, valores_metrica = autofocus(
                cam,
                grua,
                varianza_intensidad,
                pasos_abajo=3,
                pasos_arriba=3,
                paso_z=0.0025,
                espera_movimiento=0.3,
                graficamos=False,
                guardamos=False,
                carpeta_guardado=r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.25 nombre del dia\barrido z 3"
                )

            for exposure_time in exposure_times:

                cam.config_de_una(
                    pixformat=pixformat,
                    resolucion=framesize,
                    exposure_us=exposure_time,
                    gain_db=gain
                    )

                # time.sleep(1)

                carpeta_guardado = rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.29 Config param camara pasada 3\{framesize} - {pixformat}"


                hora_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                image_data = cam.capturar()
                
                cam.guardar_imagen(
                    image_data = image_data,
                    carpeta=carpeta_guardado,
                    nombre = rf"conf gain {gain}-{exposure_time}us. {hora_actual} ({grua.posicion_x}, {grua.posicion_y}, {grua.posicion_z})",
                    mostrar=False
                )
                
                time.sleep(1)



# %%
