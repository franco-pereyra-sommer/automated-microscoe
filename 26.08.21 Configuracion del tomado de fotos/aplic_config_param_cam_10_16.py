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


from busqueda_muestra import *

from camara_openmv import OpenMVCamera
from impresora_cosmos import TrimakerCOSMOS



#%%


grua = TrimakerCOSMOS("COM11"); grua.home()
cam = OpenMVCamera("COM7"); cam.conectar()


#%%


# 1) Parar la platina en un campo visual INTERIOR de la muestra
posicionar_en_muestra(cam, grua, x=150, y=150,
                    z_arriba_seguro=28.5,
                    z_min_seguridad=27.5, verificar_presencia=False,
                    hacer_foco_final=True, paso_grueso=0.005,
                    metrica_foco_final=laplacian_variance 
                    )




#%%



cam.config_autofoco_local(
)

mejor_z, posiciones_z, valores_metrica = autofocus(
    cam,
    grua,
    laplacian_variance,
    pasos_abajo=15,
    pasos_arriba=15,
    paso_z=0.0025,
    espera_movimiento=0.3,
    graficamos=True,
    guardamos=False,
    carpeta_guardado=r"C:\BackUp\Facultad\Labo 7\Fotos\26"
    )

cam.config_de_una()
# grua.pregunta_posicion_actual()
cam.mostrar(cam.capturar())
grua.pregunta_posicion_actual()


#%%


pixformats = ["RGB565"] #["JPEG", "RGB565","GRAYSCALE"]

# exposure_times = [
#     15000,
#     # 175000,
#     # 20000,
#     # 225000,
#     # 25000,
#     # 275000,
#     # 300000,
#     50000,
#     # 75000,
#     100000,
#     # 125000,
#     # 175000
#     150000  
# ]

exposure_times = [
    # 15000,
    # 175000,
    # 20000,
    # 225000,
    # 25000,
    # 275000,
    15000,
    20000,
    25000,
    30000,
    45000,
    # 50000,
    # 60000,
    # 75000,
    # 100000,
    # 125000,
    # 175000,
    # 150000
    # 200000  
]

# gains = [
#     # 0,
#     3,
#     # 6,
#     9,
#     15,
#     25,
#     30
#     ] # Esta bien? No tengo idea de esto

gains = [
    # 0,
    # 3,
    # 6,
    # 9,
    # 15,
    # 25,
    30
    ] # Esta bien? No tengo idea de esto

framesizes = ["QVGA"]#["WQXGA2"] #["QVGA", "VGA", "HD", "FHD", "WQXGA2"]

whitebals = [[66.22236, 61.43028, 60.206]]

# cam.set_framesize("WQXGA2")


for framesize in framesizes:

    for pixformat in pixformats:

        for gain in gains:

            cam.config_autofoco_local()

            mejor_z, posiciones_z, valores_metrica = autofocus(
                cam,
                grua,
                laplacian_variance,
                pasos_abajo=3,
                pasos_arriba=3,
                paso_z=0.0025,
                espera_movimiento=0.5,
                graficamos=False,
                guardamos=False,
                carpeta_guardado=r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.25 nombre del dia\barrido z 3"
                )

            for exposure_time in exposure_times:

                for whitebal in whitebals:


                    cam.config_de_una(
                        pixformat=pixformat,
                        resolucion=framesize,
                        exposure_us=exposure_time,
                        gain_db=gain,
                        whitebal=whitebal
                        )

                    # time.sleep(1)

                    carpeta_guardado = rf"C:\BackUp\Facultad\Labo 7\Fotos\param camara parasitos fijos\x{grua.posicion_x}-y{grua.posicion_y} {framesize} - {pixformat}"


                    hora_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                    image_data = cam.capturar()
                    
                    cam.guardar_imagen(
                        image_data = image_data,
                        carpeta=carpeta_guardado,
                        nombre = rf"conf gain {gain}-{exposure_time}us. {hora_actual} ({grua.posicion_x}, {grua.posicion_y}, {grua.posicion_z}) r{whitebal[0]}-g{whitebal[1]}-b{whitebal[2]}",
                        mostrar=False
                    )
                    
                    time.sleep(1)



# %%
