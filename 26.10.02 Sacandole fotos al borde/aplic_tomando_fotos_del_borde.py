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

from camara_openmv import OpenMVCamera
from impresora_cosmos import TrimakerCOSMOS


from busqueda_muestra import *



#%%



grua = TrimakerCOSMOS("COM11")

grua.home()

print(f"{grua.pregunta_posicion_actual() = }")



#%%



cam = OpenMVCamera("COM7")

cam.conectar()



#%%



posicionar_en_muestra(
    cam,
    grua,
    x=150,
    y=150,
    z_arriba_seguro=28,
    z_min_seguridad=27.5,
    resolucion_global="QVGA",
    resolucion_local=None,
    pixformat_global="GRAYSCALE",
    paso_grueso=0.005,
    paso_fino=0.0025,
    criterios_deteccion=None,
    margen_confirmacion=0.15,
    pasos_confirmacion=6,
    verificar_presencia=False,
    alturas_verificacion=3,
    umbral_intensidad_fondo=152.5,
    feedrate_xy=3000,
    espera_movimiento=0.5,
    hacer_foco_final=True,
    metrica_foco_final=laplacian_variance,
    pasos_arriba_foco_final=5,
    pasos_abajo_foco_final=5,
    guardar_fotos_foco_final=False,
    carpeta_fotos_foco_final=None,
    foco_final_multimetrica_activado=False,
    pixformat_color_foco_final="RGB565",
    resolucion_color_foco_final="WQXGA2",
    devolveme_la_respuesta=False,
)


# mejor_z, posiciones_z, valores_metrica = autofocus(
#     cam,
#     grua,
#     tenengrad,
#     pasos_abajo=5,
#     pasos_arriba=5,
#     paso_z=0.0025,
#     espera_movimiento=1,
#     graficamos=True,
#     guardamos=True,
#     carpeta_guardado=r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.20 barrido mas video\Configuracion 5\Pasada 1"
#     )



# %%

# Barrido

carpeta_guardado = r"C:\BackUp\Facultad\Labo 7\Fotos\PONER NOMBRE"

x_abs_ini = grua.pregunta_posicion_actual()[0]
y_abs_ini = grua.pregunta_posicion_actual()[1]
z_abs_ini = grua.pregunta_posicion_actual()[2]

grua.desp_lin_abs(x_abs_ini, y_abs_ini, z_abs_ini)

time.sleep(5)

# x_abs_ini = 10
rango_x = np.arange(x_abs_ini, x_abs_ini + 8 * 0.075, 0.075)
# rango_x= np.arange(x_abs_ini, x_abs_ini - 8 * 0.075, -0.075)
# print(f"{rango_x}")
rango_y = np.arange(y_abs_ini - 4 * 0.1, y_abs_ini + 4 * 0.1, 0.1)


for y_abs in rango_y:

    for x_abs in rango_x:

        grua.desp_lin_abs(x_abs, y_abs, z_abs_ini)

        time.sleep(0.5)

        x, y, z_abs_ini = grua.pregunta_posicion_actual()

        carpeta_af = carpeta_guardado / f"({x}, {y})"

        mejor_z, posiciones_z, valores_metrica = autofocus(
            cam,
            grua,
            laplacian_variance,
            pasos_abajo=4,
            pasos_arriba=4,
            paso_z=0.0025,
            espera_movimiento=0.5,
            graficamos=False,
            guardamos=True,
            carpeta_guardado=carpeta_af
            )

        time.sleep(0.5)


        hora_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        image_data = cam.capturar()


        if x_abs == 100:
            mostrar = True
        
        else:
            mostrar = False

        
        cam.guardar_imagen(
            image_data = image_data,
            carpeta=carpeta_guardado,
            nombre = f"{hora_actual} ({x_abs}, {y_abs}, {z_abs_ini})",
            mostrar=mostrar
        )
        
    rango_x = rango_x[::-1]

