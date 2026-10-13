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



mejor_z, posiciones_z, valores_metrica = autofocus(
    cam,
    grua,
    tenengrad,
    pasos_abajo=5,
    pasos_arriba=5,
    paso_z=0.0025,
    espera_movimiento=1,
    graficamos=True,
    guardamos=True,
    carpeta_guardado=r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.20 barrido mas video\Configuracion 5\Pasada 1"
    )


# %%

# Barrido

x_abs_ini = grua.pregunta_posicion_actual()[0]
y_abs_ini = grua.pregunta_posicion_actual()[1]
z_abs_ini = grua.pregunta_posicion_actual()[2]

grua.desp_lin_abs(x_abs_ini, y_abs_ini, z_abs_ini)

time.sleep(5)


rango_x = np.arange(x_abs_ini, x_abs_ini + 5 * 0.12, 0.12)
rango_y = np.arange(y_abs_ini, y_abs_ini + 5 * 0.08, 0.08)


for y_abs in rango_y:
    for x_abs in rango_x:

        z_abs_ini = grua.pregunta_posicion_actual()[2]

        grua.desp_lin_abs(x_abs, y_abs, z_abs_ini)

        time.sleep(1)

        mejor_z, posiciones_z, valores_metrica = autofocus(
            cam,
            grua,
            tenengrad,
            pasos_abajo=4,
            pasos_arriba=4,
            paso_z=0.0025,
            espera_movimiento=1,
            graficamos=True,
            guardamos=True,
            carpeta_guardado=rf"C:\BackUp\Facultad\Labo 7\Fotos\26.08.20 barrido mas video\barrido 3\x.{x_abs}-y.{y_abs}"
            )

        time.sleep(1)

        carpeta_guardado = r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.20 barrido mas video\barrido 3"

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
        time.sleep(1)


# %%

# Barrido pero sin mejor z


x_abs_ini = 150
y_abs_ini = 150

z_fija =  grua.pregunta_posicion_actual()[2] # Altura Z que querés mantener durante todo el barrido

grua.desp_lin_abs(
    x_abs_ini,
    y_abs_ini,
    z_fija
)

time.sleep(5)


rango_x = np.arange(
    x_abs_ini,
    x_abs_ini + 50 * 0.12,
    0.12
)

rango_y = np.arange(
    y_abs_ini,
    y_abs_ini + 50 * 0.08,
    0.08
)


for y_abs in rango_y:

    for x_abs in rango_x:

        # Movimiento únicamente en X e Y,
        # manteniendo siempre la misma altura Z.
        grua.desp_lin_abs(
            x_abs,
            y_abs,
            z_fija
        )

        time.sleep(1)

        carpeta_guardado = (
            rf"C:\BackUp\Facultad\Labo 7\Fotos"
            rf"\26.08.20 barrido mas video\barrido 2"
            rf"\x.{x_abs}-y.{y_abs}"
        )

        hora_actual = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        image_data = cam.capturar()

        if x_abs == 100:
            mostrar = True
        else:
            mostrar = False

        cam.guardar_imagen(
            image_data=image_data,
            carpeta=carpeta_guardado,
            nombre=f"{hora_actual} ({x_abs}, {y_abs}, {z_fija})",
            mostrar=mostrar
        )

        time.sleep(1)

    rango_x = rango_x[::-1]


# %%





x_abs_ini = 150
y_abs_ini = 150

# Altura Z fija para todo el barrido
z_fija = grua.pregunta_posicion_actual()[2]

# Llevar la cámara a la posición inicial
grua.desp_lin_abs(
    x_abs_ini,
    y_abs_ini,
    z_fija
)

time.sleep(1)


# Generar los rangos de X e Y
rango_x = np.arange(
    x_abs_ini,
    x_abs_ini + 50 * 0.12,
    0.12
)

rango_y = np.arange(
    y_abs_ini,
    y_abs_ini + 50 * 0.08,
    0.08
)


for y_abs in rango_y:

    for x_abs in rango_x:

        # Mover a la nueva posición manteniendo Z fija
        grua.desp_lin_abs(
            x_abs,
            y_abs,
            z_fija
        )

        time.sleep(1)

        # Carpeta donde se guarda la imagen
        carpeta_guardado = (
            rf"C:\BackUp\Facultad\Labo 7\Fotos"
            rf"\26.08.20 barrido mas video\barrido 2"
        )

        # Nombre de la imagen
        hora_actual = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        # Capturar imagen
        image_data = cam.capturar()

        # Mostrar solamente si se cumple la condición
        if x_abs == 100:
            mostrar = True
        else:
            mostrar = False

        # Guardar imagen
        cam.guardar_imagen(
            image_data=image_data,
            carpeta=carpeta_guardado,
            nombre=f"{hora_actual} ({x_abs}, {y_abs}, {z_fija})",
            mostrar=mostrar
        )

        time.sleep(1)

    # Invertir X para hacer el barrido en zig-zag
    rango_x = rango_x[::-1]


#%%