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

# Barrido de medición: una foto en COLOR, enfocada, en cada punto de una grilla.
#
# Antes de correr esta celda la platina tiene que estar ENFOCADA cerca del borde
# que se quiere medir (con posicionar_en_muestra() en la celda de arriba, o a
# mano). La grilla arranca en la posición actual de la platina, y el SIGNO de
# paso_x_mm / paso_y_mm decide hacia dónde se extiende en x y en y (por ejemplo,
# arrancar adentro de la muestra y avanzar hacia el borde).
#
# Cada barrido crea su propia carpeta dentro de carpeta_salida, con las fotos
# en color, puntos.csv (posición, z_enfoque y características de cada foto) y
# parametros.json. Después hay que repartir las fotos a mano en carpetas
# interior / borde / vacio y usar analisis_campo_visual.resumir_calibracion().
#
# Los pasos no pueden pasar de 0.15 mm (regla de seguridad del eje Z).
# (barrido_fotos_borde.py tiene que estar en la misma carpeta que este archivo)

from barrido_fotos_borde import barrer_fotos_borde

resultado = barrer_fotos_borde(
    cam,
    grua,
    z_arriba_seguro=28,          # los mismos que en posicionar_en_muestra()
    z_min_seguridad=27.5,
    carpeta_salida=r"C:\BackUp\Facultad\Labo 7\Fotos\PONER NOMBRE",
    nombre_barrido=None,         # None -> "barrido <fecha> (x, y)"
    n_x=8,
    n_y=8,
    paso_x_mm=0.075,
    paso_y_mm=0.100,
    # Cámara en la foto en color (None = automático). Para comparar
    # fraccion_rojos / saturacion_media entre fotos conviene fijarlos.
    exposure_us=None,
    gain_db=None,
    whitebal=None,               # (r, g, b) en dB
    guardar_fotos_autofoco=False,
    # Si no es None, en los campos claros (vacíos) no se corre autofocus():
    # evita que el Z siga ruido. Calibrar con la columna intensidad_fondo del CSV.
    umbral_intensidad_fondo=None,
)


# %%

# Mapa del barrido: dónde quedó el borde dentro de la grilla.
# (otras opciones: "contraste", "fraccion_rojos", "saturacion_media", "z_enfoque")

resultado.graficar_mapa("distancia_normalizada")
