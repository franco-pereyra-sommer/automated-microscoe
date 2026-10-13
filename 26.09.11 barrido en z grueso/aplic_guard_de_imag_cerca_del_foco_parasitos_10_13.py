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


from metricas_enfoque import (
    tenengrad, 
    varianza_intensidad,
    barrido_z_mas_guardar_imagenes
    )

from busqueda_muestra import *



#%%



from camara_openmv import OpenMVCamera
from impresora_cosmos import TrimakerCOSMOS


grua = TrimakerCOSMOS("COM11"); grua.home()
cam = OpenMVCamera("COM7"); cam.conectar()



#%%



# cam = OpenMVCamera("COM7")

# cam.conectar()

cam.set_framesize("HD") 

cam.set_pixformat("RGB565") 

cam.set_pixformat("JPEG") 

cam.set_exposure_auto()

cam.set_gain_auto()
# cam.set_gain_auto()
# cam.set_whitebal_auto()



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

_,_, z_actual = grua.pregunta_posicion_actual()

# Dependiendo esto sale toda la medicion



#%%


grua.home()

# time.sleep(15)


for xy in [[150,150], [150.0125, 150], [150, 150.0125], [150.0125, 150.0125]]:

    x = xy[0]
    y = xy[1]


    barrido_z_mas_guardar_imagenes(
        cam,
        grua,
        x,
        y,
        z_max = z_actual+0.06,
        z_limites = [z_actual-0.06],
        pasos_lista = [0.0025],
        tiempo_espera_lista = [0.5],
        cam_pixformat = ["RGB565"], #JPEG
        cam_framesize = ["QVGA", "WQXGA2"],
        carpeta_guardado = rf"C:\BackUp\Facultad\Labo 7\Fotos\Barrido cerca del foco parasitos x{xy[0]}-y{xy[1]}"
    )


# %% 