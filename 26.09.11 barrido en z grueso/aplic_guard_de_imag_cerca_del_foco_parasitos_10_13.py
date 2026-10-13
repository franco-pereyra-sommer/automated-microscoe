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



from camara_openmv import *
from impresora_cosmos import TrimakerCOSMOS


grua = TrimakerCOSMOS("COM11"); grua.home()
cam = OpenMVCamera("COM7"); cam.conectar()



#%%



# cam = OpenMVCamera("COM7")

# cam.conectar()

# cam.set_framesize(Framesize.QVGA) 

# cam.set_pixformat("RGB565") 

# cam.set_pixformat("JPEG") 

# cam.set_exposure_auto()

# cam.set_gain_auto()


cam.set_exposure(18000)

cam.set_gain(35)


cam.set_whitebal_auto()

cam.descartar_frames(20)

cam.mostrar(cam.capturar())

cam.set_whitebal(35, 41. , 46.5)

cam.descartar_frames(20)

cam.mostrar(cam.capturar())



#%%



import io, numpy as np
from PIL import Image

def canales(data):
    m = np.array(Image.open(io.BytesIO(data))).reshape(-1, 3).mean(axis=0)
    return m.round(1), (m[0]/m[1]).round(3), (m[2]/m[1]).round(3)   # medias RGB, R/G, B/G

cam.config_de_una("RGB565", "QVGA")
cam.set_exposure(18000)
cam.set_gain(35)
cam.set_whitebal_auto()
cam.descartar_frames(20)

cam.ser.reset_input_buffer()
cam.ser.write(b"info\n")
print(cam.ser.readline().decode())          # ej.: 8000 1.5 (2.1, 0.0, 3.4)
print("auto  :", canales(cam.capturar()))
cam.mostrar(cam.capturar())

r, g, b = 2.1, 0.0, 3.4                     # <- lo que imprimió info
cam.set_whitebal(r, g, b); cam.descartar_frames(20)
print("manual igual al auto:", canales(cam.capturar()))   # tiene que dar casi lo mismo
cam.mostrar(cam.capturar())

cam.set_whitebal(r - 6, g, b); cam.descartar_frames(20)
print("R -6 dB:", canales(cam.capturar()))                # R/G tiene que bajar a ~la mitad
cam.mostrar(cam.capturar())


#%%



import re

cam.set_exposure(18000)
cam.set_gain(35)
cam.set_whitebal_auto()
cam.descartar_frames(20)

cam.ser.reset_input_buffer(); cam.ser.write(b"info\n")
exp_us, gain_db, r, g, b = map(float, re.findall(r"[-\d.]+", cam.ser.readline().decode()))
print(f"{exp_us = }")
print(f"{gain_db = }")
print(f"{r = }")
print(f"{g = }")
print(f"{b = }")
print("auto  :", canales(cam.capturar()))
cam.mostrar(cam.capturar())

cam.set_whitebal(r, g, b); cam.descartar_frames(20)
print("manual = auto:", canales(cam.capturar()))      # debería dar casi igual al auto
cam.mostrar(cam.capturar())

cam.set_whitebal(r - 6.02, g, b); cam.descartar_frames(20)
print("R a la mitad:", canales(cam.capturar()))       # R/G debería bajar a ~la mitad (0.896 -> ~0.45)
cam.mostrar(cam.capturar())



#%%



# cam.config_de_una()

cam.set_exposure_auto()
cam.set_gain_auto()

cam.ser.reset_input_buffer()
cam.ser.write(b"info\n")
exp_us, gain_db, r, g, b = map(float, re.findall(r"[-\d.]+", cam.ser.readline().decode()))

print(f"{exp_us = }")
print(f"{gain_db = }")
print(f"{r = }")
print(f"{g = }")
print(f"{b = }")


posicionar_en_muestra(
    cam,
    grua,
    x=150,
    y=150,
    z_arriba_seguro=27.9,
    z_min_seguridad=27.6,
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
    resolucion_color_foco_final="QVGA",
    devolveme_la_respuesta=False,
)

_,_, z_actual = grua.pregunta_posicion_actual() # (np.float64(150.0), np.float64(150.0), np.float64(27.755))

# Dependiendo esto sale toda la medicion

cam.mostrar(cam.capturar())

cam.ser.reset_input_buffer()
cam.ser.write(b"info\n")
exp_us, gain_db, r, g, b = map(float, re.findall(r"[-\d.]+", cam.ser.readline().decode()))

print(f"{exp_us = }")
print(f"{gain_db = }")
print(f"{r = }")
print(f"{g = }")
print(f"{b = }")



#%%


# grua.home()

# time.sleep(15)


for xy in [[150.5,150], [150.0125, 150], [150, 150.0125], [150.0125, 150.0125]]:

    for gain in [23.8, 25, 35]:#[15, 23.8, 25, 35]:

        for exp in [20000, 21450, 50000, 120000]:

            for rel in ["QVGA"]: #["QVGA", "WQXGA2"]

                for pixformat in ["RGB565"]:#["RGB565", "GRAYSCALE"]:

                    x = xy[0]
                    y = xy[1]


                    barrido_z_mas_guardar_imagenes(
                        cam,
                        grua,
                        x,
                        y,
                        z_max = z_actual+0.035,
                        z_limites = [z_actual-0.035],
                        pasos_lista = [0.0025],
                        tiempo_espera_lista = [0.5],
                        cam_pixformat = [pixformat], #["RGB565", "GRAYSCALE"], #JPEG
                        cam_framesize = [rel], #["QVGA"],# ["WQXGA2"],
                        cam_exposure=[exp],
                        cam_gain=[gain],
                        carpeta_guardado = rf"C:\BackUp\Facultad\Labo 7\Fotos\Barrido cerca del foco parasitos 3 x{xy[0]}-y{xy[1]}"
                    )


# %%
