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

# print(f"{grua.pregunta_posicion_actual() = }")



#%%



tiempos = [1]

for tiempo_esp in tiempos:

    grua.coord_abs()

    grua.ser.write(b"G1 X20 F3000\n")

    time.sleep(5)


    grua.ser.reset_input_buffer()

    t0 = time.perf_counter()

    grua.ser.write(b"G1 X150 F3000\n")

    print(f"{tiempo_esp = }")

    time.sleep(tiempo_esp)

    grua.ser.write(b"G203\n")

    # grua.ser.write(b"M114\n")

    grua.escuchar(tiempo_maximo=2)

    print("\n\n\n")



# %%



grua.ser.reset_input_buffer()

grua.ser.write(b"M115\n")

grua.escuchar(tiempo_maximo=2)



# %%



grua.ser.reset_input_buffer()

grua.ser.write(b"M503\n")

grua.escuchar(tiempo_maximo=2)




# %%



grua.ser.reset_input_buffer()

grua.ser.write(b"M400\n")

grua.escuchar(tiempo_maximo=2)



# %%



grua.coord_abs()

grua.ser.write(b"G1 X20 F3000\n")

time.sleep(2)


grua.ser.reset_input_buffer()

grua.coord_abs()

grua.ser.reset_input_buffer()

t0 = time.perf_counter()

grua.ser.write(b"G1 X150 F3000\n")
grua.ser.write(b"G4 P0\n")

respuestas = grua.escuchar(tiempo_maximo=10)

t1 = time.perf_counter()

print(f"Tiempo total: {t1 - t0:.6f} s")



# %%

####
#### Segunda prueba
####


grua.ser.reset_input_buffer()

t0 = time.perf_counter()

grua.ser.write(b"G1 X150 F3000\n")

grua.esperar_ok()

t1 = time.perf_counter()

print(f"Tiempo hasta ok: {t1-t0:.6f} s")



#%%


grua.ser.write(b"G4 P0\n")

t0 = time.perf_counter()

grua.esperar_ok()

t1 = time.perf_counter()

print(f"Tiempo hasta ok de G4: {t1-t0:.6f} s")