#%%



import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Main import OpenMVCamera
from Main import TrimakerCOSMOS
import time
from datetime import datetime
import numpy as np



#%%



grua = TrimakerCOSMOS("COM11", clean_buffer = False, devolveme_la_respuesta=True)

# grua.home()

# Escribo la funicon home aca para ver que mensajes me da
print("Empieza el comando home():\n")
grua.enviar_gcode("G28", devolveme_la_respuesta = True)

# Setea la posicion de la camara 

grua.posicion_x = 0
grua.posicion_y = 0
grua.posicion_z = 0

# # Para que baje 20mm en Z
# print("\nEmpieza el comando desp_lin_rel():\n")
# grua.desp_lin_rel(0, 0, 20, devolveme_la_respuesta=True) 



#%%



cam = OpenMVCamera("COM7")

cam.conectar()



#%%



grua.enviar_gcode("M400", espera=1, devolveme_la_respuesta=True)



#%%




grua.enviar_gcode("G1 X20 F3000", 
espera=0.1, devolveme_la_respuesta=True)




#%%



grua.enviar_gcode("G1 X150 F3000", espera=0.1, devolveme_la_respuesta=True)
grua.enviar_gcode("M400", espera=0.1, devolveme_la_respuesta=True)

cam.mostrar(cam.capturar())



#%%



t0 = time.perf_counter()

# grua.ser.write(b"M400\n")

grua.ser.write(b"G1 X150 F3000\n")

grua.ser.write(b"M400\n")

cam.mostrar(cam.capturar())

grua.escuchar(tiempo_maximo=10)

t1 = time.perf_counter()    

print(f"Tiempo hasta el ok: {t1 - t0:.6f} s")



#%%



t0 = time.perf_counter()

grua.ser.write(b"G1 X150 F3000\n")

grua.escuchar(tiempo_maximo=5)



# %%



grua.ser.write(b"M400\n")

grua.escuchar(tiempo_maximo=5)



# %%



t0 = time.perf_counter()

grua.ser.write(b"G1 X150 F3000\n")

grua.escuchar(tiempo_maximo=10)

t1 = time.perf_counter()

print(f"Tiempo: {t1 - t0:.6f} s")



# %%
t0 = time.perf_counter()

grua.ser.write(b"G1 X150 F3000\n")

respuestas = grua.escuchar(tiempo_maximo=7)

t1 = time.perf_counter()



# %%
grua.ser.write(b"G1 X150 F3000\n")
grua.ser.write(b"G1 X20 F3000\n")

respuestas = grua.escuchar(tiempo_maximo=10)



# %%
grua.ser.write(b"G1 X150 F3000\n")

time.sleep(1)

grua.ser.write(b"M114\n")

time.sleep(0.2)

respuestas = grua.escuchar(tiempo_maximo=3)



# %%
