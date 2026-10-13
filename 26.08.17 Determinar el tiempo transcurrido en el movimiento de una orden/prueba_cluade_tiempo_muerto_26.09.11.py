#%%



# Agrego la raiz del proyecto
import sys
import os

os.environ["PYTHONPATH"] = r"C:\BackUp\Facultad\Labo 7\env_labo7"

sys.path.insert(0, os.environ["PYTHONPATH"])


import time
import numpy as np

from Main import (
    OpenMVCamera,
    TrimakerCOSMOS
)



#%%



grua = TrimakerCOSMOS("COM11")

grua.home()

# print(f"{grua.pregunta_posicion_actual() = }")




#%%



cam = OpenMVCamera("COM7")

cam.conectar()



#%%



target_start = {"x": 0, "y": 0, "z": 40}
target_finish = {"x": 150, "y": 150, "z": 40}

grua.desp_lin_abs(**target_start)

time.sleep(10)


# Prueba 1: confirmar que M114 salta directo al destino
grua.desp_lin_abs(**target_finish)
grua.test_M114_polling(target_finish, poll_interval= 0.3)



#%%



target_start = {"x": 0, "y": 0, "z": 40}
target_finish = {"x": 150, "y": 150, "z": 40}


grua.desp_lin_abs(**target_start)

time.sleep(10)

# Prueba 2: la solución real
grua.desp_lin_abs(**target_finish)
grua.wait_until_stopped_M400()
cam.mostrar(cam.capturar())



# %%



target_start = {"x": 0, "y": 0, "z": 40}
target_finish = {"x": 150, "y": 150, "z": 40}


grua.desp_lin_abs(**target_start)

time.sleep(10)

t0 = time.time()
grua.desp_lin_abs(**target_finish)
t1 = time.time()
print(f"desp_lin_abs() volvió a Python en {t1 - t0:.2f}s")

grua.wait_until_stopped_M400()
t2 = time.time()
print(f"wait_until_stopped_M400() volvió en {t2 - t1:.2f}s")
print(f"TIEMPO TOTAL (G1 -> M400 listo): {t2 - t0:.2f}s")



# %%



def enviar_directo(ser, gcode, timeout=5):
    """Manda una línea de gcode cruda y bloquea hasta ver 'ok'."""
    ser.write((gcode + "\n").encode())
    start = time.time()
    while time.time() - start < timeout:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print(f"[{gcode}] -> {line}")
            if line.lower().startswith("ok"):
                return True
    raise TimeoutError(f"No llegó 'ok' para: {gcode}")


def mover_directo(ser, target, feedrate=3000):
    x = target["x"] + grua.offset_x
    y = target["y"] + grua.offset_y
    z = target["z"] + grua.offset_z
    enviar_directo(ser, "G90")  # fuerzo modo absoluto, por las dudas
    enviar_directo(ser, f"G1 Z{z} F{feedrate}")
    enviar_directo(ser, f"G1 X{x} Y{y} F{feedrate}")


enviar_directo(grua.ser, "M9999")



#%%



target_start  = {"x": 0,   "y": 0,   "z": 40}
target_finish = {"x": 150, "y": 150, "z": 40}

# Ir al punto de partida
mover_directo(grua.ser, target_start)
time.sleep(10)

# --- Prueba cruda ---
t0 = time.time()
mover_directo(grua.ser, target_finish)
t1 = time.time()
print(f"\nG1 enviados, 'ok' recibido en {t1 - t0:.2f}s")

grua.ser.reset_input_buffer()
grua.ser.write(b"M400\n")
start = time.time()
while True:
    line = grua.ser.readline().decode(errors="ignore").strip()
    if line:
        print(f"[M400] recibido: {line}")
        if line.lower().startswith("ok"):
            break
t2 = time.time()

print(f"M400 tardó {t2 - t1:.2f}s en confirmar")
print(f"TIEMPO TOTAL (G1 -> M400 listo): {t2 - t0:.2f}s")



#%%



import threading

# Arrancar el logueo ANTES de mover, y dejarlo corriendo durante todo el movimiento + M400
resultados = []
def logger():
    resultados.extend(grua.escuchar(tiempo_maximo=8))

hilo = threading.Thread(target=logger)
hilo.start()

t0 = time.time()
grua.desp_lin_abs(**target_finish)
t1 = time.time()
print(f"desp_lin_abs volvió en {t1-t0:.2f}s")

grua.ser.write(b"M400\n")   # mandado directo, sin reset_input_buffer, para no perder nada
t2 = time.time()

hilo.join()
print(f"M400 enviado en t={t2-t0:.2f}s desde el inicio")



# %%



grua.desp_lin_abs(0,0,40)

time.sleep(6)

# Funcion modificada de la de arriba, tiene que empezar desde el x=0, y=0

def drenar_buffer(ser, silencio=0.3, printear = False):
    """Lee y descarta todo hasta que no llegue nada nuevo durante 'silencio' segundos."""
    ultima_llegada = time.time()
    while time.time() - ultima_llegada < silencio:
        if ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if printear:
                if line:
                    print(f"[drenado, descartado] {line}")
            ultima_llegada = time.time()


import threading

resultados = []
def logger():
    resultados.extend(grua.escuchar(tiempo_maximo=5))  # el ÚNICO lector del puerto

drenar_buffer(grua.ser)     # esto sí, antes de arrancar el hilo, sin conflicto

hilo = threading.Thread(target=logger)
hilo.start()

t0 = time.time()
target_finish = {"x": 30, "y": 0, "z": 40}
# grua.desp_lin_abs(**target_finish)   # solo ESCRIBE, no lee (devolveme_la_respuesta=False) -> no compite con el logger

# mover_directo(grua.ser, target_finish)

x = target_finish["x"] + grua.offset_x
y = target_finish["y"] + grua.offset_y
z = target_finish["z"] + grua.offset_z


grua.ser.write(("G90" + "\n").encode())
grua.ser.write((f"G1 Z{z} F3000" + "\n").encode())
grua.ser.write((f"G1 X{x} Y{y} F3000" + "\n").encode())

# Esto te manda tres ok's

t1 = time.time()


grua.ser.write(b"M400\n")            # solo escribe, no lee -> tampoco compite
# Esto te manda un ok mas
# luego el logger "muere" unos segundo y aparece devuelta diciendo "wait"
grua.ser.write(b"M114\n")  
t2 = time.time()
 # t2-t0 y la resta de tiempos entre el ok del M400 y el primer wait no son iguales


hilo.join()

print(f"desp_lin_abs volvió en {t1-t0:.2f}s")
print(f"\nM400 se mandó en t={t2-t0:.2f}s desde el inicio del movimiento")
print("\nRevisá en 'resultados' el primer 'ok' que aparece DESPUÉS de ese instante — ese es el de M400.")



# %%
