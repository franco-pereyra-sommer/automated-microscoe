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


import re
import time
import matplotlib.pyplot as plt



#%%



grua = TrimakerCOSMOS("COM11")

grua.escuchar(2)
time.sleep(1)

grua.home(devolveme_timepos=True)

grua.escuchar(2)
time.sleep(1)

# print("olo")



#%% ABS



grua.desp_lin_abs(50,0,40, devolveme_tiempos=True)

print("Termino la ida")

grua.desp_lin_abs(0,0,40, devolveme_tiempos=True)

print("Termino la vuelta")



#%% REL

# def drenar_buffer(ser, silencio=0.3, printear = False):
#     """Lee y descarta todo hasta que no llegue nada nuevo durante 'silencio' segundos."""
#     ultima_llegada = time.time()
#     while time.time() - ultima_llegada < silencio:
#         if ser.in_waiting:
#             line = ser.readline().decode(errors="ignore").strip()
#             if printear:
#                 if line:
#                     print(f"[drenado, descartado] {line}")
#             ultima_llegada = time.time()

# drenar_buffer(grua.ser)

grua.desp_lin_rel(50,0,10, devolveme_tiempos=True)

print("Termino el movimiento")

grua.desp_lin_rel(-50,0,-10, devolveme_tiempos=True)




#%% 



def es_linea_de_posicion(line):
    """True si la línea tiene el formato 'X:.. Y:.. Z:.. E:..' que manda M114."""
    return bool(re.search(r"X:-?\d+\.?\d*\s+Y:-?\d+\.?\d*\s+Z:-?\d+\.?\d*", line))


def esperar_fin_movimiento(ser, timeout=15):
    """
    Manda M400 + M114 y bloquea hasta ver la línea de posición.
    Devuelve el tiempo (en segundos) que tardó en aparecer esa línea,
    o None si se pasó del timeout.
    """
    t0 = time.time()
    # ser.write(b"M400\n")
    # ser.write(b"M114\n")

    while time.time() - t0 < timeout:
        if ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if es_linea_de_posicion(line):
                return time.time() - t0

    return None



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



#%% NO ME SALE LA FUNCION HOME()



resultados = []
def logger():
    resultados.extend(grua.escuchar(tiempo_maximo=15))  # el ÚNICO lector del puerto

grua = TrimakerCOSMOS("COM11")

drenar_buffer(grua.ser)     # esto sí, antes de arrancar el hilo, sin conflicto

# hilo = threading.Thread(target=logger)
# hilo.start()

grua.ser.write(b"G28\n")
grua.ser.write(b"M400\n")
# grua.ser.write(b"M114\n")

tiempo_home = esperar_fin_movimiento(grua.ser)

grua.ser.write(b"G1 Z40 F3000\n")

grua.ser.write(b"M400\n")
grua.ser.write(b"M114\n")

tiempo_bajar = esperar_fin_movimiento(grua.ser)

# hilo.join()

# print(f"{tiempo_home = }")

# print(f"{tiempo_bajar = }")

grua.desconectar()



#%% AHORA NO ME SALE LA FUNCION DE desp_lin_rel

x = 50
z = 10


resultados = []
def logger():
    resultados.extend(grua.escuchar(tiempo_maximo=15))  # el ÚNICO lector del puerto

# grua = TrimakerCOSMOS("COM11") 

drenar_buffer(grua.ser)     # esto sí, antes de arrancar el hilo, sin conflicto

# hilo = threading.Thread(target=logger)
# hilo.start()

grua.ser.write(b"G91\n")
# grua.ser.write(b"M400\n")
# grua.ser.write(b"M114\n")


grua.ser.write(f"G1 X{x} F3000\n".encode())

grua.ser.write(b"M400\n")
grua.ser.write(b"M114\n")

# tiempo_xy = esperar_fin_movimiento(grua.ser)
tiempo_xy = grua.esperar_fin_movimiento()

grua.ser.write(f"G1 Z{z} F3000\n".encode())

grua.ser.write(b"M400\n")
grua.ser.write(b"M114\n")

# tiempo_z = esperar_fin_movimiento(grua.ser)
tiempo_z = grua.esperar_fin_movimiento()

# hilo.join()

print(f"{tiempo_xy = }")
print(f"{tiempo_z = }")

# grua.desconectar()

grua.ser.write(f"G1 X{-x} F3000\n".encode())
grua.ser.write(f"G1 Z{-z} F3000\n".encode())



#%%



grua = TrimakerCOSMOS("COM11")

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
    resultados.extend(grua.escuchar(tiempo_maximo=10))  # el ÚNICO lector del puerto

drenar_buffer(grua.ser)     # esto sí, antes de arrancar el hilo, sin conflicto

hilo = threading.Thread(target=logger)
hilo.start()

tiempo_home = grua.home(devolveme_la_respuesta=False, devolveme_tiempo=True)
# print("olo")

# print(f"{tiempo_home = }")

# grua.desp_lin_abs(0,0,40,devolveme_tiempos=False)

# print(f"{grua.pregunta_posicion_actual() = }")

hilo.join()


#%%



cam = OpenMVCamera("COM7")

cam.conectar()



#%%



def es_linea_de_posicion(line):
    """True si la línea tiene el formato 'X:.. Y:.. Z:.. E:..' que manda M114."""
    return bool(re.search(r"X:-?\d+\.?\d*\s+Y:-?\d+\.?\d*\s+Z:-?\d+\.?\d*", line))


def esperar_fin_movimiento(ser, timeout=15):
    """
    Manda M400 + M114 y bloquea hasta ver la línea de posición.
    Devuelve el tiempo (en segundos) que tardó en aparecer esa línea,
    o None si se pasó del timeout.
    """
    t0 = time.time()
    # ser.write(b"M400\n")
    # ser.write(b"M114\n")

    while time.time() - t0 < timeout:
        if ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()
            if es_linea_de_posicion(line):
                return time.time() - t0

    return None


def medir_tiempos_vs_x(grua, x_rango, y=0, z=40, feedrate=3000, x_inicio=0, timeout=15):
    """
    Para cada x en x_rango:
      1. Vuelve a x_inicio (misma distancia recorrida en cada medición).
      2. Manda G90 + G1 Z + G1 XY hacia el x objetivo.
      3. Mide cuánto tarda en aparecer la confirmación de fin de movimiento (M400+M114).
    Devuelve una lista de tuplas (x, tiempo_seg).
    """
    resultados = []

    for x in x_rango:

        # --- Volver al punto de partida, esperando la confirmación real ---
        x_off_ini = x_inicio + grua.offset_x
        y_off_ini = y + grua.offset_y
        z_off_ini = z + grua.offset_z

        grua.ser.write(b"G90\n")
        grua.ser.write(f"G1 Z{z_off_ini} F{feedrate}\n".encode())
        grua.ser.write(f"G1 X{x_off_ini} Y{y_off_ini} F{feedrate}\n".encode())

        grua.ser.write(b"M400\n")
        grua.ser.write(b"M114\n")
        
        esperar_fin_movimiento(grua.ser, timeout=timeout)

        time.sleep(0.5)

        # --- Medición real: x_inicio -> x ---
        x_off = x + grua.offset_x
        y_off = y + grua.offset_y
        z_off = z + grua.offset_z

        # t0 = time.time()
        grua.ser.write(b"G90\n")
        grua.ser.write(f"G1 Z{z_off} F{feedrate}\n".encode())
        grua.ser.write(f"G1 X{x_off} Y{y_off} F{feedrate}\n".encode())

        grua.ser.write(b"M400\n")
        grua.ser.write(b"M114\n")


        tiempo = esperar_fin_movimiento(grua.ser, timeout=timeout)
        print("ORDEN DE FINALIZADO\n")

        if tiempo is None:
            print(f"[AVISO] timeout esperando movimiento a x={x}, se salta este punto")
            continue

        resultados.append((x, tiempo))
        print(f"x={x:>6} -> {tiempo:.3f}s")

    return resultados


def graficar_resultados(resultados):
    xs = [r[0] for r in resultados]
    ts = [r[1] for r in resultados]

    plt.figure(figsize=(7, 5))
    plt.plot(xs, ts, "o-")
    plt.xlabel("x destino (mm)")
    plt.ylabel("Tiempo hasta fin de movimiento (s)")
    plt.title(f"Tiempo de movimiento vs X (desde x={0})")
    plt.grid(True)
    plt.show()



#%%



time.sleep(4)
x_rango = np.arange(0.1, 100, 10)  # o np.arange si preferís pasos no enteros
x_rango = np.linspace(0.0125, 1, 30)
# x_rango = np.array([90.1])

resultados = medir_tiempos_vs_x(grua, x_rango, x_inicio=0)
graficar_resultados(resultados)



