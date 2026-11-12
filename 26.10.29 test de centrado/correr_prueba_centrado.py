#%%
# Prueba de centrado con la impresora y la cámara REALES (aplicación 26.10.29 test de centrado).
# Se corre celda por celda, como tus otros scripts. Qué hace y cómo se mide: ver el encabezado de prueba_centrado.py.
#
# Por cada prueba: foto -> píxel al azar -> foto con el píxel marcado -> centrar -> foto centrada -> medir dónde quedó.
# Las imágenes quedan en una subcarpeta con la hora de la corrida, dentro de CARPETA_SALIDA (no se pisan entre corridas).
# Para mirar: prueba_NN_4_comparacion.png (antes y después juntos).

import os
import sys
from datetime import datetime
from pathlib import Path

RAIZ = r"C:\BackUp\Facultad\Labo 7\env_labo7"
CARPETA_CODIGO = RAIZ + r"\26.10.29 test de centrado"

os.environ["PYTHONPATH"] = RAIZ
for ruta in (RAIZ, CARPETA_CODIGO):
    if ruta not in sys.path:
        sys.path.insert(0, ruta)

import numpy as np

from busqueda_muestra import posicionar_en_muestra, foco_final_multimetrica
from camara_openmv import OpenMVCamera
from impresora_cosmos import TrimakerCOSMOS

from prueba_centrado import crear_calibracion, probar_centrado, resumir_pruebas


#%% PARÁMETROS (ajustar)

CARPETA_SALIDA = r"C:\BackUp\Facultad\Labo 7\Fotos\26.11.12 test de centrado"

N_PRUEBAS = 3                   # cada una elige un píxel al azar y vuelve al campo inicial
SEMILLA = 43                  # un entero repite los mismos píxeles; None = distintos en cada corrida

X_INI, Y_INI = 151, 150        # donde se busca la muestra (celda 2)
Z_ARRIBA_SEGURO, Z_MIN_SEGURIDAD = 28, 27.5   # los mismos que en posicionar_en_muestra() de tus otras aplicaciones

FRAMES_A_DESCARTAR = 2          # cuadros que se tiran antes de cada foto, por si la cámara entrega uno viejo.
                                # Si el centrado falla, probar con 0 y comparar.
KWARGS_CAMARA = dict()          # vacío = exposición, ganancia y balance de blancos automáticos. Para fijarlos:
# KWARGS_CAMARA = dict(exposure_us=37966, gain_db=20.423784, whitebal=(65.20144, 61.05388, 60.206))
                                # dict(exposure_us=37966, gain_db=20.423784, whitebal=(65.20144, 61.05388, 60.206)) ("""Franco: Thank u, es muy practico este comentario""")


#%% 1) CONEXIÓN

grua = TrimakerCOSMOS("COM11")
grua.home()

cam = OpenMVCamera()            # puerto por defecto de la base (COM7)
cam.conectar()


#%% 2) POSICIONARSE EN LA MUESTRA Y ENFOCAR
# La prueba solo mueve decenas de µm: necesita la cámara ya sobre la muestra y enfocada. Si tu celda de siempre
# (con tus criterios_deteccion) anda, usala en lugar de esta. Con criterios_deteccion=None se usan los de la base
# (metricas_imagen.CRITERIOS_POR_DEFECTO, pendiente de recalibrar: BITACORA.txt, pendiente 7).

dict_params_camra = {
                "pixformat": "GRAYSCALE",            # o "RGB565" / "JPEG" (o PixFormat.<...>)
                "resolucion": "QVGA",                # o "WQXGA2" / "4K" / ... (o Framesize.<...>)
                "exposure_us": None, #50000,                # o None = exposición automática
                "gain_db": None, #25,                       # o None = ganancia automática
                "whitebal": None, #(66.22236, 61.43028, 60.206),   # (rojo, verde, azul) o None = automático
            }

resultado = posicionar_en_muestra(
    cam,
    grua,
    x=X_INI,
    y=Y_INI,
    z_arriba_seguro=Z_ARRIBA_SEGURO,
    z_min_seguridad=Z_MIN_SEGURIDAD,
    hacer_foco_final=True,
    dict_params_global=dict_params_camra,
    criterios_deteccion={
        "laplacian": 12,
        "tenengrad": 170,#165,#203,
        "varianza": 130,#235#100,#670,
    },
    paso_grueso=0.005,
)

print(f"{grua.pregunta_posicion_actual() = }")


#%% 3) LA PRUEBA

# calibracion = crear_calibracion()
# print(calibracion)


CARPETA_CENTRADO = Path(__file__).resolve().parent.parent / "26.10.16 detector de puntos violetas y sus fotos"
if str(CARPETA_CENTRADO) not in sys.path:
    sys.path.append(str(CARPETA_CENTRADO))

from centrado import Calibracion

calibracion = Calibracion.desde_fov_mm(320, 240, 1e-3*240*12.5/(168.759-51.290), 1e-3*320*12.5/(168.759-51.290))
print(calibracion)



#%%


carpeta_corrida = Path(CARPETA_SALIDA) / f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
generador = np.random.default_rng(SEMILLA)

resultados = []
for numero in range(1, N_PRUEBAS + 1):
    resultados.append(probar_centrado(
        cam,
        grua,
        calibracion,
        Z_ARRIBA_SEGURO,
        carpeta_corrida,
        numero=numero,
        generador=generador,
        # configuracion_camara=KWARGS_CAMARA,
        dic_param_cam=dict_params_camra,
        frames_a_descartar=FRAMES_A_DESCARTAR,
        max_correcciones = 1,
    ))

print()
print(resumir_pruebas(resultados))
print(f"Imágenes en: {carpeta_corrida}")


#%% 4) CERRAR

cam.desconectar()
grua.desconectar()


#%%
