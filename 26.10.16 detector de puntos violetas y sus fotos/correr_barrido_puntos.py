#%%
# Barrido en busca de puntos violetas con la impresora y la cámara REALES.
# Se corre celda por celda (como tus otros scripts). Antes de la celda 5, mirá lo que detecta (celda 3).

import os
import sys
from datetime import datetime
from pathlib import Path


import time

RAIZ = r"C:\BackUp\Facultad\Labo 7\env_labo7"
CARPETA_CODIGO = RAIZ + r"\26.10.22"

os.environ["PYTHONPATH"] = RAIZ
for ruta in (RAIZ, CARPETA_CODIGO):
    if ruta not in sys.path:
        sys.path.insert(0, ruta)

import cv2
import matplotlib.pyplot as plt

from busqueda_muestra import posicionar_en_muestra, foco_final_multimetrica, tenengrad, laplacian_variance
from camara_openmv import OpenMVCamera
from impresora_cosmos import TrimakerCOSMOS

from adaptador_hardware import MicroscopioCosmos
from barrido_puntos import ConfigBarrido, barrer, campos_espiral
from centrado import Calibracion, RegistroVisitados, calibrar_escala
# from deteccion_contraste import DetectorContraste, ParametrosContraste, cargar_mascara_fija
from deteccion_puntos import DetectorColor, ParametrosColor, dibujar_detecciones

from deteccion_contraste import DetectorContraste, ParametrosContraste, cargar_mascara_fija
# detector = DetectorContraste(ParametrosContraste(umbral=55, peso_azul=1.5),
                                # cargar_mascara_fija(r"C:\BackUp\Facultad\Labo 7\env_labo7\26.10.16 detector de puntos violetas y sus fotos\debug_detector\mascara_fija.png"))

#%% PARÁMETROS (ajustar)


CARPETA_SALIDA = r"C:\BackUp\Facultad\Labo 7\Fotos\26.10.22"

# X_INI, Y_INI = 150, 150                         # donde arranca el barrido
X_INI, Y_INI = 150.9625, 150 
Z_ARRIBA_SEGURO, Z_MIN_SEGURIDAD = 27.9, 27.4   # los mismos que en posicionar_en_muestra()

# Campo visual de QVGA con la lente nueva (x = filas de la imagen = 240 px, y = columnas = 320 px).
# Medido moviendo la platina una distancia conocida: ~0.388 µm/px tanto en filas como en columnas, o sea
#   x (alto, 240 px) = 93 µm = 0.09263 mm      y (ancho, 320 px) = 124 µm = 0.1243 mm
# Antes estaban al revés (0.1243, 0.09263): los movimientos de centrado se pasaban ~34 % en x y se quedaban ~25 %
# cortos en y, de ahí los centrados fallidos y las fotos repetidas. La celda 4 vuelve a medirlo.
FOV_X_MM, FOV_Y_MM = 0.09263, 0.1243
ANCHO_BUSQUEDA, ALTO_BUSQUEDA = 320, 240        # QVGA

# Detector de puntos: "contraste" (recomendado: más oscuro y más azul que el entorno, no depende del tono exacto)
# o "color" (rango HSV). Los valores salen de calibrar_detector.py (ver "Para usar:" al final de su salida).
DETECTOR = "contraste"
PARAMETROS_CONTRASTE = ParametrosContraste(umbral=55, peso_azul=1.5)

# Manchas fijas (polvo): la guarda calibrar_detector.py en la carpeta de --salida. Tiene que haberse calculado con
# fotos de ESTA lente y cámara. None = no ignorar nada.
# RUTA_MASCARA_FIJA = CARPETA_CODIGO + r"\debug_detector\mascara_fija.png"
RUTA_MASCARA_FIJA = r"C:\BackUp\Facultad\Labo 7\Fotos\26.10.20 nueva lente + barrido para determinar puntos violetas\calibracion\debug_detector mejor\mascara_fija.png"
# Umbrales de color (solo si DETECTOR = "color"): los de fábrica son provisorios.
PARAMETROS_COLOR = ParametrosColor()

# Cámara en la búsqueda: FIJAR exposición, ganancia y balance de blancos (con balance automático el color
# cambia de un campo a otro y los umbrales de color dejan de valer). Ej.:
#   dict(exposure_us=18000, gain_db=35, whitebal=[66.22236, 61.43028, 60.206])
KWARGS_CAMPO = dict(
    exposure_us = 37966,
    gain_db = 20.423784,
    whitebal = (65.20144, 61.05388, 60.206),
)


#%% 1) CONEXIÓN Y POSICIONAMIENTO EN LA MUESTRA

grua = TrimakerCOSMOS("COM11")
grua.home()

cam = OpenMVCamera("COM7")
cam.conectar()

# cam.mostrar(cam.capturar())


#%%


cam.config_de_una()

# Igual que en tu barrido espiral: sin foco final (el foco fino lo hace el adaptador en cada parada).
resultado = posicionar_en_muestra(
    cam,
    grua,
    x=X_INI,
    y=Y_INI,
    z_arriba_seguro=Z_ARRIBA_SEGURO,
    z_min_seguridad=Z_MIN_SEGURIDAD,
    resolucion_global="QVGA",
    pixformat_global="GRAYSCALE",
    criterios_deteccion={
    "laplacian": 12,
    "tenengrad": 203,
    "varianza": 165#235#100,#670,
},
    paso_grueso=0.005,
    paso_fino=0.0025,
    hacer_foco_final=False,
    espera_movimiento=0.5,
)

print(f"{grua.pregunta_posicion_actual() = }")


#%%


foco_final_multimetrica(
    cam,
    grua,
    z_min_seguridad=Z_MIN_SEGURIDAD,
    z_max_seguro=Z_ARRIBA_SEGURO,
    paso_z=0.0025,
    pasos_arriba=10,
    pasos_abajo=10,
    vecinos_para_confirmar=3,
    pasos_extension_maxima=5,
    espera_movimiento=0.5,
    resolucion_grayscale="QVGA",
    pixformat_color="RGB565",
    resolucion_color="WQXGA2", #WQXGA2 QVGA  # alias: "4K"
    guardamos=True,
    carpeta_guardado=CARPETA_SALIDA + rf"\fotos multimetrica\gl con QVGA 7",
    graficamos=True,
)


#%% 2) EL MICROSCOPIO (adaptador) Y LA CALIBRACIÓN

micro = MicroscopioCosmos(
    cam,
    grua,
    carpeta_fotos=CARPETA_SALIDA,
    z_arriba_seguro=Z_ARRIBA_SEGURO,
    z_min_seguridad=Z_MIN_SEGURIDAD,
    kwargs_campo=KWARGS_CAMPO,
    metrica_foco=laplacian_variance
)

cal = Calibracion.desde_fov_mm(ANCHO_BUSQUEDA, ALTO_BUSQUEDA, FOV_X_MM, FOV_Y_MM)
print(cal)


#%% 3) VER QUÉ DETECTA EN EL CAMPO ACTUAL (no mueve la platina, solo reenfoca)

# grua.desp_lin_rel(0.04, 0, 0)
# grua.desp_lin_rel(0, 0.055, 0)
# grua.desp_lin_rel(-0.04, 0, 0)
# time.sleep(0.6)

micro.enfocar()
imagen = micro.capturar_campo()

if DETECTOR == "contraste":
    mascara_fija = cargar_mascara_fija(RUTA_MASCARA_FIJA) if RUTA_MASCARA_FIJA else None
    detector = DetectorContraste(PARAMETROS_CONTRASTE, mascara_fija)
else:
    detector = DetectorColor(PARAMETROS_COLOR)
detecciones = detector.detectar(imagen) # Pone a la camara en RGB565 y QVGA

for d in detecciones:
    print(f"centro (col={d.col:.1f}, fila={d.fila:.1f})  diámetro {d.diametro_px:.1f} px  "
          f"área {d.area_px:.0f} px²  toca el borde: {d.toca_borde}")

plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.show()

plt.imshow(cv2.cvtColor(dibujar_detecciones(imagen, detecciones), cv2.COLOR_BGR2RGB))
plt.title(f"{len(detecciones)} detección(es); cruz blanca = centro de la imagen")
plt.show()

### ### ### ### ###
#
### ### ### ### ###
#
### ### ### ### ###


#%%


# Para calibrar los umbrales con calibrar_detector.py: guardar campos CON punto en una carpeta y SIN punto en otra.
# (cambiar 'con_punto' por 'sin_punto' según lo que se ve en este campo)
# carpeta_calibracion = Path(CARPETA_SALIDA) / "calibracion" / "con_punto"
carpeta_calibracion = Path(CARPETA_SALIDA) / "calibracion" / "sin_punto"
carpeta_calibracion.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(carpeta_calibracion / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.png"), imagen)


#%% 4) (OPCIONAL, recomendado la primera vez) MEDIR LA ESCALA REAL EN QVGA
# Mueve la platina 0.05 mm en x y 0.05 mm en y, y mide cuánto se corre la imagen. Hacerlo sobre glóbulos
# (necesita textura). Si da distinto de lo supuesto, usar la medida: cal = cal_medida

x_ahora, y_ahora = micro.posicion_mm()
cal_medida = calibrar_escala(micro, x_ahora, y_ahora, ANCHO_BUSQUEDA, ALTO_BUSQUEDA)

print(f"supuesta: {cal.um_por_px_fila:.3f} µm/px (filas), {cal.um_por_px_col:.3f} µm/px (columnas), "
      f"signos ({cal.signo_x:+d}, {cal.signo_y:+d})")
print(f"medida:   {cal_medida.um_por_px_fila:.3f} µm/px (filas), {cal_medida.um_por_px_col:.3f} µm/px (columnas), "
      f"signos ({cal_medida.signo_x:+d}, {cal_medida.signo_y:+d})")


#%% 5) BARRIDO
# Espiral con paso = mitad del campo visual (campos superpuestos). Empezar con n_radial=1 (9 campos).

# grua.desp_lin_abs(150.9625, 150, 27.58)
# time.sleep(2)

x0, y0 = micro.posicion_mm()
campos = list(campos_espiral(x0, y0, FOV_X_MM / 2, FOV_Y_MM / 2, n_radial=2))

config = ConfigBarrido(
    carpeta_salida=CARPETA_SALIDA + rf"\barrido de puntos violetas 2",
    tolerancia_repetidos_um=25.0,   # una parada a <= 25 µm (por eje) de otra ya fotografiada cuenta como el mismo punto
    guardar_imagenes=True,          # CARPETA_SALIDA/diagnostico/crudas y /marcadas (mismos nombres): para comparar
    # verificar=True, max_correcciones=2  <- ya son los valores por defecto: tras moverse, vuelve a detectar y corrige
)

# Para retomar un barrido interrumpido, cargar el registro que se va guardando tras cada foto:
# registro = RegistroVisitados.cargar(Path(CARPETA_SALIDA) / "registro_visitados.json")
registro = None
micro.invalidar_modo()
resultado_barrido, registro = barrer(micro, detector, cal, campos, config, registro=registro)

print(resultado_barrido)
print(f"Autofocos sin pico confirmado: {micro.autofocos_fallidos}")


#%%
