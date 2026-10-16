#%%
# Barrido en busca de puntos violetas con la impresora y la cámara REALES.
# Se corre celda por celda (como tus otros scripts). Antes de la celda 5, mirá lo que detecta (celda 3).

import os
import sys
from datetime import datetime
from pathlib import Path

RAIZ = r"C:\BackUp\Facultad\Labo 7\env_labo7"
CARPETA_CODIGO = RAIZ + r"\26.10.16 detector de puntos violetas y sus fotos"

os.environ["PYTHONPATH"] = RAIZ
for ruta in (RAIZ, CARPETA_CODIGO):
    if ruta not in sys.path:
        sys.path.insert(0, ruta)

import cv2
import matplotlib.pyplot as plt

from busqueda_muestra import posicionar_en_muestra
from camara_openmv import OpenMVCamera
from impresora_cosmos import TrimakerCOSMOS

from adaptador_hardware import MicroscopioCosmos
from barrido_puntos import ConfigBarrido, barrer, campos_espiral
from centrado import Calibracion, RegistroVisitados, calibrar_escala
from deteccion_puntos import DetectorColor, ParametrosColor, dibujar_detecciones


#%% PARÁMETROS (ajustar)

CARPETA_SALIDA = r"C:\BackUp\Facultad\Labo 7\Fotos\26.10.16 puntos violetas"

X_INI, Y_INI = 150, 153                         # donde arranca el barrido
Z_ARRIBA_SEGURO, Z_MIN_SEGURIDAD = 27.9, 27.5   # los mismos que en posicionar_en_muestra()

# Campo visual medido con las fotos 4K (x = filas, y = columnas). Acá se supone que QVGA ve lo mismo;
# la celda 4 lo mide de verdad.
FOV_X_MM, FOV_Y_MM = 0.214, 0.286
ANCHO_BUSQUEDA, ALTO_BUSQUEDA = 320, 240        # QVGA

# Umbrales de color: los de fábrica son provisorios. Reemplazar por la salida de calibrar_detector.py
PARAMETROS_COLOR = ParametrosColor()

# Cámara en la búsqueda: FIJAR exposición, ganancia y balance de blancos (con balance automático el color
# cambia de un campo a otro y los umbrales de color dejan de valer). Ej.:
#   dict(exposure_us=18000, gain_db=35, whitebal=[66.22236, 61.43028, 60.206])
KWARGS_CAMPO = dict()


#%% 1) CONEXIÓN Y POSICIONAMIENTO EN LA MUESTRA

grua = TrimakerCOSMOS("COM11")
grua.home()

cam = OpenMVCamera("COM7")
cam.conectar()

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
    paso_grueso=0.005,
    paso_fino=0.0025,
    hacer_foco_final=False,
    espera_movimiento=0.3,
)

print(f"{grua.pregunta_posicion_actual() = }")


#%% 2) EL MICROSCOPIO (adaptador) Y LA CALIBRACIÓN

micro = MicroscopioCosmos(
    cam,
    grua,
    carpeta_fotos=CARPETA_SALIDA,
    z_arriba_seguro=Z_ARRIBA_SEGURO,
    z_min_seguridad=Z_MIN_SEGURIDAD,
    kwargs_campo=KWARGS_CAMPO,
)

cal = Calibracion.desde_fov_mm(ANCHO_BUSQUEDA, ALTO_BUSQUEDA, FOV_X_MM, FOV_Y_MM)
print(cal)


#%% 3) VER QUÉ DETECTA EN EL CAMPO ACTUAL (no mueve la platina, solo reenfoca)

micro.enfocar()
imagen = micro.capturar_campo()

detector = DetectorColor(PARAMETROS_COLOR)
detecciones = detector.detectar(imagen)

for d in detecciones:
    print(f"centro (col={d.col:.1f}, fila={d.fila:.1f})  diámetro {d.diametro_px:.1f} px  "
          f"área {d.area_px:.0f} px²  toca el borde: {d.toca_borde}")

plt.imshow(cv2.cvtColor(dibujar_detecciones(imagen, detecciones), cv2.COLOR_BGR2RGB))
plt.title(f"{len(detecciones)} detección(es); cruz blanca = centro de la imagen")
plt.show()

# Para calibrar los umbrales con calibrar_detector.py: guardar campos CON punto en una carpeta y SIN punto en otra.
# (cambiar 'con_punto' por 'sin_punto' según lo que se ve en este campo)
carpeta_calibracion = Path(CARPETA_SALIDA) / "calibracion" / "con_punto"
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

x0, y0 = micro.posicion_mm()
campos = list(campos_espiral(x0, y0, FOV_X_MM / 2, FOV_Y_MM / 2, n_radial=1))

config = ConfigBarrido(carpeta_salida=CARPETA_SALIDA)

# Para retomar un barrido interrumpido, cargar el registro que se va guardando tras cada foto:
# registro = RegistroVisitados.cargar(Path(CARPETA_SALIDA) / "registro_visitados.json")
registro = None

resultado_barrido, registro = barrer(micro, detector, cal, campos, config, registro=registro)

print(resultado_barrido)
print(f"Autofocos sin pico confirmado: {micro.autofocos_fallidos}")


#%%
