#%%
"""
Medición de aumentos y parámetros de cámara -- barre en Z con cada combinación de exposición y ganancia y guarda las fotos.
Carpeta: 26.11.09 medicion de aumentos y parametros de camara          Inicio: 2026-11-09
Objetivo: elegir los aumentos ("x", uno por métrica) de buscar_foco_en_xy() y ver qué parámetros de cámara
    ayudan más a encontrar el plano de enfoque. Se mide (esta celda) y después se analiza (correr_analisis.py).
Enfoque: por cada (x, y) y cada combinación de exposición y ganancia, un descenso completo de Z_ARRIBA_SEGURO a
    Z_MIN_SEGURIDAD, con la cámara configurada UNA vez (con None el sensor se acomoda mientras baja, como en la
    búsqueda real), guardando cada foto y lo que informó la cámara. Lo único que queda fijo es el sistema óptico.
    Entre corridas se cambian a mano la muestra, el (x, y) y, si hace falta, la Z.
Módulos base que usa: camara_openmv (OpenMVCamera, validar_dict_params_camara), impresora_cosmos (TrimakerCOSMOS,
    moverse_a_xy), busqueda_muestra (RiesgoDeColision), metricas_imagen (en el análisis).
Propuestas para la base: ninguna por ahora.
Estado: NO PROBADO        Resultado: (completar después de medir)

Se corre celda por celda. La celda 1 (conexión) una sola vez; la 3, una vez por cada (x, y) y muestra.
"""

import os
import sys
from pathlib import Path

RAIZ = r"C:\BackUp\Facultad\Labo 7\env_labo7"
CARPETA_CODIGO = RAIZ + r"\26.11.09 medicion de aumentos y parametros de camara"

os.environ["PYTHONPATH"] = RAIZ
for ruta in (RAIZ, CARPETA_CODIGO):
    if ruta not in sys.path:
        sys.path.insert(0, ruta)

from camara_openmv import OpenMVCamera, validar_dict_params_camara
from impresora_cosmos import TrimakerCOSMOS

from barrido_con_fotos import barrido_con_fotos, nombre_de_carpeta_de_parametros


#%% 1) CONEXIÓN

grua = TrimakerCOSMOS("COM11")
grua.home()

cam = OpenMVCamera("COM7")
cam.conectar()


#%% 2) PARÁMETROS (los que cambiás entre corridas)

CARPETA_SALIDA = r"C:\BackUp\Facultad\Labo 7\Fotos\26.11.09 medicion de aumentos y parametros de camara"

MUESTRA = "muestra 1"           # cambiar al cambiar de muestra (va en el nombre de la carpeta)
PASADA = 1                      # cambiar en cada corrida del mismo (x, y): no se mezclan fotos de corridas distintas

X_MM, Y_MM = 150, 150           # el campo visual de esta corrida

Z_ARRIBA_SEGURO = 27.9          # LEJOS del foco: de acá sale la línea de base (los de 26.09.11 para x=150, y=150: ajustar)
Z_MIN_SEGURIDAD = 27.6          # el barrido nunca baja de acá (distancia de trabajo del objetivo: 300 µm)
PASO_Z_MM = 0.0025              # una foto cada 2.5 µm; el análisis puede saltear fotos para emular un paso de 5 µm

PIXFORMAT = "RGB565"            # constantes de toda la medición
RESOLUCION = "QVGA"
WHITEBAL = (66.22236, 61.43028, 60.206)     # (rojo, verde, azul), fijo; None = automático (el de 26.09.11)

EXPOSICIONES_US = [None, 15000, 50000]      # None = exposición automática
GANANCIAS_DB = [None, 15, 35]               # None = ganancia automática

ESPERA_MOVIMIENTO = 0.5         # segundos que se espera después de cada movimiento (vibraciones), como la búsqueda


def armar_dict_params(exposure_us, gain_db):
    return {
        "pixformat": PIXFORMAT,
        "resolucion": RESOLUCION,
        "exposure_us": exposure_us,
        "gain_db": gain_db,
        "whitebal": WHITEBAL,
    }


CARPETA_DE_ESTE_XY = Path(CARPETA_SALIDA) / f"{MUESTRA} xy {X_MM}_{Y_MM} - pasada {PASADA}"


def carpeta_del_barrido(exposure_us, gain_db):
    return CARPETA_DE_ESTE_XY / nombre_de_carpeta_de_parametros(armar_dict_params(exposure_us, gain_db))


#%% 3) MEDICIÓN: un barrido por cada combinación de exposición y ganancia

fotos_por_barrido = round((Z_ARRIBA_SEGURO - Z_MIN_SEGURIDAD) / PASO_Z_MM) + 1
n_barridos = len(EXPOSICIONES_US) * len(GANANCIAS_DB)

print(f"{n_barridos} barridos de unas {fotos_por_barrido} fotos en (x, y) = ({X_MM}, {Y_MM}), {MUESTRA}, pasada {PASADA}")

# Se revisa TODO antes de mover: diccionarios de cámara y que ninguna carpeta tenga ya fotos
for exposure_us in EXPOSICIONES_US:
    for gain_db in GANANCIAS_DB:

        validar_dict_params_camara(armar_dict_params(exposure_us, gain_db))

        carpeta = carpeta_del_barrido(exposure_us, gain_db)

        if carpeta.exists() and any(carpeta.glob("*.jpg")):
            raise FileExistsError(f"Ya hay fotos en {carpeta}: cambiá PASADA.")


for exposure_us in EXPOSICIONES_US:

    for gain_db in GANANCIAS_DB:

        print(f"\n=== Exposición {exposure_us} | Ganancia {gain_db} ===")

        barrido_con_fotos(
            cam,
            grua,
            X_MM,
            Y_MM,
            z_arriba_seguro=Z_ARRIBA_SEGURO,
            z_min_seguridad=Z_MIN_SEGURIDAD,
            dict_params_camara=armar_dict_params(exposure_us, gain_db),
            carpeta_guardado=carpeta_del_barrido(exposure_us, gain_db),
            paso_z=PASO_Z_MM,
            espera_movimiento=ESPERA_MOVIMIENTO,
        )

print("\nListo. Fotos en:", CARPETA_DE_ESTE_XY)


#%%
