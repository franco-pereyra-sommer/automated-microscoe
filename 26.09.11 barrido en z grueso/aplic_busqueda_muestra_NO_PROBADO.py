#%%


import time

import cv2
import numpy as np

from metricas_enfoque import (
    varianza_intensidad,
    bytes_a_imagen,
    autofocus,
)


# ============================================================
# EXCEPCIONES PROPIAS
# ============================================================

class MuestraNoEncontrada(Exception):
    pass


class RiesgoDeColision(Exception):
    pass


# ============================================================
# FRACCION DE PIXELES ROJOS (UTILIDAD SECUNDARIA / DIAGNÓSTICO)
# ============================================================
# NO se usa por defecto para decidir si hay muestra: con los datos
# reales calibrados, la varianza de intensidad separa mejor que el
# rojo (ver calibrar_umbral_rojo.py). Se deja disponible por si más
# adelante hace falta, por ejemplo, para diferenciar tipo 2 de tipo 3.
#
# Solo tiene sentido si la imagen fue capturada con color real
# (pixformat RGB565 o JPEG a color). En GRAYSCALE no aporta nada.

def fraccion_pixeles_rojos(imagen_bgr, s_min=50, v_min=50, v_max=255):

    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)

    rango_bajo_1 = np.array([0, s_min, v_min])
    rango_alto_1 = np.array([10, 255, v_max])

    rango_bajo_2 = np.array([160, s_min, v_min])
    rango_alto_2 = np.array([180, 255, v_max])

    mascara_1 = cv2.inRange(hsv, rango_bajo_1, rango_alto_1)
    mascara_2 = cv2.inRange(hsv, rango_bajo_2, rango_alto_2)

    mascara = cv2.bitwise_or(mascara_1, mascara_2)

    return mascara.mean() / 255.0


# ============================================================
# CLASIFICAR EL CAMPO VISUAL (¿HAY MUESTRA O NO?)
# ============================================================
# Calibrado con datos reales (n = 63 tipo 1, n = 71 tipo 2, n = 11 tipo 3):
#   tipo 1 (sin muestra / demasiado lejos del plano ideal):
#       varianza_intensidad entre 10.42 y 11.82
#   tipo 2 y tipo 3 (con muestra, nubes rojas, sin importar
#   qué tan cerca del plano ideal):
#       varianza_intensidad entre 18.18 y 117.73
# Separación limpia, sin superposición. Umbral por defecto en el
# punto medio (15.0), con margen simétrico a ambos lados.

def clasificar_campo_visual(imagen, umbral_varianza=15.0):

    varianza = varianza_intensidad(imagen)

    tipo = "nubes" if varianza >= umbral_varianza else "fondo_blanco"

    return {
        "tipo": tipo,
        "varianza": varianza,
    }


# ============================================================
# BARRIDO GRUESO DESCENDENTE EN Z (BUSCA LA MUESTRA)
# ============================================================

def buscar_muestra_en_z(
    camara,
    impresora,
    z_arriba_seguro,
    z_min_seguridad,
    paso_z_grueso=0.05,
    resolucion="VGA",
    umbral_varianza=15.0,
    confirmaciones_necesarias=2,
    espera_movimiento=0.3,
    devolveme_la_respuesta=False,
):
    """
    Desciende en Z, de a pasos de tamaño 'paso_z_grueso' (mm), desde
    'z_arriba_seguro' hasta 'z_min_seguridad', buscando el primer
    campo visual tipo "nubes" (varianza_intensidad >= umbral_varianza).

    Supuestos de seguridad (¡no tocar sin revisar el hardware!):
    - Z creciente = alejarse de la muestra (más seguro).
    - Z decreciente = acercarse a la muestra (más riesgo de choque).
    - 'z_min_seguridad' ya tiene que venir con el margen de seguridad
      incorporado respecto a la distancia de trabajo del objetivo
      (300 µm). Esta función NUNCA manda una posición menor a ese valor.

    Frena en cuanto detecta 'confirmaciones_necesarias' frames seguidos
    tipo "nubes" (para no pasarse de largo del entorno del plano ideal).

    No hace foco fino. Devuelve una Z "cerca" de la muestra para que
    después el autofocus() de metricas_enfoque.py la afine.
    """

    if z_min_seguridad < z_arriba_seguro:
        raise ValueError(
            "z_min_seguridad tiene que ser menor que z_arriba_seguro"
        )

    if paso_z_grueso <= 0:
        raise ValueError(
            "paso_z_grueso tiene que ser positivo "
            "(la función arma el signo del descenso sola)"
        )

    _, _, z_actual = impresora.pregunta_posicion_actual()

    if z_actual is None:
        raise RuntimeError(
            "La posición de la impresora no está inicializada. "
            "Llamá primero a impresora.home()"
        )

    if z_actual < z_min_seguridad:
        raise RiesgoDeColision(
            f"La posición actual en Z ({z_actual} mm) ya está por debajo "
            f"del límite de seguridad ({z_min_seguridad} mm). "
            f"Se aborta antes de mover nada."
        )

    # --------------------------------------------------------
    # Subir primero a la altura segura
    # --------------------------------------------------------

    desplazamiento_a_seguro = z_arriba_seguro - z_actual

    if desplazamiento_a_seguro != 0:
        impresora.desp_lin_rel(
            0, 0, desplazamiento_a_seguro,
            devolveme_la_respuesta=devolveme_la_respuesta,
        )
        time.sleep(espera_movimiento)

    # --------------------------------------------------------
    # Configurar cámara (mismo formato que usa el foco fino)
    # --------------------------------------------------------

    camara.set_pixformat("GRAYSCALE")
    camara.set_framesize(resolucion)

    # --------------------------------------------------------
    # Barrido descendente
    # --------------------------------------------------------

    posiciones_z = []
    clasificaciones = []

    contador_confirmaciones = 0
    z_encontrado = None

    z = z_arriba_seguro

    while True:

        image_data = camara.capturar()

        imagen = bytes_a_imagen(image_data)

        resultado = clasificar_campo_visual(
            imagen, umbral_varianza=umbral_varianza
        )

        posiciones_z.append(impresora.posicion_z)
        clasificaciones.append(resultado)

        print(
            f"Z = {impresora.posicion_z:.4f} mm | "
            f"tipo = {resultado['tipo']} | "
            f"varianza = {resultado['varianza']:.2f}"
        )

        # ----------------------------------------------------
        # Confirmación de detección
        # ----------------------------------------------------

        if resultado["tipo"] == "nubes":
            contador_confirmaciones += 1
        else:
            contador_confirmaciones = 0

        if contador_confirmaciones >= confirmaciones_necesarias:
            z_encontrado = impresora.posicion_z
            break

        # ----------------------------------------------------
        # ¿Se acabó el rango de seguridad sin encontrar nada?
        # ----------------------------------------------------

        if z <= z_min_seguridad:
            break

        # ----------------------------------------------------
        # Siguiente paso, sin pasarse del límite de seguridad
        # ----------------------------------------------------

        paso = -min(paso_z_grueso, z - z_min_seguridad)

        impresora.desp_lin_rel(
            0, 0, paso, devolveme_la_respuesta=devolveme_la_respuesta
        )
        time.sleep(espera_movimiento)

        z = impresora.posicion_z

    if z_encontrado is None:
        raise MuestraNoEncontrada(
            f"No se detectó la muestra entre Z = {z_arriba_seguro} mm y "
            f"Z = {z_min_seguridad} mm en esta posición (x, y)."
        )

    return z_encontrado, posiciones_z, clasificaciones


# ============================================================
# ORQUESTADOR: (x, y) -> MUESTRA ENCONTRADA (+ FOCO FINO OPCIONAL)
# ============================================================

def posicionar_en_muestra(
    camara,
    impresora,
    x,
    y,
    z_arriba_seguro,
    z_min_seguridad,
    paso_z_grueso=0.05,
    resolucion="VGA",
    umbral_varianza=15.0,
    confirmaciones_necesarias=2,
    feedrate_xy=3000,
    espera_movimiento=0.3,
    hacer_foco_fino=True,
    metrica_foco_fino=varianza_intensidad,
    pasos_arriba_foco_fino=10,
    pasos_abajo_foco_fino=10,
    paso_z_foco_fino=0.0025,
    devolveme_la_respuesta=False,
):
    """
    Automatiza el flujo completo para una posición (x, y):

    1) Sube a una altura segura ANTES de moverse en XY (para no
       arrastrar el objetivo por encima de otra parte de la muestra
       a una altura riesgosa).
    2) Se mueve en el plano XY a (x, y), todavía en altura segura.
    3) Barrido grueso descendente en Z buscando el campo visual
       tipo "nubes" (buscar_muestra_en_z).
    4) (Opcional) corre el autofocus() fino de metricas_enfoque.py
       alrededor de la Z encontrada, respetando el mismo
       z_min_seguridad. Como ambas etapas usan GRAYSCALE, no hace
       falta cambiar el pixformat entre una y otra.

    Devuelve un diccionario con la Z encontrada en el barrido grueso,
    la Z de foco fino (si se pidió) y las series de datos de ambos
    barridos, por si después querés graficarlas o auditarlas.
    """

    # --------------------------------------------------------
    # 1) Altura segura antes de moverse en XY
    # --------------------------------------------------------

    _, _, z_actual = impresora.pregunta_posicion_actual()

    if z_actual is None:
        raise RuntimeError("Llamá primero a impresora.home()")

    if z_actual < z_arriba_seguro:
        impresora.desp_lin_rel(
            0, 0, z_arriba_seguro - z_actual,
            devolveme_la_respuesta=devolveme_la_respuesta,
        )
        time.sleep(espera_movimiento)

    # --------------------------------------------------------
    # 2) Movimiento en XY, en altura segura
    # --------------------------------------------------------

    impresora.desp_lin_abs(
        x=x, y=y, z=z_arriba_seguro,
        feedrate=feedrate_xy,
        devolveme_la_respuesta=devolveme_la_respuesta,
    )
    time.sleep(espera_movimiento)

    # --------------------------------------------------------
    # 3) Barrido grueso
    # --------------------------------------------------------

    z_muestra, posiciones_z, clasificaciones = buscar_muestra_en_z(
        camara=camara,
        impresora=impresora,
        z_arriba_seguro=z_arriba_seguro,
        z_min_seguridad=z_min_seguridad,
        paso_z_grueso=paso_z_grueso,
        resolucion=resolucion,
        umbral_varianza=umbral_varianza,
        confirmaciones_necesarias=confirmaciones_necesarias,
        espera_movimiento=espera_movimiento,
        devolveme_la_respuesta=devolveme_la_respuesta,
    )

    resultado = {
        "x": x,
        "y": y,
        "z_muestra_encontrada": z_muestra,
        "posiciones_z_barrido": posiciones_z,
        "clasificaciones_barrido": clasificaciones,
        "z_foco_fino": None,
        "posiciones_z_foco_fino": None,
        "valores_metrica_foco_fino": None,
    }

    # --------------------------------------------------------
    # 4) Foco fino opcional
    # --------------------------------------------------------

    if hacer_foco_fino:

        # No dejar que el foco fino baje del límite de seguridad
        margen_disponible_abajo = z_muestra - z_min_seguridad

        pasos_abajo_reales = min(
            pasos_abajo_foco_fino,
            int(margen_disponible_abajo // paso_z_foco_fino),
        )

        pasos_abajo_reales = max(pasos_abajo_reales, 0)

        mejor_z, posiciones_z_fino, valores_metrica_fino = autofocus(
            camara=camara,
            impresora=impresora,
            metrica=metrica_foco_fino,
            pasos_arriba=pasos_arriba_foco_fino,
            pasos_abajo=pasos_abajo_reales,
            paso_z=paso_z_foco_fino,
            espera_movimiento=espera_movimiento,
            graficamos=False,
        )

        resultado["z_foco_fino"] = mejor_z
        resultado["posiciones_z_foco_fino"] = posiciones_z_fino
        resultado["valores_metrica_foco_fino"] = valores_metrica_fino

    return resultado




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

from Main import (OpenMVCamera,TrimakerCOSMOS)

from metricas_enfoque import (tenengrad, varianza_intensidad)



# #%%



grua = TrimakerCOSMOS("COM11")

grua.home()



# #%%



cam = OpenMVCamera("COM7")

cam.conectar()



# #%%


resultado = posicionar_en_muestra(
    camara=cam,
    impresora=grua,
    x=150.0,
    y=150.0,
    z_arriba_seguro=30.0,       # altura conocida, lejos de cualquier muestra
    z_min_seguridad=42.0,       # calculado a mano con margen bajo las 300 um de WD
    paso_z_grueso=0.05,
    hacer_foco_fino=True,
    pasos_arriba_foco_fino=20,
    pasos_abajo_foco_fino=20,
    metrica_foco_fino=tenengrad
)

print(resultado["z_muestra_encontrada"])
print(resultado["z_foco_fino"])