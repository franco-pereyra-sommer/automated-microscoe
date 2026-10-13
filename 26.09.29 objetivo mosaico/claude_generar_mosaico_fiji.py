#%%


# Agrego la raiz del proyecto
import sys
import os

os.environ["PYTHONPATH"] = r"C:\BackUp\Facultad\Labo 7\env_labo7"

sys.path.insert(0, os.environ["PYTHONPATH"])

import time
from pathlib import Path
from datetime import datetime

import numpy as np

# from metricas_enfoque import varianza_intensidad
# from busqueda_muestra import posicionar_en_muestra, FUNCIONES_METRICA
from busqueda_muestra import *

from Main import OpenMVCamera
from Main import TrimakerCOSMOS


from pathlib import Path
import subprocess
import re

from PIL import Image
import cv2

from metricas_enfoque import extraer_coordenadas
# from reconstruir_mosaico import _escalas_px_por_mm, _posicion_nominal_px


# =================
# EJE DE LA CÁMARA vs. EJE DE LA IMPRESORA
# =================
# Confirmado a mano (dos y tres tiles, sin rotar las fotos): el eje
# x de la impresora corresponde al eje de FILAS de la imagen (no al
# de columnas), y el eje y de la impresora corresponde al eje de
# COLUMNAS. Es decir, la cámara está montada ~90° rotada respecto de
# la platina. Esto es consistente con las medidas: el campo visual
# estimado es más chico en x (~100 µm) que en y (~140 µm), mientras
# que la foto es más ancha (columnas) que alta (filas) — si x fuera
# columnas, sería al revés.
#
# Si en algún momento se cambia el montaje de la cámara (o la
# óptica) y esto deja de cumplirse, hay que volver a confirmarlo con
# el mismo método (dos fotos vecinas en x, dos en y, mirar si
# encastran) y ajustar 'stage_x_son_filas' más abajo.


def _escalas_px_por_mm(ancho_px, alto_px, fov_x_mm, fov_y_mm, stage_x_son_filas=True):
    """
    'fov_x_mm'/'fov_y_mm' son el campo visual real (en mm) que cubre
    un tile en cada eje DE LA IMPRESORA (no de la imagen). Esta
    función devuelve (escala_filas, escala_columnas), en píxeles por
    mm, ya con el intercambio de ejes aplicado según
    'stage_x_son_filas'.
    """

    if stage_x_son_filas:
        escala_filas = alto_px / fov_x_mm
        escala_columnas = ancho_px / fov_y_mm
    else:
        escala_filas = alto_px / fov_y_mm
        escala_columnas = ancho_px / fov_x_mm

    return escala_filas, escala_columnas


def _posicion_nominal_px(x_mm, y_mm, escala_filas, escala_columnas,
                          invertir_filas, invertir_columnas,
                          stage_x_son_filas=True):
    """
    Posición (fila, columna) en píxeles, SIN normalizar a un origen
    todavía, de un punto (x_mm, y_mm) de la impresora.
    """

    if stage_x_son_filas:
        fila = x_mm * escala_filas
        columna = y_mm * escala_columnas
    else:
        fila = y_mm * escala_filas
        columna = x_mm * escala_columnas

    if invertir_filas:
        fila = -fila
    if invertir_columnas:
        columna = -columna

    return fila, columna


# ============================================================
# DIAGNÓSTICO DEL ArrayIndexOutOfBoundsException: 2
# ============================================================
# TranslationModel3D.applyInPlace() reventando en el índice 2 quiere
# decir que el modelo de transformación que se está usando es de 3
# dimensiones (x, y, z) pero algún punto que le pasaste solo tiene 2
# coordenadas (x, y) — le falta la posición [2], la z. Eso pasa
# típicamente por una de estas dos cosas en el TileConfiguration.txt:
#   1) puso "dim = 2" pero alguna línea tiene 3 números en vez de 2
#      (por ejemplo, si armaste el archivo a mano copiando el
#      "(x, y, z)" completo del nombre de archivo en vez de solo
#      (x, y)), o
#   2) el tipo de layout elegido en el diálogo no era 2D.
# La función de acá abajo escribe SIEMPRE "dim = 2" con exactamente
# 2 números por línea, para no pisar el mismo problema.


# ============================================================
# LISTAR TILES SIN CARGAR LOS PÍXELES (liviano)
# ============================================================
# A diferencia de cargar_tiles() de reconstruir_mosaico.py, esto NO
# decodifica la imagen completa — Fiji va a leer los archivos de
# imagen directamente del disco, así que en Python solo hace falta
# el nombre, la posición (x, y, z) y el tamaño en píxeles (que se
# puede sacar del header del archivo, sin decodificar el jpeg
# entero). Para un barrido de muchos tiles en alta resolución, cargar
# todos los píxeles en memoria solo para esto sería un desperdicio
# grande de RAM.

def listar_tiles_liviano(carpeta, extension="*.jpg"):

    carpeta = Path(carpeta)

    archivos = sorted(carpeta.glob(extension))

    tiles = []

    for archivo in archivos:

        try:
            x, y, z = extraer_coordenadas(archivo)
        except ValueError:
            print(f"Se salteó (sin coordenadas en el nombre): {archivo.name}")
            continue

        with Image.open(archivo) as img:
            ancho_px, alto_px = img.size  # PIL da (ancho, alto)

        tiles.append({
            "archivo": archivo,
            "x": x,
            "y": y,
            "z": z,
            "alto_px": alto_px,
            "ancho_px": ancho_px,
        })

    if len(tiles) == 0:
        raise ValueError(f"No se encontraron tiles válidos en {carpeta}")

    alto_0, ancho_0 = tiles[0]["alto_px"], tiles[0]["ancho_px"]
    for tile in tiles:
        if (tile["alto_px"], tile["ancho_px"]) != (alto_0, ancho_0):
            raise ValueError(
                f"{tile['archivo'].name} mide {tile['ancho_px']}x{tile['alto_px']}, "
                f"distinto de los demás ({ancho_0}x{alto_0}) — ¿se mezclaron fotos "
                f"de resoluciones distintas?"
            )

    return tiles


# ============================================================
# GENERAR TileConfiguration.txt
# ============================================================

def generar_tile_configuration(
    tiles,
    fov_x_mm,
    fov_y_mm,
    ruta_salida,
    stage_x_son_filas=True,
    invertir_filas=False,
    invertir_columnas=False,
):
    """
    Escribe un TileConfiguration.txt en el formato que espera
    "Grid/Collection stitching" con type=[Positions from file]:
    dim = 2, y una línea por tile con
    "nombre_de_archivo.jpg; ; (columna_px, fila_px)".

    Las posiciones se calculan igual que en reconstruir_mosaico.py
    (mismo intercambio de ejes cámara/impresora, mismas
    fov_x_mm/fov_y_mm/stage_x_son_filas/invertir_filas/
    invertir_columnas) — así que si ya calibraste esos valores para
    tu propio mosaico, son los mismos acá. Fiji espera las
    coordenadas como (x, y) DE LA IMAGEN, es decir (columna, fila),
    no (fila, columna).

    'ruta_salida' tiene que quedar en la MISMA carpeta que las fotos
    (el 'directory' que se le pasa al plugin es una sola carpeta,
    tanto para las imágenes como para este archivo).
    """

    alto_px, ancho_px = tiles[0]["alto_px"], tiles[0]["ancho_px"]

    escala_filas, escala_columnas = _escalas_px_por_mm(
        ancho_px, alto_px, fov_x_mm, fov_y_mm, stage_x_son_filas
    )

    posiciones = [
        _posicion_nominal_px(
            tile["x"], tile["y"], escala_filas, escala_columnas,
            invertir_filas, invertir_columnas, stage_x_son_filas,
        )
        for tile in tiles
    ]

    # Se normaliza al mínimo de cada eje: las coordenadas absolutas
    # (con el x-y real en mm de la impresora, capaz ~150 mm) darían
    # valores de cientos de miles de píxeles, sin ningún problema
    # para Fiji (usa diferencias relativas), pero así el archivo
    # queda más legible si alguna vez hay que revisarlo a mano.
    fila_min = min(pos[0] for pos in posiciones)
    col_min = min(pos[1] for pos in posiciones)

    lineas = [
        "# Define the number of dimensions we are working on",
        "dim = 2",
        "",
        "# Define the image coordinates",
    ]

    for tile, (fila, columna) in zip(tiles, posiciones):

        nombre = Path(tile["archivo"]).name

        # Fiji: (x, y) = (columna, fila), en píxeles, dim=2 -> 2 números.
        lineas.append(
            f"{nombre}; ; ({columna - col_min:.4f}, {fila - fila_min:.4f})"
        )

    ruta_salida = Path(ruta_salida)
    ruta_salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    print(f"TileConfiguration.txt escrito en {ruta_salida} ({len(tiles)} tiles)")

    return ruta_salida


# ============================================================
# CORRER FIJI EN MODO HEADLESS
# ============================================================
# OJO: esto no lo pude probar de punta a punta (no tengo Fiji acá
# para correrlo) — sí probé que generar_tile_configuration() escribe
# un archivo con el formato correcto. Los nombres de parámetros del
# comando de abajo son los que documenta el plugin (ver
# https://imagej.net/plugins/grid-collection-stitching), pero pueden
# variar un poco entre versiones de Fiji — si tira error de
# "parámetro desconocido", corré una vez el plugin a mano con
# Record Macro activado (Plugins > Macros > Record...) para ver el
# nombre exacto que usa tu versión, y ajustá MACRO_STITCHING.

MACRO_STITCHING = """
run("Grid/Collection stitching", "type=[Positions from file] order=[Defined by TileConfiguration] directory=[{carpeta}] layout_file={layout_file} fusion_method=[Linear Blending] regression_threshold={regression_threshold} max/avg_displacement_threshold={max_avg_displacement_threshold} absolute_displacement_threshold={absolute_displacement_threshold} compute_overlap subpixel_accuracy computation_parameters=[Save memory (but be slower)] image_output=[Write to disk] output_directory=[{salida}]");
eval("script", "System.exit(0);");
"""


def _ruta_para_macro(ruta):
    """
    Las rutas que se insertan DENTRO de un string de macro de ImageJ
    (no las que van como argumento de línea de comandos) tienen que
    ir con '/' en vez de '\\': el lenguaje de macros interpreta '\\'
    como carácter de escape (igual que Java), así que una ruta de
    Windows tal cual puede perder pedazos enteros (por ejemplo,
    "\\26" se lee como un escape octal y desaparece). Java acepta
    '/' como separador de carpetas en Windows sin ningún problema,
    así que no hace falta escapar nada.
    """
    return str(ruta).replace("\\", "/")


def _numero_canal(ruta):
    coincidencia = re.search(r"_c(\d+)", ruta.stem)
    return int(coincidencia.group(1)) if coincidencia else 0


def unir_canales_y_guardar(carpeta_salida, nombre_salida="mosaico.tif", ancho_preview=2000):
    """
    El plugin Grid/Collection stitching, al escribir a disco
    ("Write to disk"), separa una imagen de color en un archivo por
    canal (img_t1_z1_c1 = rojo, img_t1_z1_c2 = verde, img_t1_z1_c3 =
    azul) en vez de guardar un único archivo RGB. Esta función los
    busca, los vuelve a unir en una sola imagen y la guarda en
    'carpeta_salida/nombre_salida' (TIFF por defecto — sin pérdida,
    y sin problema con imágenes grandes).

    Si las fotos del barrido fueron en escala de grises, Fiji va a
    escribir un único archivo ('img_t1_z1_c1'): se usa directo, sin
    fusionar nada.

    También guarda una versión chica en JPG ('..._preview.jpg',
    hasta 'ancho_preview' píxeles de ancho) para poder abrirla rápido
    y mirarla sin esperar a que cargue el archivo completo — un
    mosaico de varias decenas de tiles en alta resolución puede
    pesar varios cientos de MB o más, y algunos visores de imágenes
    comunes se cuelgan o tardan mucho con eso.
    """

    carpeta_salida = Path(carpeta_salida)

    canales = sorted(carpeta_salida.glob("img_t1_z1_c*"), key=_numero_canal)

    if len(canales) == 0:
        print(f"[AVISO] No se encontró ningún archivo 'img_t1_z1_c*' en {carpeta_salida}.")
        return None

    if len(canales) == 1:
        mosaico = cv2.imread(str(canales[0]), cv2.IMREAD_UNCHANGED)

    elif len(canales) == 3:
        # Fiji: c1=rojo, c2=verde, c3=azul -> OpenCV espera BGR
        rojo = cv2.imread(str(canales[0]), cv2.IMREAD_UNCHANGED)
        verde = cv2.imread(str(canales[1]), cv2.IMREAD_UNCHANGED)
        azul = cv2.imread(str(canales[2]), cv2.IMREAD_UNCHANGED)
        mosaico = cv2.merge([azul, verde, rojo])

    else:
        print(
            f"[AVISO] Se encontraron {len(canales)} archivos de canal en "
            f"{carpeta_salida} (se esperaba 1 o 3): "
            f"{[c.name for c in canales]}. No se pudo armar la imagen "
            f"combinada automáticamente — revisalos a mano."
        )
        return None

    if mosaico is None:
        print(f"[AVISO] No se pudo leer alguno de los archivos de canal en {carpeta_salida}.")
        return None

    ruta_final = carpeta_salida / nombre_salida

    if not cv2.imwrite(str(ruta_final), mosaico):
        print(f"[AVISO] cv2.imwrite no pudo guardar {ruta_final}.")
        return None

    print(
        f"Mosaico combinado guardado en {ruta_final} "
        f"({mosaico.shape[1]}x{mosaico.shape[0]} px, "
        f"{ruta_final.stat().st_size / 1e6:.1f} MB)"
    )

    if mosaico.shape[1] > ancho_preview:
        factor = ancho_preview / mosaico.shape[1]
        preview = cv2.resize(
            mosaico, (ancho_preview, max(1, int(round(mosaico.shape[0] * factor)))),
            interpolation=cv2.INTER_AREA,
        )
        ruta_preview = carpeta_salida / f"{Path(nombre_salida).stem}_preview.jpg"
        cv2.imwrite(str(ruta_preview), preview)
        print(f"Preview liviana para mirar rápido: {ruta_preview}")

    return ruta_final


def ejecutar_fiji_headless(
    ruta_fiji,
    carpeta_fotos,
    carpeta_salida,
    layout_file="TileConfiguration.txt",
    regression_threshold=0.30,
    max_avg_displacement_threshold=2.50,
    absolute_displacement_threshold=3.50,
    nombre_salida="mosaico.tif",
    timeout_seg=None,
):
    """
    Arma un macro de ImageJ (.ijm) que corre "Grid/Collection
    stitching" con 'layout_file' (el TileConfiguration.txt ya
    generado) y lo ejecuta con Fiji en modo headless (sin abrir
    ninguna ventana), vía línea de comandos:

        <ruta_fiji> --headless --console -macro _stitch_auto.ijm

    'ruta_fiji' es el ejecutable de Fiji, por ejemplo
    r"C:\\Fiji.app\\ImageJ-win64.exe" en Windows o
    "/opt/Fiji.app/ImageJ-linux64" en Linux.

    El plugin guarda el resultado con un nombre fijo que arranca con
    "img_t1_z1_c1" (no se puede elegir el nombre desde el diálogo),
    dentro de 'carpeta_salida' — esta función lo busca y devuelve la
    ruta encontrada, para que después lo renombres o lo muevas como
    quieras.

    Los tres thresholds (regression / max-avg displacement /
    absolute displacement) son los mismos que aparecen en el diálogo
    de Fiji, para filtrar pares de tiles cuya correlación cruzada no
    fue confiable, antes de la optimización global de posiciones.
    Los valores por defecto son los que trae Fiji; si tu barrido
    tiene focos de baja textura (fuera de foco, sin células) y eso
    hace que se descarten de más pares válidos, hay que subirlos.
    """

    carpeta_fotos = Path(carpeta_fotos)
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    macro = MACRO_STITCHING.format(
        carpeta=_ruta_para_macro(carpeta_fotos),
        layout_file=layout_file,
        regression_threshold=regression_threshold,
        max_avg_displacement_threshold=max_avg_displacement_threshold,
        absolute_displacement_threshold=absolute_displacement_threshold,
        salida=_ruta_para_macro(carpeta_salida),
    )

    ruta_macro = carpeta_fotos / "_stitch_auto.ijm"
    ruta_macro.write_text(macro, encoding="utf-8")

    comando = [str(ruta_fiji), "--headless", "--console", "-macro", str(ruta_macro)]

    print("Corriendo Fiji en modo headless (con muchos tiles puede tardar varios minutos)...")
    print(" ".join(comando))

    resultado = subprocess.run(comando, capture_output=True, text=True, timeout=timeout_seg)

    print(resultado.stdout)

    if resultado.returncode != 0:
        print(resultado.stderr)
        raise RuntimeError(f"Fiji terminó con código de salida {resultado.returncode}")

    candidatos = sorted(carpeta_salida.glob("img_t1_z1_c*"))

    if not candidatos:
        print(
            f"[AVISO] No se encontró ningún archivo 'img_t1_z1_c*' en "
            f"{carpeta_salida}. Revisá la salida de Fiji de arriba."
        )
        return None

    return unir_canales_y_guardar(carpeta_salida, nombre_salida=nombre_salida)


# ============================================================
# TODO JUNTO
# ============================================================

def generar_mosaico_fiji(
    carpeta_fotos,
    ruta_fiji,
    fov_x_mm,
    fov_y_mm,
    carpeta_salida=None,
    stage_x_son_filas=True,
    invertir_filas=False,
    invertir_columnas=False,
    extension="*.jpg",
    **kwargs_stitching,
):
    """
    1) Lista los tiles de 'carpeta_fotos' (extraer_coordenadas() del
       nombre de archivo, igual que en reconstruir_mosaico.py).
    2) Escribe TileConfiguration.txt en esa misma carpeta.
    3) Corre Fiji headless con el Grid/Collection stitching.

    Devuelve la ruta del mosaico generado (o None si algo falló).
    """

    carpeta_fotos = Path(carpeta_fotos)

    if carpeta_salida is None:
        carpeta_salida = carpeta_fotos / "mosaico_fiji"

    tiles = listar_tiles_liviano(carpeta_fotos, extension=extension)
    print(f"{len(tiles)} tiles encontrados en {carpeta_fotos}")

    generar_tile_configuration(
        tiles, fov_x_mm, fov_y_mm, carpeta_fotos / "TileConfiguration.txt",
        stage_x_son_filas, invertir_filas, invertir_columnas,
    )

    return ejecutar_fiji_headless(ruta_fiji, carpeta_fotos, carpeta_salida, **kwargs_stitching)


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":

    generar_mosaico_fiji(
        carpeta_fotos=r"C:\BackUp\Facultad\Labo 7\Fotos\26.09.29 Param cam - Fiji mosaico\mosaico prueba",
        ruta_fiji=r"C:\BackUp\Facultad\Labo 7\Fiji.app\ImageJ-win64.exe",
        fov_x_mm=0.21,
        fov_y_mm=0.286,
    )


#%%


if __name__ == "__main__":

    generar_mosaico_fiji(
        carpeta_fotos=r"C:\BackUp\Facultad\Labo 7\Fotos\26.09.29 Param cam - Fiji mosaico\mosaico prueba pasasa parasitos fijos",
        ruta_fiji=r"C:\BackUp\Facultad\Labo 7\Fiji.app\ImageJ-win64.exe",
        fov_x_mm=0.21,
        fov_y_mm=0.286,
    )


# %%
