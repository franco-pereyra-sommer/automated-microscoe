#%%



# Agrego la raiz del proyecto
import sys
import os

os.environ["PYTHONPATH"] = r"C:\BackUp\Facultad\Labo 7\env_labo7"

sys.path.insert(0, os.environ["PYTHONPATH"])

from pathlib import Path

import cv2
import numpy as np
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt

from metricas_enfoque import extraer_coordenadas, cargar_imagen


# ============================================================
# CARGAR TILES DE UNA CARPETA
# ============================================================

def cargar_tiles(carpeta, extension="*.jpg"):
    """
    Carga todas las imágenes de 'carpeta' (no recursivo, así que no
    entra a las subcarpetas de fotos de autofoco) y les extrae
    (x, y, z) del nombre de archivo con extraer_coordenadas() — la
    misma convención de nombre "... (x, y, z).jpg" que ya se usa en
    el resto del proyecto.
    """

    carpeta = Path(carpeta)

    archivos = sorted(carpeta.glob(extension))

    tiles = []

    for archivo in archivos:

        try:
            x, y, z = extraer_coordenadas(archivo)
        except ValueError:
            print(f"Se salteó (sin coordenadas en el nombre): {archivo.name}")
            continue

        imagen = cargar_imagen(archivo)

        tiles.append({
            "archivo": archivo,
            "x": x,
            "y": y,
            "z": z,
            "imagen": imagen,
        })

    if len(tiles) == 0:
        raise ValueError(f"No se encontraron tiles válidos en {carpeta}")

    return tiles


# ============================================================
# EJE DE LA CÁMARA vs. EJE DE LA IMPRESORA
# ============================================================
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
# REFINAMIENTO POR CORRELACIÓN DE FASE
# ============================================================

def _agrupar_por_grilla(tiles, x_ref, y_ref, paso_x_mm, paso_y_mm, tolerancia_mm=1e-3):
    """
    Devuelve un diccionario (i, j) -> tile, calculando (i, j) como la
    cantidad de pasos nominales de cada tile respecto de
    (x_ref, y_ref) (el x_ini/y_ini que se usó para armar el barrido
    espiral). Avisa por pantalla si algún tile no cae justo en la
    grilla esperada (más allá de 'tolerancia_mm').
    """

    grilla = {}

    for tile in tiles:

        i = round((tile["x"] - x_ref) / paso_x_mm)
        j = round((tile["y"] - y_ref) / paso_y_mm)

        x_nominal = x_ref + i * paso_x_mm
        y_nominal = y_ref + j * paso_y_mm

        if (abs(tile["x"] - x_nominal) > tolerancia_mm
                or abs(tile["y"] - y_nominal) > tolerancia_mm):
            print(
                f"[AVISO] {Path(tile['archivo']).name}: posición real "
                f"({tile['x']}, {tile['y']}) no coincide con la grilla "
                f"nominal esperada para (i={i}, j={j}) -> "
                f"({x_nominal}, {y_nominal})"
            )

        if (i, j) in grilla:
            print(
                f"[AVISO] Dos tiles cayeron en la misma celda (i={i}, "
                f"j={j}): {Path(grilla[(i, j)]['archivo']).name} y "
                f"{Path(tile['archivo']).name}. Se usa el segundo."
            )

        grilla[(i, j)] = tile

    return grilla


def _a_gris_float(imagen):
    if imagen.ndim == 2:
        return imagen.astype(np.float64)
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY).astype(np.float64)


def refinar_posiciones_con_correlacion(
    tiles,
    x_ref,
    y_ref,
    paso_x_mm,
    paso_y_mm,
    escala_filas,
    escala_columnas,
    invertir_filas=False,
    invertir_columnas=False,
    stage_x_son_filas=True,
    max_correccion_px=None,
    upsample_factor=10,
):
    """
    Corrige, tile por tile, el desplazamiento nominal (calculado a
    partir del x-y real que reportó la impresora) con el
    desplazamiento que en verdad hace coincidir a cada tile con sus
    vecinos ya ubicados, en la franja donde se superponen. Útil
    cuando hay un error mecánico (juego/backlash al revertir el
    sentido de movimiento, por ejemplo) que un factor de escala fijo
    (campo visual) no puede corregir — un desplazamiento así no
    depende de la distancia recorrida, así que ningún valor de
    fov_x_mm/fov_y_mm lo va a arreglar.

    Arma la grilla (i, j) a partir de la posición real de cada tile
    respecto de (x_ref, y_ref) (ver _agrupar_por_grilla) y la
    recorre en anillos concéntricos desde (0, 0) (BFS): cada tile se
    ubica sumando, a la posición YA CORREGIDA de un vecino
    inmediato, el paso nominal más la corrección que da la
    correlación de fase entre la franja de superposición nominal de
    ambos. Así, un error que se arrastra de tile a tile (como el
    backlash) se acumula en la corrección de la misma forma en que
    se acumuló en la platina real, en vez de recalcularse desde cero
    en cada tile.

    Devuelve un diccionario tile["archivo"] -> (fila_px, columna_px)
    — la esquina superior-izquierda de cada tile, con (0, 0) en el
    tile de (x_ref, y_ref). construir_mosaico() se encarga de
    normalizar esto al origen del canvas.

    'max_correccion_px': acota cuánto puede corregir la correlación
    de fase respecto de lo nominal, tile a tile. Una correlación de
    fase sobre una franja sin textura clara (fuera de foco, sin
    células) puede "engancharse" en un pico espurio y tirar el tile
    a un lugar disparatado — mejor acotarla (por ejemplo, a la mitad
    de un paso nominal en píxeles) que dejarla libre. Por defecto no
    hay tope.
    """

    alto_px, ancho_px = tiles[0]["imagen"].shape[:2]

    grilla = _agrupar_por_grilla(tiles, x_ref, y_ref, paso_x_mm, paso_y_mm)

    if (0, 0) not in grilla:
        raise ValueError(
            "No se encontró ningún tile en (x_ref, y_ref) — hace falta "
            "un tile en el origen de la grilla para anclar la corrección."
        )

    d_fila_i, d_col_i = _posicion_nominal_px(
        paso_x_mm, 0.0, escala_filas, escala_columnas,
        invertir_filas, invertir_columnas, stage_x_son_filas,
    )
    d_fila_j, d_col_j = _posicion_nominal_px(
        0.0, paso_y_mm, escala_filas, escala_columnas,
        invertir_filas, invertir_columnas, stage_x_son_filas,
    )

    vecinos_nominales = {
        (1, 0): (d_fila_i, d_col_i),
        (-1, 0): (-d_fila_i, -d_col_i),
        (0, 1): (d_fila_j, d_col_j),
        (0, -1): (-d_fila_j, -d_col_j),
    }

    posiciones = {(0, 0): (0.0, 0.0)}
    visitados = {(0, 0)}
    cola = [(0, 0)]

    grises = {ij: _a_gris_float(tile["imagen"]) for ij, tile in grilla.items()}

    while cola:

        i, j = cola.pop(0)
        gris_actual = grises[(i, j)]

        for (di, dj), (d_fila, d_col) in vecinos_nominales.items():

            vecino = (i + di, j + dj)

            if vecino not in grilla or vecino in visitados:
                continue

            gris_vecino = grises[vecino]

            fila_ini_a = max(0, int(round(d_fila)))
            fila_fin_a = alto_px + min(0, int(round(d_fila)))
            col_ini_a = max(0, int(round(d_col)))
            col_fin_a = ancho_px + min(0, int(round(d_col)))

            franja_actual = gris_actual[fila_ini_a:fila_fin_a, col_ini_a:col_fin_a]

            fila_ini_v = max(0, -int(round(d_fila)))
            fila_fin_v = alto_px + min(0, -int(round(d_fila)))
            col_ini_v = max(0, -int(round(d_col)))
            col_fin_v = ancho_px + min(0, -int(round(d_col)))

            franja_vecino = gris_vecino[fila_ini_v:fila_fin_v, col_ini_v:col_fin_v]

            if (franja_actual.size == 0 or franja_vecino.size == 0
                    or franja_actual.shape != franja_vecino.shape):
                # No hay superposición nominal (paso >= campo visual):
                # no se puede refinar con correlación, se usa lo nominal.
                correccion_fila, correccion_col = 0.0, 0.0
            else:
                shift, _error, _fase = phase_cross_correlation(
                    franja_actual, franja_vecino, upsample_factor=upsample_factor,
                )
                correccion_fila, correccion_col = float(shift[0]), float(shift[1])

            if max_correccion_px is not None:
                correccion_fila = float(np.clip(correccion_fila, -max_correccion_px, max_correccion_px))
                correccion_col = float(np.clip(correccion_col, -max_correccion_px, max_correccion_px))

            fila_base, col_base = posiciones[(i, j)]
            posiciones[vecino] = (
                fila_base + d_fila + correccion_fila,
                col_base + d_col + correccion_col,
            )
            visitados.add(vecino)
            cola.append(vecino)

    if len(visitados) < len(grilla):
        faltantes = set(grilla) - visitados
        print(
            f"[AVISO] {len(faltantes)} tiles no tienen ningún vecino "
            f"adyacente en la grilla (huecos en el barrido) y no se "
            f"pudieron refinar: {sorted(faltantes)}. Van a usar su "
            f"posición nominal, sin corrección."
        )
        for ij in faltantes:
            i, j = ij
            fila, col = _posicion_nominal_px(
                i * paso_x_mm, j * paso_y_mm, escala_filas, escala_columnas,
                invertir_filas, invertir_columnas, stage_x_son_filas,
            )
            posiciones[ij] = (fila, col)

    return {grilla[ij]["archivo"]: pos for ij, pos in posiciones.items()}


# ============================================================
# CONSTRUIR EL MOSAICO
# ============================================================

def construir_mosaico(
    tiles,
    fov_x_mm,
    fov_y_mm,
    invertir_filas=False,
    invertir_columnas=False,
    stage_x_son_filas=True,
    usar_feather=True,
    refinar_con_correlacion=False,
    x_ref=None,
    y_ref=None,
    paso_x_mm=None,
    paso_y_mm=None,
    max_correccion_px=None,
):
    """
    Ubica cada tile en un canvas único usando su (x, y) real en mm y
    funde las zonas de superposición con una ventana de Hann
    (feather blending), para que no se note el borde entre fotos
    vecinas. No hace falta ninguna librería de stitching por feature
    matching: la posición real ya se conoce de antemano.

    EJES: 'stage_x_son_filas=True' (default, confirmado a mano) dice
    que el eje x de la impresora corresponde al eje de FILAS de la
    imagen, y el eje y al de COLUMNAS — la cámara está ~90° rotada
    respecto de la platina. No hace falta rotar ninguna foto: esta
    función arma el mosaico así directamente. Si algún día se
    verifica que ya no es así, poné stage_x_son_filas=False.

    'fov_x_mm'/'fov_y_mm': campo visual real (en mm) que cubre CADA
    tile en cada eje DE LA IMPRESORA, a la resolución con la que se
    sacaron estas fotos (no mezclar con el campo visual a otra
    resolución).

    'invertir_filas'/'invertir_columnas': si el mosaico queda
    espejado en algún eje. Con los ejes ya intercambiados
    (stage_x_son_filas=True) y sin rotar las fotos, no debería hacer
    falta ninguno de los dos — probado a mano con hasta 3 tiles.

    'refinar_con_correlacion': si es True, antes de armar el canvas
    corrige la posición de cada tile con correlación de fase contra
    sus vecinos (ver refinar_posiciones_con_correlacion) — hace
    falta pasar también 'x_ref', 'y_ref' (el x_ini/y_ini del
    barrido) y 'paso_x_mm'/'paso_y_mm' (el paso nominal entre
    tiles). Esto corrige errores de un mismo tamaño en píxeles sin
    importar la distancia recorrida (como backlash mecánico al
    revertir el sentido de movimiento), que ningún valor de
    fov_x_mm/fov_y_mm puede arreglar porque no son un problema de
    escala.
    """

    alto_px, ancho_px = tiles[0]["imagen"].shape[:2]

    for tile in tiles:
        if tile["imagen"].shape[:2] != (alto_px, ancho_px):
            raise ValueError(
                f"Los tiles no tienen todos el mismo tamaño de imagen "
                f"({Path(tile['archivo']).name} mide distinto) — ¿se "
                f"mezclaron fotos de resoluciones distintas?"
            )

    escala_filas, escala_columnas = _escalas_px_por_mm(
        ancho_px, alto_px, fov_x_mm, fov_y_mm, stage_x_son_filas
    )

    if refinar_con_correlacion:

        if x_ref is None or y_ref is None or paso_x_mm is None or paso_y_mm is None:
            raise ValueError(
                "refinar_con_correlacion=True necesita x_ref, y_ref, "
                "paso_x_mm y paso_y_mm."
            )

        posiciones_por_archivo = refinar_posiciones_con_correlacion(
            tiles, x_ref, y_ref, paso_x_mm, paso_y_mm,
            escala_filas, escala_columnas,
            invertir_filas, invertir_columnas, stage_x_son_filas,
            max_correccion_px,
        )
        posiciones = [posiciones_por_archivo[tile["archivo"]] for tile in tiles]

    else:
        posiciones = [
            _posicion_nominal_px(
                tile["x"], tile["y"], escala_filas, escala_columnas,
                invertir_filas, invertir_columnas, stage_x_son_filas,
            )
            for tile in tiles
        ]

    filas = [pos[0] for pos in posiciones]
    columnas = [pos[1] for pos in posiciones]

    fila_min, fila_max = min(filas), max(filas)
    col_min, col_max = min(columnas), max(columnas)

    alto_canvas = int(round(fila_max - fila_min)) + alto_px
    ancho_canvas = int(round(col_max - col_min)) + ancho_px

    es_color = tiles[0]["imagen"].ndim == 3
    canales = tiles[0]["imagen"].shape[2] if es_color else 1

    acumulado = np.zeros((alto_canvas, ancho_canvas, canales), dtype=np.float64)
    pesos = np.zeros((alto_canvas, ancho_canvas, 1), dtype=np.float64)

    if usar_feather:
        ventana_fila = np.hanning(alto_px)
        ventana_columna = np.hanning(ancho_px)
        ventana_2d = np.outer(ventana_fila, ventana_columna)
        # Piso chico en vez de 0 puro: en las esquinas del mosaico,
        # donde no hay superposición con ningún otro tile, un peso
        # exactamente 0 en el borde dejaría esa franja en negro.
        ventana_2d = np.clip(ventana_2d, 1e-3, None)
    else:
        ventana_2d = np.ones((alto_px, ancho_px))

    peso_tile = ventana_2d[:, :, None]

    for tile, (fila, columna) in zip(tiles, posiciones):

        imagen = tile["imagen"].astype(np.float64)
        if imagen.ndim == 2:
            imagen = imagen[:, :, None]

        px_fila = int(round(fila - fila_min))
        px_col = int(round(columna - col_min))

        acumulado[px_fila:px_fila + alto_px, px_col:px_col + ancho_px, :] += imagen * peso_tile
        pesos[px_fila:px_fila + alto_px, px_col:px_col + ancho_px, :] += peso_tile

    pesos_seguros = np.where(pesos == 0, 1, pesos)
    mosaico = acumulado / pesos_seguros
    mosaico = np.clip(mosaico, 0, 255).astype(np.uint8)

    if not es_color:
        mosaico = mosaico[:, :, 0]

    return mosaico


# ============================================================
# GUARDAR + MOSTRAR
# ============================================================

def guardar_mosaico(mosaico, ruta_salida):

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(ruta_salida), mosaico)

    print(f"Mosaico guardado en:\n{ruta_salida}")


def mostrar_mosaico(mosaico, titulo="Mosaico"):

    fig, ax = plt.subplots(figsize=(10, 10))

    if mosaico.ndim == 2:
        ax.imshow(mosaico, cmap="gray")
    else:
        ax.imshow(cv2.cvtColor(mosaico, cv2.COLOR_BGR2RGB))

    ax.set_title(titulo)
    ax.axis("off")

    plt.show()


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":

    carpeta = r"C:\BackUp\Facultad\Labo 7\Fotos\26.10.02 mosaico parasitos fijos pasada 4"

    tiles = cargar_tiles(carpeta)

    print(f"{len(tiles)} tiles cargados de {carpeta}")

    mosaico = construir_mosaico(
        tiles,
        fov_x_mm=0.21, #0.214,  # campo visual COMPLETO en x (de la impresora), no el paso entre tiles
        fov_y_mm=0.275, #0.286,  # campo visual COMPLETO en y (de la impresora), no el paso entre tiles
        refinar_con_correlacion=False,
        x_ref=150.0,     # el x_ini que se usó en barrido_espiral_muestra()
        y_ref=153.0,     # el y_ini que se usó en barrido_espiral_muestra()
        paso_x_mm=0.1,
        paso_y_mm=0.15,
    )

    # guardar_mosaico(mosaico, Path(carpeta) / "mosaico.jpg")
    mostrar_mosaico(mosaico)
_escalas_px_por_mm

#%%


