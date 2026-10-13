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

from camara_openmv import OpenMVCamera
from impresora_cosmos import TrimakerCOSMOS


# ============================================================
# AJUSTE A LA GRILLA DE LA IMPRESORA (XY)
# ============================================================
# La impresora solo puede moverse en múltiplos de 0.0125 mm en x-y.
# TrimakerCOSMOS.ajustar_movimiento() redondea HACIA ABAJO
# (np.floor(valor / resolucion)), y por ruido de punto flotante un
# valor que "debería" ser múltiplo exacto (ej: 149.95) puede dar
# 11999.999999999998 y caer un paso entero (12.5 µm) más abajo.
# Para que eso no pase, todas las coordenadas XY del barrido se
# pasan primero por _a_grilla() y se les suma un epsilon de 1 nm
# (_EPS_MM): no cambia nada si algún día ajustar_movimiento() pasa
# a usar np.round, pero evita el escalón espurio con floor.

PASO_MINIMO_XY_MM = 0.0125
# _EPS_MM = 1e-6
_EPS_MM = 0


def _a_grilla(valor_mm, paso_minimo=PASO_MINIMO_XY_MM):
    """Redondea 'valor_mm' al múltiplo más cercano de 'paso_minimo'."""
    return float(np.round(np.round(valor_mm / paso_minimo) * paso_minimo, 4))


# ============================================================
# GENERADOR DE OFFSETS EN ESPIRAL CUADRADA
# ============================================================

def generar_offsets_espiral(pasos_radiales):
    """
    Genera offsets (i, j) enteros que recorren, en orden de espiral
    (empezando en el centro y expandiéndose hacia afuera), una
    grilla cuadrada de lado (2*pasos_radiales + 1) centrada en
    (0, 0).

    'pasos_radiales' es el tope del barrido: cuántos "anillos"
    alrededor del punto inicial se recorren como máximo. Con
    pasos_radiales=2, por ejemplo, se recorren 25 posiciones (una
    grilla de 5x5 centrada en el punto inicial).

    Cada offset (i, j) se multiplica después por el paso en x/y
    (en mm) para obtener el desplazamiento real. Se devuelven en el
    orden en que la espiral los visita, así que si en algún momento
    se quiere cortar el barrido antes de tiempo (por ejemplo, por un
    error de cámara), lo ya capturado sigue siendo una espiral
    completa y simétrica hasta ese anillo.
    """

    if pasos_radiales < 0:
        raise ValueError("pasos_radiales tiene que ser >= 0")

    offsets = [(0, 0)]

    if pasos_radiales == 0:
        return offsets

    i, j = 0, 0
    paso = 1

    # Se generan un par de vueltas de más (margen) y se recorta con
    # el filtro de abajo, en vez de calcular a mano en qué punto
    # exacto de la vuelta termina el último anillo.
    while paso <= 2 * pasos_radiales + 2:

        for _ in range(paso):  # derecha
            i += 1
            if max(abs(i), abs(j)) <= pasos_radiales:
                offsets.append((i, j))

        for _ in range(paso):  # arriba
            j += 1
            if max(abs(i), abs(j)) <= pasos_radiales:
                offsets.append((i, j))

        paso += 1

        for _ in range(paso):  # izquierda
            i -= 1
            if max(abs(i), abs(j)) <= pasos_radiales:
                offsets.append((i, j))

        for _ in range(paso):  # abajo
            j -= 1
            if max(abs(i), abs(j)) <= pasos_radiales:
                offsets.append((i, j))

        paso += 1

    return offsets


# ============================================================
# ORQUESTADOR: HOME -> POSICIONAR EN MUESTRA -> BARRIDO ESPIRAL
# ============================================================

def barrido_espiral_muestra(
    camara,
    impresora,
    x_ini,
    y_ini,
    z_arriba_seguro,
    z_min_seguridad,
    carpeta_salida,
    pasos_radiales_maximo=3,
    paso_grueso=0.01,
    paso_fino=0.0025,
    paso_x_mm=0.05,
    paso_y_mm=0.0625,
    resolucion_busqueda="QVGA",
    resolucion_alta="WQXGA2",
    # pixformat_busqueda=OpenMVCamera.GRAYSCALE,
    # pixformat_alta=OpenMVCamera.RGB565,
    pixformat_busqueda="GRAYSCALE",
    pixformat_alta="RGB565",
    metrica_foco_final=varianza_intensidad,
    pasos_arriba_foco_local=3,
    pasos_abajo_foco_local=3,
    guardar_fotos_autofoco_local=True,
    feedrate_xy=3000,
    espera_movimiento=0.3,
    espera_movimiento_alta=0.5,
    devolveme_la_respuesta=False,
    kwargs_posicionar=None,
):
    """
    Secuencia completa:

    1) impresora.home()
    2) posicionar_en_muestra() en (x_ini, y_ini): verificación +
       barrido grueso + confirmación, con 'resolucion_busqueda'
       (QVGA por defecto). NO hace el foco final (hacer_foco_final=
       False): el foco fino se hace después, en cada tile.
    3) Barrido en espiral en XY alrededor de (x_ini, y_ini). Para
       CADA tile, en este orden:
         a) moverse en XY a la posición del tile (Z queda donde
            estaba, que es el mejor Z del tile anterior),
         b) camara.config_autofoco_local() y autofocus() ahí mismo
            (el foco se pierde rápido al moverse, así que hay que
            enfocar EN el tile nuevo, no en el anterior). Deja la
            platina en el mejor Z encontrado,
         c) camara.config_de_una() con pixformat_alta/resolucion_alta
            y captura de la foto de calidad alta,
         d) se guarda la foto en 'carpeta_salida'.
       Las fotos de cada autofoco (si guardar_fotos_autofoco_local)
       quedan en la subcarpeta 'carpeta_salida/{índice} i{i}_j{j}/'.

    Pasos XY: 'paso_x_mm'/'paso_y_mm' es el PASO entre tiles
    consecutivos (no el campo visual completo). Como la impresora
    solo se mueve en múltiplos de 0.0125 mm, ambos pasos (y x_ini/
    y_ini) se redondean al múltiplo más cercano (se avisa por
    pantalla si eso los cambia). Por eso el default en y es 0.0625
    (5 pasos, ≈45% de un campo visual de 0.14 mm) y no 0.07 (que
    caería a 0.0625 o 0.075 según la posición). El campo visual
    completo (ej: 0.10 mm / 0.14 mm) es el que después hace falta
    pasarle a construir_mosaico(), en reconstruir_mosaico.py.

    Seguridad en Z: 'pasos_abajo' de cada autofoco se acota en cada
    tile según el Z ACTUAL y 'z_min_seguridad' (no una sola vez al
    principio), porque el Z va derivando de un tile a otro. Se
    cumple z_min_seguridad < z_arriba_seguro (mayor Z = más lejos de
    la muestra = más seguro).

    Cada foto del barrido espiral se guarda directamente en
    'carpeta_salida', con nombre:

        {índice} i{i}_j{j} ({x}, {y}, {z}) - {fecha_hora}.jpg

    El "(x, y, z)" del nombre es el que después usa
    metricas_enfoque.extraer_coordenadas() para reconstruir el
    mosaico (ver reconstruir_mosaico.py) — no cambiar ese formato
    sin actualizar también esa función.

    Devuelve un diccionario con el resultado de posicionar_en_muestra()
    más la lista de tiles del barrido espiral (uno por posición, con
    su (x, y, z) real, sus índices (i, j) de grilla, la ruta del
    archivo guardado y si el autofoco quedó en un borde de su rango).
    """

    if z_min_seguridad >= z_arriba_seguro:
        raise ValueError(
            "z_min_seguridad tiene que ser menor que z_arriba_seguro"
        )

    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    if kwargs_posicionar is None:
        kwargs_posicionar = {}

    if isinstance(metrica_foco_final, str):
        metrica_foco_final_funcion = FUNCIONES_METRICA[metrica_foco_final]
    else:
        metrica_foco_final_funcion = metrica_foco_final

    # --------------------------------------------------------
    # Pasos y punto inicial sobre la grilla de la impresora
    # --------------------------------------------------------

    paso_x_grilla = _a_grilla(paso_x_mm)
    paso_y_grilla = _a_grilla(paso_y_mm)

    if paso_x_grilla <= 0 or paso_y_grilla <= 0:
        raise ValueError(
            f"paso_x_mm/paso_y_mm tienen que ser >= {PASO_MINIMO_XY_MM} mm"
        )

    if paso_x_grilla != paso_x_mm or paso_y_grilla != paso_y_mm:
        print(
            f"[AVISO] Los pasos se ajustaron a múltiplos de "
            f"{PASO_MINIMO_XY_MM} mm: "
            f"x {paso_x_mm} -> {paso_x_grilla} | "
            f"y {paso_y_mm} -> {paso_y_grilla}"
        )

    x_ini_grilla = _a_grilla(x_ini)
    y_ini_grilla = _a_grilla(y_ini)

    if x_ini_grilla != x_ini or y_ini_grilla != y_ini:
        print(
            f"[AVISO] El punto inicial se ajustó a la grilla de la "
            f"impresora: ({x_ini}, {y_ini}) -> "
            f"({x_ini_grilla}, {y_ini_grilla})"
        )

    # --------------------------------------------------------
    # 1) Home
    # --------------------------------------------------------

    impresora.home(devolveme_la_respuesta=devolveme_la_respuesta)
    time.sleep(espera_movimiento)

    # --------------------------------------------------------
    # 2) Posicionar en la muestra (sin foco final)
    # --------------------------------------------------------

    resultado_posicionamiento = posicionar_en_muestra(
        camara=camara,
        impresora=impresora,
        x=x_ini_grilla + _EPS_MM,
        y=y_ini_grilla + _EPS_MM,
        z_arriba_seguro=z_arriba_seguro,
        z_min_seguridad=z_min_seguridad,
        resolucion_global=resolucion_busqueda,
        resolucion_local=resolucion_busqueda,
        pixformat_global=pixformat_busqueda,
        hacer_foco_final=False,
        feedrate_xy=feedrate_xy,
        espera_movimiento=espera_movimiento,
        devolveme_la_respuesta=devolveme_la_respuesta,
        paso_grueso=paso_grueso,
        **kwargs_posicionar,
    )

    x_foco = x_ini_grilla
    y_foco = y_ini_grilla

    z_foco = resultado_posicionamiento["z_confirmado"]

    print(
        f"\nMuestra confirmada en "
        f"({x_foco}, {y_foco}, {z_foco:.4f}). "
        f"Arranca el barrido espiral."
    )

    # --------------------------------------------------------
    # 3) Barrido espiral: mover -> autofoco -> foto en alta
    # --------------------------------------------------------

    offsets = generar_offsets_espiral(pasos_radiales_maximo)

    tiles = []

    for indice, (i, j) in enumerate(offsets):

        # ---- a) mover en XY (Z queda donde está) ------------------

        x_objetivo = _a_grilla(x_foco + i * paso_x_grilla)
        y_objetivo = _a_grilla(y_foco + j * paso_y_grilla)

        _, _, z_actual = impresora.pregunta_posicion_actual()

        impresora.desp_lin_abs(
            x=x_objetivo + _EPS_MM,
            y=y_objetivo + _EPS_MM,
            z=z_actual,
            feedrate=feedrate_xy,
            devolveme_la_respuesta=devolveme_la_respuesta,
        )
        time.sleep(espera_movimiento_alta)

        # Chequeo: la posición que registró la impresora tiene que
        # coincidir con la pedida (si no, algún redondeo se comió un
        # paso de 12.5 µm y el mosaico va a salir corrido).
        if (
            abs(impresora.posicion_x - x_objetivo) > PASO_MINIMO_XY_MM / 2
            or abs(impresora.posicion_y - y_objetivo) > PASO_MINIMO_XY_MM / 2
        ):
            print(
                f"[AVISO] Tile {indice}: se pidió "
                f"({x_objetivo}, {y_objetivo}) pero la impresora quedó en "
                f"({impresora.posicion_x}, {impresora.posicion_y})"
            )

        # ---- b) autofoco EN el tile nuevo -----------------------

        camara.config_autofoco_local()

        _, _, z_antes_autofoco = impresora.pregunta_posicion_actual()

        margen_disponible_abajo = z_antes_autofoco - z_min_seguridad

        pasos_abajo_reales = max(
            min(
                pasos_abajo_foco_local,
                int(margen_disponible_abajo // paso_fino),
            ),
            0,
        )

        carpeta_autofoco_tile = carpeta_salida / f"{indice:04d} i{i}_j{j}"

        mejor_z, posiciones_z_final, valores_final = autofocus(
            camara=camara,
            impresora=impresora,
            metrica=metrica_foco_final_funcion,
            pasos_arriba=pasos_arriba_foco_local,
            pasos_abajo=pasos_abajo_reales,
            paso_z=paso_fino,
            espera_movimiento=espera_movimiento,
            graficamos=False,
            guardamos=guardar_fotos_autofoco_local,
            carpeta_guardado=carpeta_autofoco_tile,
        )

        # Si el máximo cae en un extremo del rango barrido, el foco
        # real probablemente está más allá: el próximo tile arranca
        # de este Z y el error se va acumulando.
        indice_mejor = int(np.argmax(valores_final))
        en_borde = indice_mejor in (0, len(valores_final) - 1)

        if en_borde:
            print(
                f"[AVISO] Tile {indice}: el autofoco quedó en el borde de "
                f"su rango (Z = {mejor_z:.4f}). Ampliá pasos_arriba/"
                f"pasos_abajo_foco_local."
            )

        # autofocus() ya deja la platina en mejor_z.

        # ---- c) foto de calidad alta ----------------------------

        camara.config_de_una(
            pixformat=pixformat_alta,
            resolucion=resolucion_alta,
        )

        image_data = camara.capturar()

        hora_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        nombre = (
            f"{indice:04d} i{i}_j{j} "
            f"({impresora.posicion_x}, "
            f"{impresora.posicion_y}, "
            f"{impresora.posicion_z})"
            f" - {hora_actual}"
        )

        archivo = camara.guardar_imagen(
            image_data=image_data,
            carpeta=carpeta_salida,
            nombre=nombre,
            mostrar=False,
        )

        print(
            f"[{indice + 1}/{len(offsets)}] "
            f"(i={i}, j={j}) -> "
            f"({impresora.posicion_x}, {impresora.posicion_y}, {impresora.posicion_z})"
        )

        tiles.append({
            "indice": indice,
            "i": i,
            "j": j,
            "x": impresora.posicion_x,
            "y": impresora.posicion_y,
            "z": impresora.posicion_z,
            "archivo": archivo,
            "autofoco_en_borde": en_borde,
        })

    resultado_posicionamiento["tiles_espiral"] = tiles
    resultado_posicionamiento["carpeta_salida"] = carpeta_salida

    return resultado_posicionamiento



#%%



grua = TrimakerCOSMOS("COM11")

grua.home()

print(f"{grua.pregunta_posicion_actual() = }")

# grua.desp_lin_abs(150,150,27.71)



#%%



cam = OpenMVCamera("COM7")

cam.conectar()

cam.mostrar(cam.capturar())



#%%



# ==================
# EJEMPLO DE USO
# ==================

resultado = barrido_espiral_muestra(
    camara=cam,
    impresora=grua,
    x_ini=150.0,
    y_ini=150.0,
    z_arriba_seguro=28,   # MAYOR Z = más lejos de la muestra (seguro)
    z_min_seguridad=27.62,   # límite inferior (con margen del objetivo)
    carpeta_salida=r"C:\BackUp\Facultad\Labo 7\Fotos\26.09.29 mosaico prueba pasasa 2",
    pasos_radiales_maximo=0,   # primera prueba chica: 9 tiles
)

print(f"{len(resultado['tiles_espiral'])} fotos guardadas en "
      f"{resultado['carpeta_salida']}")



#%%



# ==================
# EJEMPLO DE USO
# ==================

resultado = barrido_espiral_muestra(
    camara=cam,
    impresora=grua,
    x_ini=150.0,
    y_ini=128.0,
    z_arriba_seguro=28,   # MAYOR Z = más lejos de la muestra (seguro)
    z_min_seguridad=27.62,   # límite inferior (con margen del objetivo)
    carpeta_salida=r"C:\BackUp\Facultad\Labo 7\Fotos\26.09.29 mosaico prueba pasasa parasitos fijos",
    paso_grueso=0.0075,
    metrica_foco_final = laplacian_variance,
    pasos_radiales_maximo=1,   # primera prueba chica: 9 tiles
)

print(f"{len(resultado['tiles_espiral'])} fotos guardadas en "
      f"{resultado['carpeta_salida']}")



# %%



# from pathlib import Path
# import re


# def generar_tile_configuration(
#     carpeta,
#     fov_x_mm=0.10,
#     fov_y_mm=0.14,
#     x_ref=None,
#     y_ref=None,
#     ignorar_z=True,
#     nombre_salida="TileConfiguration.txt",
# ):
#     """
#     Genera un TileConfiguration.txt para Fiji a partir de las
#     coordenadas (x, y, z) presentes en los nombres de las imágenes.

#     Las posiciones generadas son aproximadas y están expresadas
#     en píxeles.

#     x de la impresora -> filas de la imagen
#     y de la impresora -> columnas de la imagen
#     """

#     carpeta = Path(carpeta)

#     archivos = sorted(carpeta.glob("*.jpg"))

#     if not archivos:
#         raise ValueError(f"No se encontraron imágenes en {carpeta}")

#     # ------------------------------------------------------------
#     # Extraer coordenadas
#     # ------------------------------------------------------------

#     datos = []

#     patron = re.compile(
#         r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*"
#         r"([-+]?\d+(?:\.\d+)?)\s*,\s*"
#         r"([-+]?\d+(?:\.\d+)?)\s*\)"
#     )

#     for archivo in archivos:

#         coincidencia = patron.search(archivo.name)

#         if coincidencia is None:
#             print(f"[AVISO] Se saltea: {archivo.name}")
#             continue

#         x, y, z = map(float, coincidencia.groups())

#         datos.append({
#             "archivo": archivo,
#             "x": x,
#             "y": y,
#             "z": z,
#         })

#     if not datos:
#         raise ValueError("No se pudieron extraer coordenadas.")

#     # ------------------------------------------------------------
#     # Elegir origen
#     # ------------------------------------------------------------

#     if x_ref is None:
#         x_ref = min(d["x"] for d in datos)

#     if y_ref is None:
#         y_ref = min(d["y"] for d in datos)

#     # ------------------------------------------------------------
#     # Conversión mm -> píxeles
#     #
#     # x impresora -> filas
#     # y impresora -> columnas
#     # ------------------------------------------------------------

#     px_por_mm_x = None
#     px_por_mm_y = None

#     # Necesitamos el tamaño real de la imagen para conocer
#     # cuántos píxeles corresponden a cada FOV.
#     import cv2

#     primera = cv2.imread(str(datos[0]["archivo"]))

#     if primera is None:
#         raise ValueError(f"No se pudo leer {datos[0]['archivo']}")

#     alto_px, ancho_px = primera.shape[:2]

#     px_por_mm_x = alto_px / fov_x_mm
#     px_por_mm_y = ancho_px / fov_y_mm

#     # ------------------------------------------------------------
#     # Crear archivo
#     # ------------------------------------------------------------

#     ruta_salida = carpeta / nombre_salida

#     with open(ruta_salida, "w", encoding="utf-8") as f:

#         f.write("dim = 3\n")
#         f.write("\n")

#         for d in datos:

#             fila_px = (d["x"] - x_ref) * px_por_mm_x
#             columna_px = (d["y"] - y_ref) * px_por_mm_y

#             if ignorar_z:
#                 z_px = 0.0
#             else:
#                 z_px = d["z"]

#             f.write(
#                 f"{d['archivo'].name}; ; "
#                 f"({fila_px:.3f}, {columna_px:.3f}, {z_px:.3f})\n"
#             )

#     print(f"TileConfiguration generado en:")
#     print(ruta_salida)

#     return ruta_salida



# #%%



# carpeta = r"C:\BackUp\Facultad\Labo 7\Fotos\26.09.29 mosaico prueba"

# generar_tile_configuration(
#     carpeta=carpeta,
#     fov_x_mm=0.214,
#     fov_y_mm=0.286,
#     x_ref=150.0,
#     y_ref=150.0,
# )


# # %%
