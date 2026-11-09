# %%

"""
Análisis de las fotos de un barrido (aplicación 26.11.09 medicion de aumentos y parametros de camara).

Es el análisis de 26.09.11 (aplic_analisis_del_barrido_grueso_claude.py) más lo que hace falta para elegir
los aumentos ("x") de buscar_foco_en_xy():
  - por foto: las métricas de enfoque (laplacian, tenengrad, varianza), la intensidad de fondo, la fracción
    de rojo y las medias de color, y lo que informó la cámara (info_camara.csv: exposición, ganancia);
  - gráficos de todo contra Z, con las curvas de varias carpetas (una por combinación de parámetros de cámara)
    en los mismos ejes;
  - lo nuevo: cada métrica dividida por su línea de base (el promedio de las primeras fotos del descenso, como
    la calcula la búsqueda), y una tabla con la línea de base, el ruido y cuánto sube la métrica en el foco.

Las métricas salen de metricas_imagen (base); extraer_coordenadas, cargar_imagen, fraccion_pixeles_rojos y
medias_de_color son las de 26.09.11, que vivían en "Oudated code/metricas_enfoque.py".
"""

import csv
import math
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from metricas_imagen import (
    FUNCIONES_METRICA,
    intensidad_fondo,
    laplacian_variance,
    linea_de_base,
    tenengrad,
    varianza_intensidad,
)

from barrido_con_fotos import NOMBRE_CSV_INFO_CAMARA


# Lo que se grafica contra Z: nombre en el gráfico -> clave de la fila
METRICAS_DE_LA_FOTO = {
    "Laplaciano": "laplacian",
    "Tenengrad": "tenengrad",
    "Varianza de intensidad": "varianza",
    "Intensidad de fondo": "intensidad_fondo",
    "Fracción de rojo": "fraccion_rojo",
    "Saturación media": "saturacion_media",
}

# Lo que informó la cámara en cada foto (info_camara.csv)
VALORES_DE_LA_CAMARA = {
    "Exposición [µs]": "exposure_us",
    "Ganancia [dB]": "gain_db",
}

CLAVES_INFO_CAMARA = ("exposure_us", "gain_db", "rgb_r_db", "rgb_g_db", "rgb_b_db")


# ============================================================
# LEER UNA FOTO
# ============================================================

def extraer_coordenadas(ruta):
    """Extrae (x, y, z) de un nombre como "2026-11-09_18-04-55 (150.0, 150.0, 27.9).jpg"."""

    patron = r"\(([-+0-9.eE]+),\s*([-+0-9.eE]+),\s*([-+0-9.eE]+)\)"

    coincidencia = re.search(patron, Path(ruta).stem)

    if coincidencia is None:
        raise ValueError(f"No se encontraron coordenadas en:\n{Path(ruta).name}")

    x, y, z = map(float, coincidencia.groups())

    return x, y, z


def cargar_imagen(ruta):
    """Lee una foto guardada como imagen BGR (cv2.imread siempre entrega 3 canales, también si era GRAYSCALE)."""

    imagen = cv2.imread(str(ruta))

    if imagen is None:
        raise ValueError(f"No se pudo cargar:\n{ruta}")

    return imagen


def fraccion_pixeles_rojos(imagen_bgr, s_min=50, v_min=50, v_max=255):
    """Fracción de píxeles rojos (HSV) de una imagen a color."""

    if imagen_bgr.ndim == 2:
        raise ValueError(
            "fraccion_pixeles_rojos necesita una imagen a color (pixformat RGB565 o JPEG a color); "
            "recibió una imagen en escala de grises."
        )

    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)

    mascara_1 = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([10, 255, v_max]))
    mascara_2 = cv2.inRange(hsv, np.array([160, s_min, v_min]), np.array([180, 255, v_max]))

    return cv2.bitwise_or(mascara_1, mascara_2).mean() / 255.0


def medias_de_color(imagen_bgr):
    """Media de cada canal (R, G, B) y saturación media (HSV): cómo va apareciendo el color al cambiar Z."""

    b, g, r = cv2.split(imagen_bgr.astype(np.float64))

    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)

    return {
        "media_r": r.mean(),
        "media_g": g.mean(),
        "media_b": b.mean(),
        "saturacion_media": hsv[:, :, 1].mean(),
    }


# ============================================================
# ANALIZAR UNA CARPETA (UN BARRIDO) Y UNA MEDICIÓN (VARIOS BARRIDOS)
# ============================================================

def _leer_info_camara(carpeta):
    """{nombre de archivo: {clave: valor}} de info_camara.csv, o {} si la carpeta no lo tiene."""

    ruta_csv = Path(carpeta) / NOMBRE_CSV_INFO_CAMARA

    if not ruta_csv.exists():
        return {}

    info = {}

    with open(ruta_csv, newline="", encoding="utf-8") as archivo_csv:
        for fila in csv.DictReader(archivo_csv):
            info[fila["archivo"]] = {
                clave: float(fila[clave]) if fila.get(clave) not in (None, "") else math.nan
                for clave in CLAVES_INFO_CAMARA
            }

    return info


def analizar_carpeta(carpeta):
    """
    Calcula las métricas de todas las fotos .jpg de una carpeta (un barrido). Devuelve una lista de diccionarios
    (una por foto, ordenada por z creciente) con archivo, x, y, z, las métricas y lo que informó la cámara
    (NaN si la carpeta no tiene info_camara.csv).
    """

    carpeta = Path(carpeta)

    archivos = sorted(carpeta.glob("*.jpg"))

    if len(archivos) == 0:
        raise ValueError(f"No se encontraron imágenes JPG en:\n{carpeta}")

    info = _leer_info_camara(carpeta)

    resultados = []

    for archivo in archivos:

        try:
            x, y, z = extraer_coordenadas(archivo)

            imagen = cargar_imagen(archivo)

            fila = {
                "archivo": archivo.name,
                "x": x,
                "y": y,
                "z": z,
                "laplacian": laplacian_variance(imagen),
                "tenengrad": tenengrad(imagen),
                "varianza": varianza_intensidad(imagen),
                "intensidad_fondo": intensidad_fondo(imagen),
                "fraccion_rojo": fraccion_pixeles_rojos(imagen),
            }

            fila.update(medias_de_color(imagen))
            fila.update(info.get(archivo.name, {clave: math.nan for clave in CLAVES_INFO_CAMARA}))

            resultados.append(fila)

        except Exception as motivo:
            print(f"No se pudo procesar {archivo.name}: {motivo}")

    resultados.sort(key=lambda resultado: resultado["z"])

    return resultados


def etiquetas_cortas(nombres_de_carpeta):
    """
    Etiqueta de cada carpeta para la leyenda: solo las partes del nombre (separadas por " - ") que cambian
    entre carpetas. Si todas las partes son iguales, usa el nombre entero.
    """

    partes = [nombre.split(" - ") for nombre in nombres_de_carpeta]

    if len(partes) < 2 or len({len(p) for p in partes}) != 1:
        return list(nombres_de_carpeta)

    cambian = [i for i in range(len(partes[0])) if len({p[i] for p in partes}) > 1]

    if not cambian:
        return list(nombres_de_carpeta)

    return [" - ".join(p[i] for i in cambian) for p in partes]


def analizar_medicion(carpeta_de_la_medicion):
    """
    Analiza todas las carpetas de barrido (una por combinación de parámetros de cámara) que hay dentro de
    'carpeta_de_la_medicion' (la de un (x, y): "<muestra> xy 150_150 - pasada 1"). Devuelve
    {etiqueta: lista de resultados de analizar_carpeta()}.
    """

    carpetas = sorted(
        carpeta for carpeta in Path(carpeta_de_la_medicion).iterdir()
        if carpeta.is_dir() and any(carpeta.glob("*.jpg"))
    )

    if not carpetas:
        raise ValueError(f"No hay carpetas con fotos dentro de:\n{carpeta_de_la_medicion}")

    etiquetas = etiquetas_cortas([carpeta.name for carpeta in carpetas])

    resultados_por_etiqueta = {}

    for carpeta, etiqueta in zip(carpetas, etiquetas):
        print(f"Analizando: {carpeta.name}")
        resultados_por_etiqueta[etiqueta] = analizar_carpeta(carpeta)

    return resultados_por_etiqueta


# ============================================================
# EL DESCENSO COMO LO VE LA BÚSQUEDA (LÍNEA DE BASE Y RESUMEN)
# ============================================================

def serie_del_descenso(resultados, paso_grueso=0.005):
    """
    Las fotos en el orden en que las ve la búsqueda: de la Z más alta a la más baja, una cada 'paso_grueso'
    (si el barrido se hizo con un paso más fino, se saltean fotos).
    """

    filas = sorted(resultados, key=lambda resultado: resultado["z"], reverse=True)

    if len(filas) < 2:
        return filas

    paso_medido = float(np.median(np.abs(np.diff([fila["z"] for fila in filas]))))

    salto = max(1, round(paso_grueso / paso_medido))

    return filas[::salto]


def resumen_por_metrica(resultados, pasos_linea_base=5, paso_grueso=0.005):
    """
    Para cada métrica de enfoque (laplacian, tenengrad, varianza), sobre el descenso que ve la búsqueda:
      "linea_base"       promedio de las primeras 'pasos_linea_base' fotos;
      "ruido_relativo"   desvío de esas fotos dividido por su promedio (un aumento menor que esto dispara por ruido);
      "pico_relativo"    el máximo del resto del descenso dividido por la línea de base (un aumento mayor que
                         pico_relativo - 1 nunca dispara);
      "z_pico"           la Z de ese máximo.
    """

    serie = serie_del_descenso(resultados, paso_grueso)

    valores_por_foto = [{nombre: fila[nombre] for nombre in FUNCIONES_METRICA} for fila in serie]

    base = linea_de_base(valores_por_foto, FUNCIONES_METRICA, pasos_linea_base)

    if base is None:
        raise ValueError(
            f"El descenso tiene {len(serie)} fotos (de a {paso_grueso} mm): no alcanza para una línea de base de "
            f"{pasos_linea_base} fotos"
        )

    resumen = {}

    for nombre in FUNCIONES_METRICA:

        primeras = [valores[nombre] for valores in valores_por_foto[:pasos_linea_base]]
        resto = serie[pasos_linea_base:]

        fila_pico = max(resto, key=lambda fila: fila[nombre]) if resto else None

        resumen[nombre] = {
            "linea_base": base[nombre],
            "ruido_relativo": float(np.std(primeras) / base[nombre]) if base[nombre] > 0 else math.nan,
            "pico_relativo": fila_pico[nombre] / base[nombre] if fila_pico and base[nombre] > 0 else math.nan,
            "z_pico": fila_pico["z"] if fila_pico else math.nan,
        }

    return resumen


def imprimir_resumen(resultados_por_etiqueta, pasos_linea_base=5, paso_grueso=0.005):
    """Imprime una tabla con resumen_por_metrica() de cada carpeta."""

    print(f"{'carpeta':50s} {'métrica':10s} {'línea de base':>14s} {'ruido':>8s} {'pico/base':>10s} {'z del pico':>11s}")

    for etiqueta, resultados in resultados_por_etiqueta.items():
        try:
            resumen = resumen_por_metrica(resultados, pasos_linea_base, paso_grueso)
        except ValueError as motivo:
            print(f"{etiqueta[:50]:50s} {motivo}")
            continue

        for nombre, datos in resumen.items():
            print(
                f"{etiqueta[:50]:50s} {nombre:10s} {datos['linea_base']:14.2f} "
                f"{datos['ruido_relativo']:8.3f} {datos['pico_relativo']:10.2f} {datos['z_pico']:11.4f}"
            )


# ============================================================
# GRÁFICOS
# ============================================================

def graficar_metricas(resultados_por_etiqueta, titulo="Métricas vs Z"):
    """Todas las métricas y lo que informó la cámara contra Z, con una curva por carpeta (etiqueta). Devuelve la figura."""

    paneles = {**METRICAS_DE_LA_FOTO, **VALORES_DE_LA_CAMARA}

    figura, ejes = plt.subplots(2, 4, figsize=(22, 9))
    ejes = ejes.flatten()

    for eje, (nombre, clave) in zip(ejes, paneles.items()):

        for etiqueta, resultados in resultados_por_etiqueta.items():
            eje.plot(
                [resultado["z"] for resultado in resultados],
                [resultado[clave] for resultado in resultados],
                marker=".",
                linewidth=1,
                label=etiqueta,
            )

        eje.set_xlabel("Z [mm]")
        eje.set_ylabel(nombre)
        eje.grid(True)

    ejes[0].legend(fontsize=7)

    figura.suptitle(titulo, fontsize=14)
    plt.tight_layout()

    return figura


def graficar_relativo_a_la_linea_de_base(
    resultados_por_etiqueta,
    aumentos_candidatos=(),
    pasos_linea_base=5,
    paso_grueso=0.005,
    escala_log=True,
    titulo="Cada métrica dividida por su línea de base",
):
    """
    Laplaciano, tenengrad y varianza divididos por su línea de base, contra Z (una curva por carpeta). Las
    líneas grises punteadas están en 1 + aumento de cada 'aumentos_candidatos': una curva que cruza la
    línea antes de llegar al foco, y a partir de ahí no vuelve a bajarla, es un aumento que sirve. Devuelve la figura.
    """

    figura, ejes = plt.subplots(1, 3, figsize=(22, 6))

    for eje, nombre in zip(ejes, FUNCIONES_METRICA):

        for etiqueta, resultados in resultados_por_etiqueta.items():

            serie = serie_del_descenso(resultados, paso_grueso)
            valores_por_foto = [{nombre: fila[nombre]} for fila in serie]
            base = linea_de_base(valores_por_foto, [nombre], pasos_linea_base)

            if base is None or base[nombre] <= 0:
                print(f"'{etiqueta}': sin línea de base para {nombre}; no se grafica.")
                continue

            eje.plot(
                [fila["z"] for fila in serie],
                [fila[nombre] / base[nombre] for fila in serie],
                marker=".",
                linewidth=1,
                label=etiqueta,
            )

        eje.axhline(1, color="black", linewidth=0.8)

        for aumento in aumentos_candidatos:
            eje.axhline(1 + aumento, color="gray", linestyle="--", linewidth=0.8)

        if escala_log:
            eje.set_yscale("log")

        eje.set_xlabel("Z [mm]")
        eje.set_ylabel(f"{nombre} / línea de base")
        eje.grid(True)

    ejes[0].legend(fontsize=7)

    figura.suptitle(titulo, fontsize=14)
    plt.tight_layout()

    return figura


def exportar_csv(resultados, ruta_csv):
    """Guarda las filas de analizar_carpeta() en un CSV, para mirar los números en detalle."""

    if len(resultados) == 0:
        return

    with open(ruta_csv, "w", newline="", encoding="utf-8") as archivo_csv:
        escritor = csv.DictWriter(archivo_csv, fieldnames=list(resultados[0].keys()))
        escritor.writeheader()
        escritor.writerows(resultados)

    print(f"CSV guardado en:\n{ruta_csv}")


# %%
