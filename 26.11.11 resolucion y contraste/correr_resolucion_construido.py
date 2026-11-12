# %%

"""
Resolución y contraste de una foto del micro construido, con los glóbulos rojos marcados en ImageJ.
Carpeta: 26.11.11 resolucion y contraste          Inicio: 2026-11-11
Objetivo: medir resolución (ancho del borde, MTF) y contraste de glóbulos rojos individuales en la foto
    de un sistema óptico, para poder comparar sistemas (el de mesa y las distintas versiones del construido).
Enfoque (qué se propone para resolverlo): desde el centro de cada glóbulo se trazan N rayos; en cada rayo se
    ajusta el borde con una función error (ERF) y el rayo se descarta si choca con un vecino o el ajuste es
    malo. Resolución = sigma mediano de los rayos válidos (y su dispersión); contraste = Michelson y Weber
    del borde y Weber del interior. Detalle y justificación en perfiles_de_borde.py y resolucion_contraste.py.
Módulos base que usa: ninguno. Todo es imagen pura (los módulos de esta carpeta no importan nada del proyecto).
Propuestas para la base (si hay): ninguna por ahora (Franco, 2026-11-11: no agregar nada a la base todavía).
    La escala (um/px) de cada sistema entra como parámetro (UM_POR_PX); cuando se la mida con la grilla de
    10 um corresponde proponerla para la base (ver REGLAS_CODIGO.txt, sección 4).
Estado: EN PRUEBA        Resultado: probado con glóbulos sintéticos (test_perfiles_de_borde.py) y con la
    foto del 2026-10-29; falta correrlo en el equipo y medir la escala con la grilla.

Archivos de esta carpeta:
    circulos_imagej.py        lee los CSV de ImageJ (centro y diámetro aproximado de cada glóbulo)
    perfiles_de_borde.py      rayos, ajuste de la ERF por rayo, validez de cada rayo, perfil medio alineado
    resolucion_contraste.py   sigma -> FWHM, ancho 10-90, MTF; contraste; medir_globulo()
    graficos_resolucion.py    figura de diagnóstico de 4 paneles por glóbulo
    test_perfiles_de_borde.py pruebas con glóbulos sintéticos
"""

import csv
from pathlib import Path

import cv2

from circulos_imagej import leer_circulos_imagej
from graficos_resolucion import graficar_globulo
from perfiles_de_borde import ConfigPerfiles
from resolucion_contraste import canal_de_trabajo, medir_globulo


# ============================================================
# CONFIGURACIÓN (lo que se cambia entre corridas)
# ============================================================

CARPETA_FOTOS = Path(r"C:\BackUp\Facultad\Labo 7\Fotos\26.11.11 en busca de sacar resolucion y contraste")
RUTA_IMAGEN = CARPETA_FOTOS / "116684 (147.7875, 139.3625, 27.58) 2026-10-29_15-07-58 micro const.jpg"
RUTA_CSV = CARPETA_FOTOS / "circulos generados en ImagJ de los globulos rojos aproximados_muestra_microscopio_construido.csv"
CARPETA_SALIDA = Path(r"C:\BackUp\Facultad\Labo 7\env_labo7\26.11.11 resolucion y contraste\resultados")

# Escala de ESTA foto en um/px, medida con la grilla de 10 um (o con ImageJ). None = resultados solo en px.
UM_POR_PX = None
# Unidades del CSV de ImageJ: 1.0 si está en píxeles (micro construido); 96.0 si es el CSV del micro de
# mesa sin calibrar (el TIFF trae 96 dpi y ImageJ mide en pulgadas).
PX_POR_UNIDAD_CSV = 1.0

# En el canal azul los glóbulos y el fondo quedan casi iguales (Weber ~ 0 en la foto del 2026-10-29): no hay borde
# que medir. Se puede agregar "B" si cambia la tinción o la iluminación.
CANALES_A_MEDIR = ("gris", "G", "R")      # opciones: ver resolucion_contraste.CANALES
CANALES_CON_GRAFICO = ("gris", "R")       # se guarda una figura por glóbulo para estos canales
CONFIG = ConfigPerfiles(n_rayos=72)       # n_rayos = 2 * (cantidad de líneas por el centro)


# ============================================================
# PROCESAMIENTO
# ============================================================

COLUMNAS_CSV = [
    "globulo", "canal", "rayos_validos", "rayos_totales",
    "sigma_px", "sigma_dispersion_px", "sigma_error_mediana_px", "sigma_perfil_medio_px",
    "fwhm_px", "w1090_px", "mtf50_ciclos_px", "mtf10_ciclos_px", "mtf50_no_param_ciclos_px",
    "michelson", "michelson_dispersion", "weber_borde", "weber_borde_dispersion", "weber_interior",
    "nivel_interior", "nivel_fondo", "radio_px", "desvio_circulo_px",
    "um_por_px", "fwhm_um", "w1090_um", "mtf50_ciclos_um", "motivos_descarte", "advertencias",
]


def _fila_resumen(circulo, canal, resultado):
    """Una fila del CSV de resumen con lo medido en un glóbulo y un canal."""
    return {
        "globulo": circulo.indice, "canal": canal,
        "rayos_validos": resultado.n_validos, "rayos_totales": resultado.n_rayos,
        "sigma_px": resultado.sigma_mediana_px, "sigma_dispersion_px": resultado.sigma_std_px,
        "sigma_error_mediana_px": resultado.sigma_error_mediana_px, "sigma_perfil_medio_px": resultado.sigma_esf_media_px,
        "fwhm_px": resultado.fwhm_px, "w1090_px": resultado.w1090_px,
        "mtf50_ciclos_px": resultado.mtf50_ciclos_px, "mtf10_ciclos_px": resultado.mtf10_ciclos_px,
        "mtf50_no_param_ciclos_px": resultado.mtf50_no_param_ciclos_px,
        "michelson": resultado.michelson_mediana, "michelson_dispersion": resultado.michelson_std,
        "weber_borde": resultado.weber_borde_mediana, "weber_borde_dispersion": resultado.weber_borde_std,
        "weber_interior": resultado.weber_interior,
        "nivel_interior": resultado.nivel_interior, "nivel_fondo": resultado.nivel_fondo,
        "radio_px": resultado.circulo.radio_px, "desvio_circulo_px": resultado.circulo.rms_px,
        "um_por_px": resultado.um_por_px, "fwhm_um": resultado.fwhm_um, "w1090_um": resultado.w1090_um,
        "mtf50_ciclos_um": resultado.mtf50_ciclos_um,
        "motivos_descarte": "; ".join(f"{motivo}: {n}" for motivo, n in resultado.motivos_descarte.items()),
        "advertencias": " | ".join(resultado.advertencias),
    }


def _imprimir_tabla(filas):
    """Imprime el resumen por glóbulo y canal."""
    print(f"{'glob':>4} {'canal':>5} {'rayos':>7} {'sigma [px]':>16} {'FWHM':>7} {'w10-90':>7} {'MTF50':>7} "
          f"{'Michel.':>8} {'Weber b.':>8} {'Weber i.':>8}")
    for fila in filas:
        if fila.get("error"):
            print(f"{fila['globulo']:>4} {fila['canal']:>5}  ERROR: {fila['error']}")
            continue
        sigma = f"{fila['sigma_px']:.2f} ± {fila['sigma_dispersion_px']:.2f}" if fila["sigma_px"] == fila["sigma_px"] else "—"
        print(f"{fila['globulo']:>4} {fila['canal']:>5} {fila['rayos_validos']:>3}/{fila['rayos_totales']:<3} {sigma:>16} "
              f"{fila['fwhm_px']:7.2f} {fila['w1090_px']:7.2f} {fila['mtf50_ciclos_px']:7.4f} "
              f"{fila['michelson']:8.3f} {fila['weber_borde']:8.3f} {fila['weber_interior']:8.3f}")
        if fila["advertencias"]:
            print(f"       ! {fila['advertencias']}")


def procesar_imagen(ruta_imagen, ruta_csv, carpeta_salida, canales=CANALES_A_MEDIR, canales_con_grafico=CANALES_CON_GRAFICO,
                    config=CONFIG, um_por_px=UM_POR_PX, px_por_unidad_csv=PX_POR_UNIDAD_CSV):
    """Mide todos los glóbulos del CSV en los canales pedidos, guarda figuras y un CSV de resumen, imprime la tabla y devuelve las filas."""
    imagen = cv2.imread(str(ruta_imagen), cv2.IMREAD_COLOR)
    if imagen is None:
        raise FileNotFoundError(f"no se pudo leer la imagen: {ruta_imagen}")
    circulos = leer_circulos_imagej(ruta_csv, px_por_unidad_csv)
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    print(f"{Path(ruta_imagen).name}: {imagen.shape[1]}x{imagen.shape[0]} px, {len(circulos)} glóbulos, "
          f"canales {list(canales)}, {config.n_rayos} rayos, escala {um_por_px if um_por_px else 'sin dato (solo px)'}")

    filas = []
    for circulo in circulos:
        for canal in canales:
            try:
                resultado = medir_globulo(canal_de_trabajo(imagen, canal), circulo.centro_fila_px, circulo.centro_col_px,
                                          circulo.diametro_px / 2.0, config, um_por_px)
            except ValueError as error:
                filas.append({"globulo": circulo.indice, "canal": canal, "error": str(error)})
                continue
            filas.append(_fila_resumen(circulo, canal, resultado))
            if canal in canales_con_grafico:
                graficar_globulo(imagen, resultado, carpeta_salida / f"globulo_{circulo.indice}_{canal}.png",
                                 titulo=f"Glóbulo {circulo.indice}, canal {canal}  ({Path(ruta_imagen).name})")

    ruta_resumen = carpeta_salida / "resumen_resolucion_contraste.csv"
    with open(ruta_resumen, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS_CSV, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows([fila for fila in filas if not fila.get("error")])
    _imprimir_tabla(filas)
    print(f"\nResumen: {ruta_resumen}")
    return filas


# %%

if __name__ == "__main__":
    procesar_imagen(RUTA_IMAGEN, RUTA_CSV, CARPETA_SALIDA)


# %%
