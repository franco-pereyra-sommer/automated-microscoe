# %%

"""
Lectura de los CSV que genera ImageJ al medir selecciones circulares (Analyze > Measure).

Los CSV traen, entre otras, las columnas X, Y (centro), Area y Feret (diámetro máximo). Los valores
vienen en las unidades de la imagen en ImageJ, que no siempre son píxeles:
  - micro construido: píxeles (los X, Y son enteros o .5).
  - micro de mesa (TIFF con resolución de 96 dpi, unidad "pulgada"): los valores salen en pulgadas, o sea
    en unidades de 96 px. Para ese CSV se pasa px_por_unidad=96 (o se calibra la escala en ImageJ antes
    de medir).

Convención de ImageJ: el píxel (i, j) cubre [i, i+1) x [j, j+1), así que su centro está en (i+0.5, j+0.5).
Los arrays de numpy se indexan desde el centro del píxel en 0, por eso a los centros se les resta 0.5.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path


# Restarle esto a X e Y de ImageJ da el índice (columna, fila) de numpy.
OFFSET_CENTRO_IMAGEJ_PX = 0.5


@dataclass(frozen=True)
class CirculoImageJ:
    """Un círculo marcado en ImageJ, ya en píxeles de numpy (fila, columna)."""
    indice: int
    centro_fila_px: float
    centro_col_px: float
    diametro_px: float


def leer_circulos_imagej(ruta_csv, px_por_unidad=1.0):
    """Lee un CSV de ImageJ y devuelve una lista de CirculoImageJ en píxeles de numpy (fila, columna)."""
    ruta_csv = Path(ruta_csv)
    with open(ruta_csv, newline="", encoding="utf-8-sig") as f:
        lector = csv.DictReader(f)
        filas = [{(k or "").strip(): v for k, v in fila.items()} for fila in lector]

    if not filas:
        raise ValueError(f"{ruta_csv.name}: el CSV no tiene filas.")
    for clave in ("X", "Y"):
        if clave not in filas[0]:
            raise ValueError(f"{ruta_csv.name}: falta la columna {clave} (tiene {list(filas[0])}).")
    if "Feret" not in filas[0] and "Area" not in filas[0]:
        raise ValueError(f"{ruta_csv.name}: hace falta la columna Feret o Area para conocer el diámetro.")

    circulos = []
    for posicion, fila in enumerate(filas, start=1):
        indice = int(fila[""]) if fila.get("", "").strip().isdigit() else posicion
        if "Feret" in fila:
            diametro = float(fila["Feret"])
        else:
            diametro = 2.0 * math.sqrt(float(fila["Area"]) / math.pi)
        circulos.append(CirculoImageJ(
            indice=indice,
            centro_fila_px=float(fila["Y"]) * px_por_unidad - OFFSET_CENTRO_IMAGEJ_PX,
            centro_col_px=float(fila["X"]) * px_por_unidad - OFFSET_CENTRO_IMAGEJ_PX,
            diametro_px=diametro * px_por_unidad,
        ))
    return circulos


# %%
