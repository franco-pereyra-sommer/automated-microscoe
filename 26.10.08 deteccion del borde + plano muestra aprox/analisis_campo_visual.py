#%%



"""
Análisis de la distribución espacial de la muestra dentro de un campo
visual: ¿el campo visual está ADENTRO de la muestra, en un BORDE, o
VACÍO?

Idea: binarizar la imagen en píxeles OSCUROS (células) y CLAROS (fondo)
con Otsu, y comparar los centros de masa de las dos clases.

  - Interior: células y fondo (los huecos entre células) están repartidos
    por todo el encuadre -> los dos centros de masa caen cerca uno del
    otro (y cerca del centro de la imagen): distancia ~ 0.
  - Borde: la muestra ocupa un lado del encuadre y el fondo vacío el otro
    -> los centros de masa se separan. En el caso límite (mitad y mitad)
    la distancia normalizada ronda 0.4.
  - Vacío: no hay dos clases reales, Otsu solo parte el ruido -> el
    contraste entre las dos clases es ~ 0 (y la distancia no significa
    nada: por eso el contraste se evalúa PRIMERO).

La distancia se normaliza por la diagonal de la imagen, así que no
depende de la resolución (se puede calibrar en QVGA y usar en otra).

IMPORTANTE: los umbrales que separan interior / borde / vacío hay que
calibrarlos con imágenes reales. Por eso clasificar_campo_visual() los
pide como argumentos OBLIGATORIOS: no hay valores por defecto que
parezcan calibrados sin estarlo. resumir_calibracion() imprime el
análisis de mínimos/máximos por clase y sugiere un umbral cuando las
clases se separan.
"""

# Agrego la raiz del proyecto
import sys
import os

os.environ["PYTHONPATH"] = r"C:\BackUp\Facultad\Labo 7\env_labo7"

sys.path.insert(0, os.environ["PYTHONPATH"])

import numpy as np
import cv2

from busqueda_muestra import _a_escala_de_grises, intensidad_fondo


# ============================================================
# CARACTERÍSTICAS DE UNA IMAGEN
# ============================================================

def _centro_de_masa(mascara):
    """(x, y) en píxeles del centro de masa de una máscara booleana,
    o None si la máscara está vacía."""

    m = cv2.moments(mascara.astype(np.uint8), binaryImage=True)

    if m["m00"] == 0:
        return None

    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def caracteristicas_borde(imagen, umbral=None):
    """
    Calcula, para una imagen (GRAYSCALE o color, el formato se deduce
    de su forma):

      - 'umbral': nivel de gris que separa oscuros de claros (Otsu por
        defecto; se puede forzar uno fijo con el parámetro 'umbral').
      - 'fraccion_oscura': fracción de píxeles oscuros.
      - 'com_oscuro', 'com_claro': centros de masa (x, y) en píxeles
        (x = columna, y = fila, como en la imagen), o None si esa
        clase quedó vacía.
      - 'distancia_px' y 'distancia_normalizada' (dividida por la
        diagonal de la imagen) entre los dos centros.
      - 'contraste': gris medio de los claros menos gris medio de los
        oscuros. ~0 si no hay estructura (campo vacío).
      - 'direccion_hacia_oscuro': versor (dx, dy) en coordenadas de
        IMAGEN que apunta del centro de masa claro al oscuro, o None.
        Cuidado: está en ejes de imagen, no de la impresora (ver el
        mapeo de ejes cámara-impresora antes de usarla para decidir
        hacia dónde seguir moviéndose).
    """

    gris = _a_escala_de_grises(imagen)

    if gris.dtype != np.uint8:
        gris = np.clip(gris, 0, 255).astype(np.uint8)

    alto, ancho = gris.shape
    diagonal = float(np.hypot(ancho, alto))

    if umbral is None:
        umbral, _ = cv2.threshold(
            gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    oscuros = gris <= umbral
    claros = ~oscuros

    com_oscuro = _centro_de_masa(oscuros)
    com_claro = _centro_de_masa(claros)

    if com_oscuro is None or com_claro is None:
        # Una sola clase: no hay estructura que medir.
        distancia_px = 0.0
        direccion = None
        contraste = 0.0
    else:
        dx = com_oscuro[0] - com_claro[0]
        dy = com_oscuro[1] - com_claro[1]
        distancia_px = float(np.hypot(dx, dy))
        direccion = (dx / distancia_px, dy / distancia_px) if distancia_px > 0 else None
        contraste = float(gris[claros].mean() - gris[oscuros].mean())

    return {
        "umbral": float(umbral),
        "fraccion_oscura": float(oscuros.mean()),
        "com_oscuro": com_oscuro,
        "com_claro": com_claro,
        "distancia_px": distancia_px,
        "distancia_normalizada": distancia_px / diagonal,
        "contraste": contraste,
        "direccion_hacia_oscuro": direccion,
    }


# ============================================================
# CLASIFICADOR (umbrales obligatorios: hay que calibrarlos)
# ============================================================

def clasificar_campo_visual(
    imagen,
    umbral_distancia_borde,
    umbral_contraste_vacio,
    umbral=None,
):
    """
    Devuelve (estado, caracteristicas) con estado en
    {"vacio", "borde", "interior"}:

      - "vacio":    contraste < umbral_contraste_vacio (no hay dos
                    clases reales; no se mira la distancia).
      - "borde":    hay estructura y distancia_normalizada >=
                    umbral_distancia_borde.
      - "interior": hay estructura y la distancia es menor.

    'umbral_distancia_borde' y 'umbral_contraste_vacio' NO tienen
    valor por defecto a propósito (ver el docstring del módulo).
    """

    c = caracteristicas_borde(imagen, umbral=umbral)

    if c["contraste"] < umbral_contraste_vacio:
        estado = "vacio"
    elif c["distancia_normalizada"] >= umbral_distancia_borde:
        estado = "borde"
    else:
        estado = "interior"

    return estado, c


# ============================================================
# AYUDAS PARA LA CALIBRACIÓN
# ============================================================

def sugerir_umbral(valores_menores, valores_mayores):
    """
    Dadas las mediciones de dos clases (la que se espera con valores
    más chicos y la que se espera con valores más grandes), devuelve
    (umbral_sugerido, margen) si se separan limpiamente — el punto
    medio entre el máximo de la primera y el mínimo de la segunda, y
    la mitad de la brecha — o (None, solapamiento) si se pisan, con
    'solapamiento' = max(primera) - min(segunda) (> 0).
    """

    maximo_chicos = float(np.max(valores_menores))
    minimo_grandes = float(np.min(valores_mayores))

    if maximo_chicos < minimo_grandes:
        return (
            (maximo_chicos + minimo_grandes) / 2,
            (minimo_grandes - maximo_chicos) / 2,
        )

    return None, maximo_chicos - minimo_grandes


def resumir_calibracion(rutas_por_clase, umbral=None):
    """
    'rutas_por_clase': {"interior": [rutas...], "borde": [...],
    "vacio": [...]} (cualquier subconjunto de esas tres clases).

    Imprime, por clase, mínimo / media / máximo de
    distancia_normalizada, contraste, fraccion_oscura e
    intensidad_fondo (la métrica ya calibrada, como referencia), y
    sugiere umbrales con sugerir_umbral() cuando se puede. Devuelve un
    diccionario {clase: {característica: [valores...]}} por si se
    quiere graficar.
    """

    caracteristicas_interes = (
        "distancia_normalizada",
        "contraste",
        "fraccion_oscura",
        "intensidad_fondo",
    )

    datos = {}

    for clase, rutas in rutas_por_clase.items():

        datos[clase] = {nombre: [] for nombre in caracteristicas_interes}
        datos[clase]["archivo"] = []

        for ruta in rutas:

            imagen = cv2.imread(str(ruta))

            if imagen is None:
                print(f"[AVISO] No se pudo leer {ruta}")
                continue

            c = caracteristicas_borde(imagen, umbral=umbral)

            datos[clase]["distancia_normalizada"].append(c["distancia_normalizada"])
            datos[clase]["contraste"].append(c["contraste"])
            datos[clase]["fraccion_oscura"].append(c["fraccion_oscura"])
            datos[clase]["intensidad_fondo"].append(float(intensidad_fondo(imagen)))
            datos[clase]["archivo"].append(str(ruta))

    print(f"{'clase':10s} {'n':>4s}  {'característica':22s} {'mín':>10s} {'media':>10s} {'máx':>10s}")

    for clase, d in datos.items():
        for nombre in caracteristicas_interes:
            v = np.array(d[nombre])
            if v.size == 0:
                continue
            print(
                f"{clase:10s} {v.size:4d}  {nombre:22s} "
                f"{v.min():10.4f} {v.mean():10.4f} {v.max():10.4f}"
            )

    # Umbrales sugeridos cuando están las clases necesarias
    if "interior" in datos and "borde" in datos:
        u, m = sugerir_umbral(
            datos["interior"]["distancia_normalizada"],
            datos["borde"]["distancia_normalizada"],
        )
        if u is None:
            print(f"\nDistancia interior/borde: las clases se SOLAPAN en {m:.4f}")
        else:
            print(f"\nUmbral de distancia interior/borde sugerido: {u:.4f} (± {m:.4f})")

    if "vacio" in datos and ("interior" in datos or "borde" in datos):
        con_muestra = []
        for clase in ("interior", "borde"):
            if clase in datos:
                con_muestra += datos[clase]["contraste"]
        u, m = sugerir_umbral(datos["vacio"]["contraste"], con_muestra)
        if u is None:
            print(f"Contraste vacío/con muestra: las clases se SOLAPAN en {m:.4f}")
        else:
            print(f"Umbral de contraste vacío/con muestra sugerido: {u:.4f} (± {m:.4f})")

    return datos



#%%



from pathlib import Path
from collections import Counter

# from analisis_campo_visual import resumir_calibracion


def listar_jpg(carpeta):
    """Todos los JPG de 'carpeta' y de todas sus subcarpetas."""
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {carpeta}")
    return sorted(
        p for p in carpeta.rglob("*")
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")
    )


carpeta_borde = Path(
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.10.08 centro masas + ajuste a un plano\centro de masas\borde"
)

carpeta_interior = Path(
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.10.08 centro masas + ajuste a un plano\centro de masas\interiror"
)

carpeta_vacio = Path(
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.10.08 centro masas + ajuste a un plano\centro de masas\vacio"
)

rutas_borde = listar_jpg(carpeta_borde)
rutas_interior = listar_jpg(carpeta_interior)
rutas_vacio = listar_jpg(carpeta_vacio)


# Control: cuántos JPG salieron de cada subcarpeta (tiene que coincidir con lo que esperás)
print(f"Total: {len(rutas_borde)} JPG")
for sub, n in sorted(Counter(p.parent.relative_to(carpeta_borde) for p in rutas_borde).items()):
    print(f"  {n:4d}  {sub}")

# Las rutas ya están listas para resumir_calibracion()
rutas_por_clase = {
    "borde": rutas_borde,
    "interior": rutas_interior,
    "vacio": rutas_vacio,
}
datos = resumir_calibracion(rutas_por_clase)



#%%


print(f"{datos = }")


# %%

ruta = r"C:\BackUp\Facultad\Labo 7\Fotos\26.10.08 centro masas + ajuste a un plano\plano\0024 +y (150.0, 153.45, 27.74) - 2026-10-08_13-52-43.jpg"

imagen = cv2.imread(str(ruta))

c = caracteristicas_borde(imagen, umbral=None)
# %%
