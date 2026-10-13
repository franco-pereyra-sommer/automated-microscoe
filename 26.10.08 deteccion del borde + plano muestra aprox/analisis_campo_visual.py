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

Si la imagen es a color también se calculan dos características
EXPLORATORIAS (para ver qué tanto separan las clases, sin umbrales
calibrados): 'saturacion_media' y 'fraccion_rojos'. Con una imagen en
escala de grises (2 dimensiones) valen None. OJO: dependen del balance de blancos y de
la exposición de la cámara; si están en automático, la cámara los
reajusta según lo que ve en cada foto y las comparaciones entre fotos
pueden engañar.

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


# ------------------------------------------------------------
# Características de color (exploratorias)
# ------------------------------------------------------------
# Parámetros PROVISORIOS de la "fracción de rojos": no están calibrados,
# son un punto de partida. En HSV de OpenCV, H va de 0 a 179 y el rojo
# está en H ~ 0 (y también cerca de 179, porque el tono es circular).

HUE_ROJO_TOLERANCIA = 15       # |H - 0| <= 15 (unos ±30° alrededor del rojo)
SATURACION_MINIMA_ROJO = 50    # 0..255: descarta píxeles casi grises
VALOR_MINIMO_ROJO = 40         # 0..255: descarta píxeles casi negros


def _es_gris(imagen):
    """True si la imagen es en escala de grises: tiene 2 dimensiones
    (así la entrega bytes_a_imagen() con pixformat GRAYSCALE, y
    cv2.imread(ruta, cv2.IMREAD_UNCHANGED) con un JPEG en grises).

    Una imagen de 3 canales NUNCA se considera gris, aunque los canales
    coincidan (por ejemplo, un campo vacío saturado en blanco): para sus
    características de color da saturación 0 y fracción de rojos 0, que
    es lo correcto. Ojo con cv2.imread(ruta) a secas: devuelve SIEMPRE 3
    canales, también para un JPEG en grises (queda con saturación 0)."""

    return imagen.ndim == 2 or imagen.shape[2] == 1


def _a_hsv(imagen):
    """HSV (OpenCV: H 0..179, S 0..255, V 0..255) de una imagen BGR."""

    if imagen.dtype != np.uint8:
        imagen = np.clip(imagen, 0, 255).astype(np.uint8)

    return cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)


def saturacion_media(imagen):
    """Saturación media (HSV) de la imagen, normalizada a 0..1, o None
    si la imagen es en escala de grises. La imagen a color es BGR (así
    la entregan bytes_a_imagen() y cv2.imread())."""

    if _es_gris(imagen):
        return None

    return float(_a_hsv(imagen)[..., 1].mean() / 255.0)


def fraccion_rojos(
    imagen,
    tolerancia_hue=HUE_ROJO_TOLERANCIA,
    saturacion_minima=SATURACION_MINIMA_ROJO,
    valor_minimo=VALOR_MINIMO_ROJO,
):
    """
    Fracción (0..1) de píxeles "rojos": tono dentro de ±'tolerancia_hue'
    del rojo (en unidades de H de OpenCV, 0..179), con saturación >=
    'saturacion_minima' y valor (brillo) >= 'valor_minimo'. None si la
    imagen es en escala de grises.
    """

    if _es_gris(imagen):
        return None

    hsv = _a_hsv(imagen)

    hue = hsv[..., 0].astype(np.int32)
    distancia_al_rojo = np.minimum(hue, 180 - hue)

    rojo = (
        (distancia_al_rojo <= tolerancia_hue)
        & (hsv[..., 1] >= saturacion_minima)
        & (hsv[..., 2] >= valor_minimo)
    )

    return float(rojo.mean())


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
      - 'saturacion_media' y 'fraccion_rojos': características de color
        exploratorias (ver saturacion_media() y fraccion_rojos()); None
        si la imagen es en escala de grises.
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
        "saturacion_media": saturacion_media(imagen),
        "fraccion_rojos": fraccion_rojos(imagen),
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


CARACTERISTICAS_RESUMEN = (
    "distancia_normalizada",
    "contraste",
    "fraccion_oscura",
    "intensidad_fondo",
    "fraccion_rojos",
    "saturacion_media",
)

ORDEN_CLASES = ("interior", "borde", "vacio")


def _clases_ordenadas(datos):
    """Las clases conocidas en el orden interior, borde, vacio y después
    cualquier otra que haya."""

    conocidas = [c for c in ORDEN_CLASES if c in datos]
    otras = [c for c in datos if c not in ORDEN_CLASES]

    return conocidas + otras


def _valores(datos, clase, nombre):
    """Valores de una característica en una clase, sin los NaN (las
    imágenes en escala de grises no tienen características de color)."""

    v = np.asarray(datos[clase][nombre], dtype=float)

    return v[~np.isnan(v)]


def resumir_calibracion(rutas_por_clase, umbral=None, imprimir=True):
    """
    'rutas_por_clase': {"interior": [rutas...], "borde": [...],
    "vacio": [...]} (cualquier subconjunto de esas tres clases).

    Calcula las características de cada imagen y, si 'imprimir' es True,
    llama a imprimir_resumen_calibracion() (una tabla por característica
    con las clases lado a lado, y la separación entre clases). Devuelve
    un diccionario {clase: {característica: [valores...], "archivo":
    [...]}} por si se quiere graficar; las características de color de
    una imagen en escala de grises valen NaN.
    """

    datos = {}

    for clase, rutas in rutas_por_clase.items():

        datos[clase] = {nombre: [] for nombre in CARACTERISTICAS_RESUMEN}
        datos[clase]["archivo"] = []

        for ruta in rutas:

            # IMREAD_UNCHANGED: un JPEG en escala de grises vuelve de 2
            # dimensiones (y sus características de color quedan en NaN)
            # y uno a color de 3 canales.
            imagen = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)

            if imagen is None:
                print(f"[AVISO] No se pudo leer {ruta}")
                continue

            if imagen.ndim == 3 and imagen.shape[2] == 4:
                imagen = imagen[..., :3]

            c = caracteristicas_borde(imagen, umbral=umbral)

            c["intensidad_fondo"] = float(intensidad_fondo(imagen))

            for nombre in CARACTERISTICAS_RESUMEN:
                valor = c[nombre]
                datos[clase][nombre].append(np.nan if valor is None else float(valor))

            datos[clase]["archivo"].append(str(ruta))

    if imprimir:
        imprimir_resumen_calibracion(datos)

    return datos


def imprimir_resumen_calibracion(datos):
    """
    Imprime lo que devuelve resumir_calibracion(), ordenado POR
    CARACTERÍSTICA (en cada una, las clases una debajo de la otra para
    compararlas), y después, para cada par de clases, qué tan bien las
    separa cada característica:

      - "SE SEPARAN": el máximo de la clase de valores más bajos queda
        por debajo del mínimo de la otra; se da el umbral (punto medio) y
        el margen (± la mitad de la brecha).
      - "se solapan": se da la zona de solapamiento [mín de la clase
        alta, máx de la clase baja] y cuántas imágenes de cada clase
        caen adentro, para ver si son pocas (¿mal clasificadas a mano?)
        o si las clases están realmente mezcladas.
    """

    clases = _clases_ordenadas(datos)

    print("\n" + "=" * 72)
    print("CARACTERÍSTICAS POR CLASE")
    print("=" * 72)

    for nombre in CARACTERISTICAS_RESUMEN:

        print(f"\n{nombre}")

        hay_algun_dato = False

        for clase in clases:
            if _valores(datos, clase, nombre).size > 0:
                hay_algun_dato = True

        if not hay_algun_dato:
            print("    (sin datos: ¿las imágenes están en escala de grises?)")
            continue

        print(f"    {'clase':10s} {'n':>5s} {'mín':>10s} {'media':>10s} {'máx':>10s}")

        for clase in clases:

            v = _valores(datos, clase, nombre)

            if v.size == 0:
                print(f"    {clase:10s} {0:5d} {'(sin datos)':>32s}")
                continue

            print(
                f"    {clase:10s} {v.size:5d} "
                f"{v.min():10.4f} {v.mean():10.4f} {v.max():10.4f}"
            )

    print("\n" + "=" * 72)
    print("SEPARACIÓN ENTRE CLASES")
    print("=" * 72)
    print(
        "(en cada par, primero va la clase de valores más bajos; umbral = "
        "punto medio entre su máx y el mín de la otra)"
    )

    for k, a in enumerate(clases):

        for b in clases[k + 1:]:

            print(f"\n{a} vs {b}")

            for nombre in CARACTERISTICAS_RESUMEN:

                va = _valores(datos, a, nombre)
                vb = _valores(datos, b, nombre)

                if va.size == 0 or vb.size == 0:
                    continue

                if va.mean() <= vb.mean():
                    bajo, alto, nombre_bajo, nombre_alto = va, vb, a, b
                else:
                    bajo, alto, nombre_bajo, nombre_alto = vb, va, b, a

                orden = f"{nombre_bajo} < {nombre_alto}"

                umbral_sugerido, margen = sugerir_umbral(bajo, alto)

                if umbral_sugerido is not None:
                    detalle = f"SE SEPARAN: umbral {umbral_sugerido:.4f} (± {margen:.4f})"
                else:
                    desde = float(alto.min())
                    hasta = float(bajo.max())
                    k_bajo = int((bajo >= desde).sum())
                    k_alto = int((alto <= hasta).sum())
                    detalle = (
                        f"se solapan en [{desde:.4f}, {hasta:.4f}]: "
                        f"{k_bajo} de {bajo.size} {nombre_bajo} y "
                        f"{k_alto} de {alto.size} {nombre_alto} caen ahí"
                    )

                print(f"    {nombre:22s} {orden:18s} {detalle}")

    return None



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


imprimir_resumen_calibracion(datos)


# %%

ruta = r"C:\BackUp\Facultad\Labo 7\Fotos\26.10.08 centro masas + ajuste a un plano\plano\0024 +y (150.0, 153.45, 27.74) - 2026-10-08_13-52-43.jpg"

imagen = cv2.imread(str(ruta))

c = caracteristicas_borde(imagen, umbral=None)
# %%
