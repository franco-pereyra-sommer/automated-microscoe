# %%

"""
Métricas de imagen del proyecto: funciones que reciben una imagen (el array
de numpy que entrega camara_openmv.bytes_a_imagen(): 2D si es GRAYSCALE,
3D en orden BGR si es color) y devuelven números. No usan la cámara ni la
impresora: es la categoría "procesamiento de imagen sin hardware" del
árbol de decisión de REGLAS_CODIGO.txt.

Agregado como módulo base el 2026-10-27. Antes vivía en busqueda_muestra.py:
_a_escala_de_grises, intensidad_fondo, laplacian_variance, tenengrad,
varianza_intensidad, FUNCIONES_METRICA, CRITERIOS_POR_DEFECTO y
evaluar_criterios (antes _evaluar_criterios, con guion bajo: ahora es
pública porque la importa otro módulo).
"""

import cv2


# ============================================================
# ESCALA DE GRISES (soporta GRAYSCALE y color, sin necesidad de
# saber de antemano qué pixformat se usó)
# ============================================================

def _a_escala_de_grises(imagen):
    """
    Devuelve la imagen en escala de grises, sea cual sea el formato
    con el que se haya capturado:
      - 2 dimensiones (H, W): ya viene en escala de grises
        (pixformat GRAYSCALE en una captura en vivo vía
        bytes_a_imagen()), se devuelve tal cual.
      - 3 dimensiones (H, W, 3): viene a color (RGB565, JPEG a color,
        o cualquier imagen cargada desde disco con cargar_imagen(),
        que cv2.imread() siempre fuerza a 3 canales), se convierte
        con cv2.cvtColor().

    Así las 4 métricas de abajo funcionan con cualquier pixformat sin
    necesidad de que quien las llama le pase el formato a mano — el
    array ya "sabe" si es color o no por su propia forma.
    """

    if imagen.ndim == 2:
        return imagen

    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)


# ============================================================
# MÉTRICAS
# ============================================================

def intensidad_fondo(imagen):

    gray = _a_escala_de_grises(imagen)

    return 255 - gray.mean()


def laplacian_variance(imagen):

    gray = _a_escala_de_grises(imagen)

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    return laplacian.var()


def tenengrad(imagen):

    gray = _a_escala_de_grises(imagen)

    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    magnitud = sobel_x**2 + sobel_y**2

    return magnitud.mean()


def varianza_intensidad(imagen):

    gray = _a_escala_de_grises(imagen)

    return gray.var()


# ============================================================
# MÉTRICAS DISPONIBLES PARA EL BARRIDO GRUESO / CONFIRMACIÓN
# ============================================================

FUNCIONES_METRICA = {
    "laplacian": laplacian_variance,
    "tenengrad": tenengrad,
    "varianza": varianza_intensidad,
}

CRITERIOS_POR_DEFECTO = {
    "laplacian": 12,
    "tenengrad": 200,
    "varianza": 30,
} # Por alguna razon dejo de valer

# CRITERIOS_POR_DEFECTO = {
#     "laplacian": 6.35,
#     "tenengrad": 70,
#     "varianza": 10.35,
# }


# ============================================================
# EVALUAR CRITERIOS OR DE UNA IMAGEN
# ============================================================

def evaluar_criterios(imagen, criterios):
    """
    Devuelve (valores, disparados):
      valores: {nombre_metrica: valor_calculado}
      disparados: lista de nombres de métrica que superaron su umbral
    """

    valores = {
        nombre: FUNCIONES_METRICA[nombre](imagen)
        for nombre in criterios
    }

    disparados = [
        nombre
        for nombre, umbral in criterios.items()
        if valores[nombre] >= umbral
    ]

    return valores, disparados


# %%
