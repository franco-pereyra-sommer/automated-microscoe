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

import math
import numbers

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
# Desde el 2026-11-09 ya no los usa busqueda_muestra: la búsqueda del foco detecta
# con un aumento relativo a una línea de base (ver detectar_aumento() abajo).
# Quedan acá, sueltos, hasta que se haga limpieza.

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


# ============================================================
# DETECCIÓN RELATIVA A UNA LÍNEA DE BASE (desde 2026-11-09)
# ============================================================
#
# Los umbrales absolutos de CRITERIOS_POR_DEFECTO dejaron de servir: a veces la
# primera foto, tomada lejos del foco, ya los superaba. Acá el umbral se arma en
# cada campo visual: se promedia cada métrica en las primeras fotos del barrido
# (la "línea de base", tomada lejos del foco) y se detecta cuando una métrica
# pasa de ese promedio en más de su propio aumento relativo ("x %", como fracción:
# 0.3 = +30 %). Los aumentos NO tienen valor por defecto: se miden (idea de
# Franco, 2026-11-09).
#
# Las tres funciones trabajan sobre números ya medidos, sin imagen ni hardware:
# la misma regla que usa la búsqueda del foco se puede probar a mano sobre las
# mediciones guardadas de un barrido.

def _entero(valor, nombre, minimo):
    if isinstance(valor, bool) or not isinstance(valor, numbers.Integral):
        raise TypeError(f"'{nombre}' tiene que ser un entero, no {valor!r}")

    if valor < minimo:
        raise ValueError(f"'{nombre}' tiene que ser >= {minimo}, no {valor}")


def validar_aumentos(aumentos, nombre_parametro="aumentos"):
    """
    Revisa un diccionario {métrica: aumento} sin medir nada. Levanta TypeError si no es un
    diccionario y ValueError si está vacío, nombra una métrica que no existe (ver
    FUNCIONES_METRICA) o tiene un aumento que no es un número finito mayor que 0.
    """

    if not isinstance(aumentos, dict):
        raise TypeError(
            f"'{nombre_parametro}' tiene que ser un diccionario {{métrica: aumento}}, "
            f"no {type(aumentos).__name__}"
        )

    if not aumentos:
        raise ValueError(
            f"'{nombre_parametro}' está vacío: hace falta al menos una métrica "
            f"({', '.join(FUNCIONES_METRICA)})"
        )

    desconocidas = [nombre for nombre in aumentos if nombre not in FUNCIONES_METRICA]

    if desconocidas:
        raise ValueError(
            f"'{nombre_parametro}': métricas que no existen: {desconocidas}. "
            f"Las disponibles son {list(FUNCIONES_METRICA)}"
        )

    for nombre, aumento in aumentos.items():
        if (
            isinstance(aumento, bool)
            or not isinstance(aumento, numbers.Real)
            or not math.isfinite(aumento)
            or aumento <= 0
        ):
            raise ValueError(
                f"'{nombre_parametro}[{nombre!r}]' tiene que ser un número mayor que 0 "
                f"(fracción: 0.3 = +30 %), no {aumento!r}"
            )


def linea_de_base(valores_por_foto, nombres, pasos_linea_base=5):
    """
    Promedio de cada métrica en las primeras 'pasos_linea_base' fotos: {nombre: promedio}.
    'valores_por_foto' es una lista de diccionarios {métrica: valor}, una por foto y en orden.
    Devuelve None si todavía no hay tantas fotos.
    """

    _entero(pasos_linea_base, "pasos_linea_base", 1)

    if len(valores_por_foto) < pasos_linea_base:
        return None

    primeras = valores_por_foto[:pasos_linea_base]

    return {
        nombre: sum(valores[nombre] for valores in primeras) / pasos_linea_base
        for nombre in nombres
    }


def detectar_aumento(
    valores_por_foto,
    aumentos,
    pasos_linea_base=5,
    fotos_seguidas=2,
    desde=0,
):
    """
    Dice si, en una lista de mediciones ya hechas, alguna métrica superó su línea de base por más de su aumento.

    'valores_por_foto': lista de diccionarios {métrica: valor} (pueden traer otras claves, como "z"),
    una por foto y en el orden del barrido. 'aumentos': {métrica: fracción} (ver validar_aumentos();
    solo se miran las métricas que figuren acá).

    La línea de base es el promedio de cada métrica en las primeras 'pasos_linea_base' fotos
    (linea_de_base()); esas fotos no se evalúan. El umbral de cada métrica es
    promedio * (1 + aumento) y se supera con un valor ESTRICTAMENTE mayor (con una línea de base
    en 0, cualquier valor mayor que 0 la supera). Se detecta cuando una métrica supera su umbral
    en 'fotos_seguidas' fotos seguidas; con una sola foto bastaría un pico de ruido.

    'desde': índice de la primera foto que se evalúa (para retomar después de descartar una señal
    falsa: las fotos anteriores no cuentan para las seguidas; la línea de base sigue siendo la misma).

    Devuelve None si no se detectó nada (o si todavía no hay línea de base). Si se detectó, un diccionario:
      "indice"     índice de la foto donde se completaron las seguidas
      "metrica"    la métrica elegida: si varias lo logran en esa foto, la que más pasó su umbral (valor / umbral)
      "valor"      su valor en esa foto
      "linea_base" {métrica: promedio} de las métricas de 'aumentos'
      "umbrales"   {métrica: umbral}
      "disparadas" todas las que lo lograron en esa foto
    """

    validar_aumentos(aumentos)
    _entero(fotos_seguidas, "fotos_seguidas", 1)
    _entero(desde, "desde", 0)

    base = linea_de_base(valores_por_foto, aumentos, pasos_linea_base)

    if base is None:
        return None

    umbrales = {nombre: base[nombre] * (1 + aumentos[nombre]) for nombre in aumentos}
    seguidas = {nombre: 0 for nombre in aumentos}

    for indice in range(max(desde, pasos_linea_base), len(valores_por_foto)):

        valores = valores_por_foto[indice]

        for nombre in aumentos:
            seguidas[nombre] = seguidas[nombre] + 1 if valores[nombre] > umbrales[nombre] else 0

        logradas = [nombre for nombre in aumentos if seguidas[nombre] >= fotos_seguidas]

        if logradas:

            def _cuanto_paso(nombre):
                if umbrales[nombre] <= 0:
                    return math.inf
                return valores[nombre] / umbrales[nombre]

            elegida = max(logradas, key=_cuanto_paso)

            return {
                "indice": indice,
                "metrica": elegida,
                "valor": valores[elegida],
                "linea_base": base,
                "umbrales": umbrales,
                "disparadas": logradas,
            }

    return None


# %%
