# %%

"""
Perfiles de borde de un objeto redondo (un glóbulo rojo) medidos por rayos desde su centro.

Qué hace, en orden:
  1. refinar_centro_y_radio(): a partir de un círculo aproximado (el de ImageJ) busca en cada rayo dónde
     está el borde (máximo del gradiente) y ajusta una circunferencia con rechazo de atípicos.
  2. medir_rayos(): por cada rayo desde el centro ajusta SU PROPIO borde con una función error (ERF) y
     decide si el rayo es válido (no choca con un vecino, el ajuste es bueno, hay fondo plano después).
  3. esf_alineada(): con los rayos válidos, un perfil de borde medio con cada rayo alineado en su borde.

Por qué se ajusta cada rayo por separado en vez de promediar los perfiles y ajustar el promedio (como en
el programa del pelo): si el borde no es perfectamente recto (o, acá, si el glóbulo no es perfectamente
redondo o el centro está corrido), promediar sin alinear emborrona el borde y se mide un ancho mayor que
el real. Con imágenes sintéticas, un error de 2 grados en el eje inflaba un sigma de 3.0 px a 5.6 px;
ajustando cada perfil y tomando la mediana daba 3.02 px. Además, la dispersión entre rayos es una
incertidumbre real, mientras que la del ajuste del perfil promedio solo refleja el ruido.

Modelo del borde: I(r) = A + B * erf((r - r0) / (sqrt(2) * sigma)). Es el perfil de un escalón visto por un
sistema con respuesta gaussiana de desvío sigma, medido hacia afuera desde el centro:
  - interior del objeto: A - B;  exterior (fondo): A + B;  r0 = posición del borde.

Sin cámara ni impresora ni módulos del proyecto (categoría "imagen pura" de REGLAS_CODIGO.txt).
Convenciones: imágenes 2D (float); posiciones en (fila, columna) en píxeles de numpy; los ángulos van de
la columna hacia la fila (dirección del rayo = (cos, sin) en (columna, fila)) y no tienen relación con los
ejes x e y de la impresora.
"""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.optimize import curve_fit
from scipy.special import erf


# Motivos de descarte de un rayo (claves de ResultadoGlobulo.motivos_descarte).
MOTIVO_SIN_BORDE = "no se encontró el borde"
MOTIVO_AJUSTE = "falló el ajuste"
MOTIVO_RECORTADO = "ventana recortada por el borde de la imagen"
MOTIVO_SIGMA_LIMITE = "sigma en el límite permitido"
MOTIVO_SIN_CONTRASTE = "borde sin contraste"
MOTIVO_RESIDUO = "mal ajuste (residuo alto)"
MOTIVO_MESETA_CORTA = "fondo exterior demasiado corto"
MOTIVO_VECINO = "el fondo exterior no es plano (vecino o suciedad)"


@dataclass(frozen=True)
class ConfigPerfiles:
    """Parámetros de la medición por rayos. Las longitudes que dependen del tamaño del glóbulo van como fracción del radio."""
    n_rayos: int = 72                          # rayos desde el centro (2 por cada "línea por el centro")
    paso_px: float = 1.0                       # separación entre muestras a lo largo del rayo
    factor_largo: float = 2.2                  # largo del rayo = factor_largo * radio
    suavizado_borde_px: float = 1.5            # solo para ubicar el borde por gradiente (el ajuste usa los datos sin suavizar)
    rango_busqueda_borde: float = 0.35         # el borde se busca en radio * (1 +- esto)
    fraccion_ventana_min: float = 0.4          # semiventana de ajuste mínima (fracción del radio)
    fraccion_ventana_max: float = 0.8          # semiventana de ajuste máxima (fracción del radio)
    sigmas_en_ventana: float = 4.0             # la segunda pasada abre la ventana a esta cantidad de sigmas
    sigma_inicial_px: float = 3.0
    sigma_min_px: float = 0.3
    fraccion_sigma_max: float = 0.5            # sigma máximo = esto * radio
    sigmas_hasta_meseta: float = 2.5           # el fondo exterior se mira desde r0 + esto * sigma ...
    fraccion_margen_meseta: float = 0.15       # ... hasta el final de la ventana de ajuste + esto (fracción del radio)
    fraccion_meseta_min: float = 0.1           # largo mínimo del fondo exterior (fracción del radio)
    suavizado_meseta_px: float = 2.0
    tolerancia_meseta: float = 0.15            # variación máxima del fondo exterior, como fracción del salto 2|B|
    tolerancia_residuo: float = 0.08           # RMS máximo del ajuste, como fracción del salto 2|B|
    amplitud_minima: float = 6.0               # salto mínimo 2|B| en unidades de la imagen (niveles de gris)
    iteraciones_centro: int = 6                # máximo de iteraciones al refinar el círculo
    tolerancia_convergencia_px: float = 0.25   # se deja de iterar cuando el centro y el radio cambian menos que esto
    k_mad_circulo: float = 3.0                 # rechazo de atípicos al ajustar la circunferencia
    tolerancia_min_circulo_px: float = 1.0


@dataclass(frozen=True)
class CirculoRefinado:
    """Centro y radio del glóbulo después de refinar el círculo aproximado."""
    centro_fila_px: float
    centro_col_px: float
    radio_px: float
    rms_px: float            # qué tanto se aparta el borde de una circunferencia (no circularidad + ruido)
    n_puntos: int
    n_inliers: int
    polaridad: int           # +1: el exterior es más claro que el interior; -1: más oscuro


@dataclass(frozen=True)
class AjusteBorde:
    """Resultado de ajustar un borde con la ERF."""
    nivel_medio: float       # A
    amplitud: float          # B (el interior vale A - B y el exterior A + B)
    r_borde_px: float        # r0
    sigma_px: float
    err_sigma_px: float      # del ajuste (solo ruido); nan si no se pudo estimar
    rms_residuo: float
    semiventana_px: float


@dataclass
class RayoMedido:
    """Un rayo: el perfil muestreado, su ajuste (si lo hubo) y si se lo da por válido."""
    angulo_rad: float
    r: np.ndarray
    perfil: np.ndarray
    ajuste: AjusteBorde | None
    valido: bool
    motivo: str              # "" si es válido


# ============================================================
# MUESTREO
# ============================================================

def perfil_a_lo_largo_del_rayo(imagen, centro_fila, centro_col, angulo_rad, largo_px, paso_px):
    """Muestrea la imagen sobre un rayo desde el centro (interpolación bilineal) y devuelve (r, perfil).

    Solo devuelve las muestras que caen dentro de la imagen. La interpolación bilineal agrega un
    desenfoque de unas décimas de píxel: no importa si sigma >> 1 px; con sigma de ~1 px o menos el
    resultado ya está limitado por el muestreo.
    """
    r = np.arange(0.0, largo_px + 1e-9, paso_px)
    fila = centro_fila + r * np.sin(angulo_rad)
    col = centro_col + r * np.cos(angulo_rad)
    alto, ancho = imagen.shape
    dentro = (fila >= 0) & (fila <= alto - 1) & (col >= 0) & (col <= ancho - 1)
    r, fila, col = r[dentro], fila[dentro], col[dentro]
    perfil = map_coordinates(imagen, [fila, col], order=1, mode="nearest")
    return r, perfil


def _borde_por_gradiente(r, perfil, r_central, semiancho_px, polaridad, suavizado_px, paso_px):
    """Posición del mayor gradiente (en el sentido de la polaridad) dentro de r_central +- semiancho; None si no hay muestras."""
    suave = gaussian_filter1d(perfil, suavizado_px / paso_px, mode="nearest")
    gradiente = np.gradient(suave, r)
    indices = np.where((r >= r_central - semiancho_px) & (r <= r_central + semiancho_px))[0]
    if indices.size < 3:
        return None
    return float(r[indices[np.argmax(polaridad * gradiente[indices])]])


def polaridad_del_borde(imagen, centro_fila, centro_col, radio_px):
    """Dice si el exterior del glóbulo es más claro (+1) o más oscuro (-1) que su interior.

    Compara la mediana del disco interior (r < 0.5 radio) con los percentiles 10 y 90 de un anillo
    exterior (1.15 a 1.6 radios): el que más se aparta del interior marca el fondo. Hace falta el
    percentil y no la mediana porque en una zona densa la mayor parte del anillo son vecinos.
    """
    alto, ancho = imagen.shape
    m = int(np.ceil(1.7 * radio_px))
    f0, f1 = max(0, int(centro_fila) - m), min(alto, int(centro_fila) + m + 1)
    c0, c1 = max(0, int(centro_col) - m), min(ancho, int(centro_col) + m + 1)
    recorte = imagen[f0:f1, c0:c1]
    filas, cols = np.ogrid[f0:f1, c0:c1]
    distancia = np.hypot(filas - centro_fila, cols - centro_col)
    interior = recorte[distancia < 0.5 * radio_px]
    anillo = recorte[(distancia > 1.15 * radio_px) & (distancia < 1.6 * radio_px)]
    if interior.size == 0 or anillo.size == 0:
        raise ValueError("el glóbulo está demasiado cerca del borde de la imagen para decidir la polaridad.")
    referencia = np.median(interior)
    d_alto = np.percentile(anillo, 90) - referencia
    d_bajo = np.percentile(anillo, 10) - referencia
    dominante = d_alto if abs(d_alto) >= abs(d_bajo) else d_bajo
    return 1 if dominante > 0 else -1


# ============================================================
# CENTRO Y RADIO
# ============================================================

def _circulo_minimos_cuadrados(filas, cols):
    """Circunferencia por mínimos cuadrados algebraicos (Kåsa). Devuelve (centro_fila, centro_col, radio)."""
    sistema = np.column_stack([2.0 * cols, 2.0 * filas, np.ones_like(cols)])
    termino = cols ** 2 + filas ** 2
    (a, b, k), *_ = np.linalg.lstsq(sistema, termino, rcond=None)
    return float(b), float(a), float(np.sqrt(k + a * a + b * b))


def _ajustar_circulo_robusto(filas, cols, k_mad, tolerancia_min_px, iteraciones=4):
    """Circunferencia que ignora los puntos atípicos (un vecino que tapa el borde, una mancha)."""
    usar = np.ones(len(filas), dtype=bool)
    for _ in range(iteraciones):
        cf, cc, radio = _circulo_minimos_cuadrados(filas[usar], cols[usar])
        residuo = np.hypot(filas - cf, cols - cc) - radio
        mad = np.median(np.abs(residuo[usar] - np.median(residuo[usar])))
        tolerancia = max(k_mad * 1.4826 * mad, tolerancia_min_px)
        nuevo = np.abs(residuo) <= tolerancia
        if nuevo.sum() < 6 or np.array_equal(nuevo, usar):
            break
        usar = nuevo
    cf, cc, radio = _circulo_minimos_cuadrados(filas[usar], cols[usar])
    residuo = np.hypot(filas - cf, cols - cc) - radio
    return cf, cc, radio, float(np.sqrt(np.mean(residuo[usar] ** 2))), int(usar.sum())


def _radio_por_perfil_mediano(imagen, centro_fila, centro_col, radio_aprox_px, polaridad, config):
    """Radio del borde según el perfil radial MEDIANO (mediana sobre los ángulos de la intensidad a cada distancia).

    La mediana aguanta que los vecinos tapen menos de la mitad de los ángulos, y no depende de que el
    círculo aproximado esté cerca del borde (en una foto desenfocada el círculo marcado a ojo puede
    quedar 1 o 2 sigmas por dentro del borde real). Busca en 0.4 a 2 veces el radio aproximado.
    """
    largo = 2.5 * radio_aprox_px
    r_comun = np.arange(0.0, largo + 1e-9, config.paso_px)
    perfiles = np.full((config.n_rayos, r_comun.size), np.nan)
    for i, angulo in enumerate(np.linspace(0.0, 2.0 * np.pi, config.n_rayos, endpoint=False)):
        r, perfil = perfil_a_lo_largo_del_rayo(imagen, centro_fila, centro_col, angulo, largo, config.paso_px)
        perfiles[i, :r.size] = perfil
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mediano = np.nanmedian(perfiles, axis=0)
    ok = np.isfinite(mediano)
    mediano = np.interp(r_comun, r_comun[ok], mediano[ok])
    gradiente = np.gradient(gaussian_filter1d(mediano, config.suavizado_borde_px / config.paso_px, mode="nearest"), r_comun)
    indices = np.where((r_comun >= 0.4 * radio_aprox_px) & (r_comun <= 2.0 * radio_aprox_px))[0]
    return float(r_comun[indices[np.argmax(polaridad * gradiente[indices])]])


def refinar_centro_y_radio(imagen, centro_fila, centro_col, radio_aprox_px, config=ConfigPerfiles(), polaridad=None):
    """Ajusta una circunferencia al borde del glóbulo partiendo del círculo aproximado y devuelve un CirculoRefinado.

    Primero toma el radio del perfil radial mediano (robusto a vecinos y a un círculo aproximado mal
    puesto). Después cada rayo aporta un punto (el máximo del gradiente cerca de ese radio), la
    circunferencia se ajusta con rechazo de atípicos y se repite desde el centro nuevo hasta que el
    centro y el radio dejan de moverse (máximo config.iteraciones_centro veces).
    """
    imagen = np.asarray(imagen, dtype=float)
    if polaridad is None:
        polaridad = polaridad_del_borde(imagen, centro_fila, centro_col, radio_aprox_px)
    cf, cc = float(centro_fila), float(centro_col)
    radio = _radio_por_perfil_mediano(imagen, cf, cc, float(radio_aprox_px), polaridad, config)
    angulos = np.linspace(0.0, 2.0 * np.pi, config.n_rayos, endpoint=False)

    for _ in range(config.iteraciones_centro):
        filas, cols = [], []
        semiancho = config.rango_busqueda_borde * radio
        for angulo in angulos:
            r, perfil = perfil_a_lo_largo_del_rayo(imagen, cf, cc, angulo, config.factor_largo * radio, config.paso_px)
            r_borde = _borde_por_gradiente(r, perfil, radio, semiancho, polaridad, config.suavizado_borde_px, config.paso_px)
            # un máximo pegado al límite de la búsqueda suele ser una pendiente que sigue, no un borde
            if r_borde is None or abs(r_borde - radio) >= 0.98 * semiancho:
                continue
            filas.append(cf + r_borde * np.sin(angulo))
            cols.append(cc + r_borde * np.cos(angulo))
        if len(filas) < 8:
            raise ValueError(f"solo {len(filas)} rayos encontraron borde cerca del círculo aproximado: no alcanza para refinarlo.")
        cf_nuevo, cc_nuevo, radio_nuevo, rms, n_inliers = _ajustar_circulo_robusto(
            np.array(filas), np.array(cols), config.k_mad_circulo, config.tolerancia_min_circulo_px)
        cambio = max(abs(cf_nuevo - cf), abs(cc_nuevo - cc), abs(radio_nuevo - radio))
        cf, cc, radio = cf_nuevo, cc_nuevo, radio_nuevo
        if cambio < config.tolerancia_convergencia_px:
            break
    return CirculoRefinado(cf, cc, radio, rms, len(filas), n_inliers, polaridad)


# ============================================================
# AJUSTE DE UN BORDE
# ============================================================

def modelo_erf(r, nivel_medio, amplitud, r_borde, sigma):
    """Perfil de un escalón visto con respuesta gaussiana: A + B * erf((r - r0) / (sqrt(2) sigma))."""
    return nivel_medio + amplitud * erf((r - r_borde) / (np.sqrt(2.0) * sigma))


def ajustar_borde_erf(r, perfil, r_borde_inicial, semiventana_px, polaridad, sigma_inicial_px, sigma_min_px, sigma_max_px):
    """Ajusta la ERF a la parte del perfil dentro de r0 +- semiventana y devuelve un AjusteBorde (None si falla).

    El signo de B queda fijado por la polaridad y sigma queda acotado a (sigma_min, sigma_max). Sin eso
    el ajuste puede caer en la rama de sigma negativo o colapsar a un escalón (en pruebas sintéticas,
    2 de 15 ajustes del programa del pelo daban sigma de 0.1 a 0.2 px con covarianza infinita).
    """
    seleccion = (r >= r_borde_inicial - semiventana_px) & (r <= r_borde_inicial + semiventana_px)
    x, y = r[seleccion], perfil[seleccion]
    if x.size < 8:
        return None
    B0 = polaridad * (y.max() - y.min()) / 2.0
    p0 = [float(y.mean()), float(B0), float(np.clip(r_borde_inicial, x.min(), x.max())),
          float(np.clip(sigma_inicial_px, sigma_min_px * 1.5, sigma_max_px * 0.9))]
    if polaridad > 0:
        limites = ([-np.inf, 0.0, x.min(), sigma_min_px], [np.inf, np.inf, x.max(), sigma_max_px])
    else:
        limites = ([-np.inf, -np.inf, x.min(), sigma_min_px], [np.inf, 0.0, x.max(), sigma_max_px])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parametros, covarianza = curve_fit(modelo_erf, x, y, p0=p0, bounds=limites)
    except (RuntimeError, ValueError):
        return None
    err = np.sqrt(np.diag(covarianza)) if np.all(np.isfinite(covarianza)) else np.full(4, np.nan)
    rms = float(np.sqrt(np.mean((y - modelo_erf(x, *parametros)) ** 2)))
    return AjusteBorde(float(parametros[0]), float(parametros[1]), float(parametros[2]), float(parametros[3]),
                       float(err[3]), rms, float(semiventana_px))


# ============================================================
# UN RAYO Y TODOS LOS RAYOS
# ============================================================

def medir_rayo(imagen, centro_fila, centro_col, radio_px, angulo_rad, polaridad, config=ConfigPerfiles()):
    """Mide el borde sobre un rayo y decide si es válido; devuelve un RayoMedido.

    Dos pasadas: la primera con ventana chica para tener un sigma aproximado, y la segunda con la
    ventana abierta a config.sigmas_en_ventana sigmas. Un rayo se descarta si el borde no está, el
    ajuste falla o queda en un límite, no hay salto, el residuo es alto o el fondo exterior no es plano
    (un vecino, una mancha). Los motivos son las constantes MOTIVO_* de este módulo.
    """
    paso = config.paso_px
    r, perfil = perfil_a_lo_largo_del_rayo(imagen, centro_fila, centro_col, angulo_rad, config.factor_largo * radio_px, paso)

    def descartado(motivo, ajuste=None):
        return RayoMedido(angulo_rad, r, perfil, ajuste, False, motivo)

    semiancho = config.rango_busqueda_borde * radio_px
    r_borde0 = _borde_por_gradiente(r, perfil, radio_px, semiancho, polaridad, config.suavizado_borde_px, paso)
    if r_borde0 is None or abs(r_borde0 - radio_px) >= 0.98 * semiancho:
        return descartado(MOTIVO_SIN_BORDE)

    sigma_max = config.fraccion_sigma_max * radio_px
    semiventana = config.fraccion_ventana_min * radio_px
    ajuste = ajustar_borde_erf(r, perfil, r_borde0, semiventana, polaridad,
                               config.sigma_inicial_px, config.sigma_min_px, sigma_max)
    if ajuste is None:
        return descartado(MOTIVO_AJUSTE)
    semiventana = float(np.clip(config.sigmas_en_ventana * ajuste.sigma_px, semiventana, config.fraccion_ventana_max * radio_px))
    ajuste = ajustar_borde_erf(r, perfil, ajuste.r_borde_px, semiventana, polaridad,
                               ajuste.sigma_px, config.sigma_min_px, sigma_max)
    if ajuste is None:
        return descartado(MOTIVO_AJUSTE)

    if r[-1] < ajuste.r_borde_px + semiventana - paso or r[0] > ajuste.r_borde_px - semiventana + paso:
        return descartado(MOTIVO_RECORTADO, ajuste)
    if ajuste.sigma_px >= 0.99 * sigma_max or ajuste.sigma_px <= 1.01 * config.sigma_min_px:
        return descartado(MOTIVO_SIGMA_LIMITE, ajuste)
    salto = 2.0 * abs(ajuste.amplitud)
    if salto < config.amplitud_minima:
        return descartado(MOTIVO_SIN_CONTRASTE, ajuste)
    if ajuste.rms_residuo > config.tolerancia_residuo * salto:
        return descartado(MOTIVO_RESIDUO, ajuste)

    inicio_meseta = ajuste.r_borde_px + config.sigmas_hasta_meseta * ajuste.sigma_px
    fin_meseta = min(r[-1], ajuste.r_borde_px + semiventana + config.fraccion_margen_meseta * radio_px)
    if fin_meseta - inicio_meseta < config.fraccion_meseta_min * radio_px:
        return descartado(MOTIVO_MESETA_CORTA, ajuste)
    meseta = gaussian_filter1d(perfil[(r >= inicio_meseta) & (r <= fin_meseta)], config.suavizado_meseta_px / paso, mode="nearest")
    if np.ptp(meseta) > config.tolerancia_meseta * salto:
        return descartado(MOTIVO_VECINO, ajuste)

    return RayoMedido(angulo_rad, r, perfil, ajuste, True, "")


def medir_rayos(imagen, centro_fila, centro_col, radio_px, polaridad, config=ConfigPerfiles()):
    """Mide config.n_rayos rayos equiespaciados desde el centro y devuelve la lista de RayoMedido (válidos y descartados).

    n_rayos = 2 * (cantidad de líneas por el centro): una línea por el centro son dos rayos opuestos.
    """
    imagen = np.asarray(imagen, dtype=float)
    angulos = np.linspace(0.0, 2.0 * np.pi, config.n_rayos, endpoint=False)
    return [medir_rayo(imagen, centro_fila, centro_col, radio_px, angulo, polaridad, config) for angulo in angulos]


# ============================================================
# PERFIL DE BORDE MEDIO (CON LOS RAYOS ALINEADOS)
# ============================================================

def esf_alineada(rayos, semiancho_px, paso_px=0.5):
    """Perfil de borde medio de los rayos válidos, alineados en su borde y normalizados de 0 (interior) a 1 (exterior).

    Devuelve (x, media, desvio, curvas): x es la distancia al borde de cada rayo; curvas tiene una fila
    por rayo válido (nan donde ese rayo no llega). Sirve para dibujar y para la MTF no paramétrica.
    """
    validos = [rayo for rayo in rayos if rayo.valido]
    x = np.arange(-semiancho_px, semiancho_px + paso_px / 2.0, paso_px)
    curvas = np.full((len(validos), x.size), np.nan)
    for i, rayo in enumerate(validos):
        a = rayo.ajuste
        interior = a.nivel_medio - a.amplitud
        normalizado = (rayo.perfil - interior) / (2.0 * a.amplitud)
        curvas[i] = np.interp(x, rayo.r - a.r_borde_px, normalizado, left=np.nan, right=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        media = np.nanmean(curvas, axis=0)
        desvio = np.nanstd(curvas, axis=0)
    return x, media, desvio, curvas


def ajustar_esf_media(x, esf):
    """Ajusta 0.5 * (1 + erf((x - x0) / (sqrt(2) sigma))) al perfil medio normalizado; devuelve (sigma, x0) o (nan, nan)."""
    ok = np.isfinite(esf)
    if ok.sum() < 8:
        return float("nan"), float("nan")

    def modelo(xx, x0, sigma):
        return 0.5 * (1.0 + erf((xx - x0) / (np.sqrt(2.0) * sigma)))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (x0, sigma), _ = curve_fit(modelo, x[ok], esf[ok], p0=[0.0, 3.0], bounds=([-np.inf, 0.2], [np.inf, np.inf]))
    except (RuntimeError, ValueError):
        return float("nan"), float("nan")
    return float(sigma), float(x0)


# %%
