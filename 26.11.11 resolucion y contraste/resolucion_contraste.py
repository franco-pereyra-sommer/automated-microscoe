# %%

"""
Resolución y contraste de un glóbulo a partir de sus perfiles de borde (ver perfiles_de_borde.py).

Del perfil de borde (ESF) salen las dos cosas con el mismo marco:
  - Resolución: el ancho del borde. Con respuesta gaussiana de desvío sigma, la derivada del borde (LSF)
    es una gaussiana de ese sigma y la MTF es exp(-2 pi^2 sigma^2 f^2). Se informan FWHM = 2.355 sigma,
    el ancho 10-90 = 2.563 sigma, y las frecuencias donde la MTF cae a 0.5 y a 0.1 (0.187/sigma y
    0.342/sigma). Además se calcula la MTF sin suponer gaussiana (derivando el perfil medio alineado)
    como chequeo: si no coincide con la gaussiana, el borde no es gaussiano (halos, aberraciones).
  - Contraste: dos definiciones, que se miden las dos.
      * Del borde: con los niveles que da el ajuste (interior A - B, exterior A + B): Michelson = |B|/A
        y Weber = 2|B| / (A + B) (contra el fondo, que es el exterior).
      * Del glóbulo: Weber entre la mediana del interior (r < 0.6 radio) y el fondo.
    El contraste depende del canal, de la exposición/ganancia/gamma/JPEG y de la posición en el campo:
    solo se puede comparar entre fotos hechas con los mismos ajustes de cámara.

Cuidado con lo que mide: un glóbulo no es un escalón perfecto (es un disco de ~2 um de espesor en el borde),
así que sigma incluye el borde del propio glóbulo: es una cota superior de la resolución del sistema. Sirve
para comparar dos sistemas con el mismo glóbulo; como número absoluto, no.

Sin cámara ni impresora ni módulos del proyecto (categoría "imagen pura" de REGLAS_CODIGO.txt).
Las posiciones y los anchos están en píxeles; si se pasa um_por_px (la escala de ESA imagen) también salen
en micrómetros y en ciclos por micrómetro.
"""

from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.signal.windows import tukey
from scipy.special import erfinv

from perfiles_de_borde import (
    CirculoRefinado,
    ConfigPerfiles,
    RayoMedido,
    ajustar_esf_media,
    esf_alineada,
    medir_rayos,
    polaridad_del_borde,
    refinar_centro_y_radio,
)


CANALES = ("gris", "B", "G", "R")

# Constantes de la gaussiana (se calculan, no se escriben a mano).
FACTOR_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))                  # 2.35482
FACTOR_W1090 = 2.0 * np.sqrt(2.0) * erfinv(0.8)                 # 2.563103
FACTOR_MTF50 = np.sqrt(np.log(2.0) / (2.0 * np.pi ** 2))        # f50 = 0.187 / sigma
FACTOR_MTF10 = np.sqrt(np.log(10.0) / (2.0 * np.pi ** 2))       # f10 = 0.342 / sigma

# Con un sigma de este orden el borde está limitado por el tamaño del píxel y no por la óptica.
SIGMA_LIMITE_MUESTREO_PX = 1.0
# Para la MTF no paramétrica: puntos de la FFT, paso del perfil medio y apodización (ventana de Tukey: plana en
# el centro, así no angosta la LSF como lo hace Hann cuando la ventana no es mucho más ancha que sigma).
N_FFT = 4096
PASO_ESF_PX = 0.5
ALFA_TUKEY = 0.5


@dataclass
class ResultadoGlobulo:
    """Todo lo medido en un glóbulo y un canal. Los anchos están en píxeles; los *_um solo existen si se pasó um_por_px."""
    circulo: CirculoRefinado
    rayos: list[RayoMedido]
    n_rayos: int
    n_validos: int
    motivos_descarte: dict
    # resolución (px)
    sigma_mediana_px: float = np.nan
    sigma_std_px: float = np.nan              # dispersión entre rayos
    sigma_error_mediana_px: float = np.nan    # ~1.2533 * std / sqrt(n): error de la mediana si los rayos fueran independientes
    sigma_esf_media_px: float = np.nan        # sigma del perfil medio alineado
    fwhm_px: float = np.nan
    w1090_px: float = np.nan
    mtf50_ciclos_px: float = np.nan
    mtf10_ciclos_px: float = np.nan
    mtf50_no_param_ciclos_px: float = np.nan
    # contraste (adimensional)
    michelson_mediana: float = np.nan
    michelson_std: float = np.nan
    weber_borde_mediana: float = np.nan
    weber_borde_std: float = np.nan
    weber_interior: float = np.nan
    nivel_interior: float = np.nan
    nivel_fondo: float = np.nan
    # perfil medio y MTF para graficar
    esf_x: np.ndarray = field(default_factory=lambda: np.array([]))
    esf_media: np.ndarray = field(default_factory=lambda: np.array([]))
    esf_curvas: np.ndarray = field(default_factory=lambda: np.array([]))
    frecuencias_mtf: np.ndarray = field(default_factory=lambda: np.array([]))
    mtf_no_param: np.ndarray = field(default_factory=lambda: np.array([]))
    advertencias: list = field(default_factory=list)
    um_por_px: float | None = None

    def _a_um(self, valor_px):
        return valor_px * self.um_por_px if self.um_por_px is not None else None

    @property
    def sigma_mediana_um(self):
        return self._a_um(self.sigma_mediana_px)

    @property
    def fwhm_um(self):
        return self._a_um(self.fwhm_px)

    @property
    def w1090_um(self):
        return self._a_um(self.w1090_px)

    @property
    def mtf50_ciclos_um(self):
        return self.mtf50_ciclos_px / self.um_por_px if self.um_por_px is not None else None

    @property
    def mtf10_ciclos_um(self):
        return self.mtf10_ciclos_px / self.um_por_px if self.um_por_px is not None else None


# ============================================================
# CANAL DE TRABAJO
# ============================================================

def canal_de_trabajo(imagen, canal):
    """Devuelve la imagen como array 2D float del canal pedido: "gris", "B", "G" o "R" (la imagen de color viene en BGR)."""
    if canal not in CANALES:
        raise ValueError(f"canal {canal!r} no válido; las opciones son {CANALES}.")
    if imagen.ndim == 2:
        if canal != "gris":
            raise ValueError("la imagen es de un solo canal: solo se puede pedir 'gris'.")
        return imagen.astype(np.float64)
    if canal == "gris":
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return imagen[:, :, "BGR".index(canal)].astype(np.float64)


# ============================================================
# ANCHOS Y MTF
# ============================================================

def anchos_desde_sigma(sigma):
    """Devuelve (FWHM, ancho 10-90) de un borde con respuesta gaussiana de desvío sigma (en las mismas unidades)."""
    return FACTOR_FWHM * sigma, FACTOR_W1090 * sigma


def mtf_gaussiana(frecuencia, sigma):
    """MTF de una respuesta gaussiana de desvío sigma: exp(-2 pi^2 sigma^2 f^2). frecuencia en ciclos por la unidad de sigma."""
    return np.exp(-2.0 * np.pi ** 2 * sigma ** 2 * np.asarray(frecuencia) ** 2)


def mtf_desde_esf(x, esf, paso_px):
    """MTF no paramétrica: deriva el perfil de borde (LSF), la apodiza con una ventana de Tukey y toma el módulo de la FFT. Devuelve (frecuencias, mtf)."""
    ok = np.isfinite(esf)
    if ok.sum() < 8:
        return np.array([]), np.array([])
    esf = np.interp(x, x[ok], esf[ok])
    lsf = np.gradient(esf, paso_px) * tukey(esf.size, ALFA_TUKEY)
    espectro = np.abs(np.fft.rfft(lsf, N_FFT))
    return np.fft.rfftfreq(N_FFT, d=paso_px), espectro / espectro[0]


def frecuencia_de_mtf(frecuencias, mtf, nivel):
    """Primera frecuencia donde la MTF cae por debajo de `nivel` (interpolando); nan si nunca cae."""
    if mtf.size == 0:
        return float("nan")
    debajo = np.where(mtf < nivel)[0]
    if debajo.size == 0 or debajo[0] == 0:
        return float("nan")
    i = debajo[0]
    f0, f1, m0, m1 = frecuencias[i - 1], frecuencias[i], mtf[i - 1], mtf[i]
    return float(f0 + (nivel - m0) * (f1 - f0) / (m1 - m0))


# ============================================================
# MEDIR UN GLÓBULO
# ============================================================

def _mediana_y_desvio(valores):
    valores = np.asarray(valores, dtype=float)
    desvio = float(np.std(valores, ddof=1)) if valores.size > 1 else float("nan")
    return float(np.median(valores)), desvio


def medir_globulo(imagen, centro_fila_px, centro_col_px, radio_aprox_px,
                  config=ConfigPerfiles(), um_por_px=None, polaridad=None, min_rayos_validos=8):
    """Mide resolución y contraste de un glóbulo (imagen 2D de un solo canal, ver canal_de_trabajo) y devuelve un ResultadoGlobulo.

    Pasos: refina el centro y el radio del círculo aproximado, mide los config.n_rayos rayos, y con los
    válidos resume sigma (mediana y dispersión), anchos, MTF y contraste. Si hay menos de
    min_rayos_validos rayos válidos las métricas quedan en nan y se avisa en `advertencias`.
    """
    imagen = np.asarray(imagen, dtype=float)
    circulo = refinar_centro_y_radio(imagen, centro_fila_px, centro_col_px, radio_aprox_px, config, polaridad)
    rayos = medir_rayos(imagen, circulo.centro_fila_px, circulo.centro_col_px, circulo.radio_px, circulo.polaridad, config)
    validos = [rayo for rayo in rayos if rayo.valido]
    motivos = dict(Counter(rayo.motivo for rayo in rayos if not rayo.valido))
    resultado = ResultadoGlobulo(circulo=circulo, rayos=rayos, n_rayos=len(rayos), n_validos=len(validos),
                                 motivos_descarte=motivos, um_por_px=um_por_px)
    if len(validos) < min_rayos_validos:
        resultado.advertencias.append(f"solo {len(validos)} rayos válidos de {len(rayos)} (mínimo {min_rayos_validos}): no se calcularon métricas.")
        return resultado

    # --- resolución ---
    sigmas = [rayo.ajuste.sigma_px for rayo in validos]
    resultado.sigma_mediana_px, resultado.sigma_std_px = _mediana_y_desvio(sigmas)
    resultado.sigma_error_mediana_px = 1.2533 * resultado.sigma_std_px / np.sqrt(len(validos))
    resultado.fwhm_px, resultado.w1090_px = anchos_desde_sigma(resultado.sigma_mediana_px)
    resultado.mtf50_ciclos_px = FACTOR_MTF50 / resultado.sigma_mediana_px
    resultado.mtf10_ciclos_px = FACTOR_MTF10 / resultado.sigma_mediana_px
    if resultado.sigma_mediana_px < SIGMA_LIMITE_MUESTREO_PX:
        resultado.advertencias.append(
            f"sigma de {resultado.sigma_mediana_px:.2f} px (< {SIGMA_LIMITE_MUESTREO_PX} px): el borde está limitado por el tamaño del píxel, no por la óptica.")

    semiancho = config.fraccion_ventana_max * circulo.radio_px      # perfil medio ancho: la MTF necesita las colas
    x, media, _, curvas = esf_alineada(rayos, semiancho, PASO_ESF_PX)
    resultado.esf_x, resultado.esf_media, resultado.esf_curvas = x, media, curvas
    resultado.sigma_esf_media_px, _ = ajustar_esf_media(x, media)
    resultado.frecuencias_mtf, resultado.mtf_no_param = mtf_desde_esf(x, media, PASO_ESF_PX)
    resultado.mtf50_no_param_ciclos_px = frecuencia_de_mtf(resultado.frecuencias_mtf, resultado.mtf_no_param, 0.5)

    # --- contraste ---
    michelson = [abs(rayo.ajuste.amplitud) / rayo.ajuste.nivel_medio for rayo in validos]
    weber = [2.0 * abs(rayo.ajuste.amplitud) / (rayo.ajuste.nivel_medio + rayo.ajuste.amplitud) for rayo in validos]
    resultado.michelson_mediana, resultado.michelson_std = _mediana_y_desvio(michelson)
    resultado.weber_borde_mediana, resultado.weber_borde_std = _mediana_y_desvio(weber)
    resultado.nivel_fondo = float(np.median([rayo.ajuste.nivel_medio + rayo.ajuste.amplitud for rayo in validos]))

    alto, ancho = imagen.shape
    cf, cc, radio = circulo.centro_fila_px, circulo.centro_col_px, circulo.radio_px
    m = int(np.ceil(radio))
    f0, f1 = max(0, int(cf) - m), min(alto, int(cf) + m + 1)
    c0, c1 = max(0, int(cc) - m), min(ancho, int(cc) + m + 1)
    filas, cols = np.ogrid[f0:f1, c0:c1]
    interior = imagen[f0:f1, c0:c1][np.hypot(filas - cf, cols - cc) < 0.6 * radio]
    resultado.nivel_interior = float(np.median(interior))
    resultado.weber_interior = abs(resultado.nivel_fondo - resultado.nivel_interior) / resultado.nivel_fondo
    return resultado


# %%
