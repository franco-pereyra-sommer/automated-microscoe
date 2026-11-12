# %%

"""
Gráficos de diagnóstico de la medición de un glóbulo (resolucion_contraste.ResultadoGlobulo).

Una figura de 4 paneles, pensada para mirar si la medición es confiable y no solo el número:
  1. El glóbulo con el círculo ajustado y los rayos: azul = válido, naranja = descartado.
  2. El perfil de borde de cada rayo válido (alineado en su borde), el perfil medio y la ERF con el sigma mediano.
  3. Sigma de cada rayo contra su ángulo (una dependencia con el ángulo es astigmatismo o desenfoque de
     movimiento); abajo, marcados, los ángulos descartados.
  4. MTF: la gaussiana con el sigma mediano y la calculada sin suponer gaussiana, con las líneas de 0.5 y 0.1.

Colores: azul y naranja son las dos primeras posiciones de la paleta categórica de referencia; además las dos
curvas de cada panel se distinguen por el estilo de línea, no solo por el color.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf


COLOR_AZUL = "#2a78d6"
COLOR_NARANJA = "#eb6834"
COLOR_TEXTO = "#0b0b0b"
COLOR_TEXTO_SECUNDARIO = "#52514e"
COLOR_GRILLA = "#d9d8d4"
COLOR_CURVAS_FINAS = "#9a9992"


def _estilo_eje(eje, titulo, xlabel, ylabel):
    eje.set_title(titulo, fontsize=10, color=COLOR_TEXTO, loc="left")
    eje.set_xlabel(xlabel, fontsize=9, color=COLOR_TEXTO_SECUNDARIO)
    eje.set_ylabel(ylabel, fontsize=9, color=COLOR_TEXTO_SECUNDARIO)
    eje.grid(True, color=COLOR_GRILLA, linewidth=0.6)
    eje.set_axisbelow(True)
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        eje.spines[lado].set_color(COLOR_GRILLA)
    eje.tick_params(colors=COLOR_TEXTO_SECUNDARIO, labelsize=8)


def graficar_globulo(imagen_bgr, resultado, ruta_salida=None, titulo="", mostrar=False):
    """Dibuja la figura de diagnóstico de un glóbulo; la guarda en ruta_salida (PNG) si se la da y devuelve la figura."""
    circulo = resultado.circulo
    escala = resultado.um_por_px if resultado.um_por_px is not None else 1.0
    unidad = "µm" if resultado.um_por_px is not None else "px"
    hay_metricas = np.isfinite(resultado.sigma_mediana_px)

    figura, ejes = plt.subplots(2, 2, figsize=(12, 9.5))
    figura.patch.set_facecolor("white")

    # --- 1: el glóbulo y sus rayos ---
    eje = ejes[0, 0]
    alto, ancho = imagen_bgr.shape[:2]
    margen = 1.9 * circulo.radio_px
    f0, f1 = int(max(0, circulo.centro_fila_px - margen)), int(min(alto, circulo.centro_fila_px + margen))
    c0, c1 = int(max(0, circulo.centro_col_px - margen)), int(min(ancho, circulo.centro_col_px + margen))
    recorte = imagen_bgr[f0:f1, c0:c1]
    eje.imshow(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB) if recorte.ndim == 3 else recorte,
               cmap=None if recorte.ndim == 3 else "gray", extent=(c0 - 0.5, c1 - 0.5, f1 - 0.5, f0 - 0.5))
    t = np.linspace(0, 2 * np.pi, 361)
    eje.plot(circulo.centro_col_px + circulo.radio_px * np.cos(t), circulo.centro_fila_px + circulo.radio_px * np.sin(t),
             color="white", linewidth=0.8, linestyle=":")
    for rayo in resultado.rayos:
        r0 = rayo.ajuste.r_borde_px if rayo.ajuste is not None else circulo.radio_px
        semi = rayo.ajuste.semiventana_px if rayo.ajuste is not None else 0.4 * circulo.radio_px
        r = np.array([r0 - 0.5 * semi, r0 + semi])
        eje.plot(circulo.centro_col_px + r * np.cos(rayo.angulo_rad), circulo.centro_fila_px + r * np.sin(rayo.angulo_rad),
                 color=COLOR_AZUL if rayo.valido else COLOR_NARANJA, linewidth=1.2 if rayo.valido else 1.6)
    eje.plot([], [], color=COLOR_AZUL, label=f"rayo válido ({resultado.n_validos})")
    eje.plot([], [], color=COLOR_NARANJA, label=f"rayo descartado ({resultado.n_rayos - resultado.n_validos})")
    eje.legend(loc="upper right", fontsize=8, framealpha=0.9)
    eje.set_title(f"Glóbulo y rayos (círculo: radio {circulo.radio_px:.1f} px, desvío {circulo.rms_px:.2f} px)",
                  fontsize=10, color=COLOR_TEXTO, loc="left")
    eje.set_xlabel("columna [px]", fontsize=9, color=COLOR_TEXTO_SECUNDARIO)
    eje.set_ylabel("fila [px]", fontsize=9, color=COLOR_TEXTO_SECUNDARIO)
    eje.tick_params(colors=COLOR_TEXTO_SECUNDARIO, labelsize=8)

    if not hay_metricas:
        for e in (ejes[0, 1], ejes[1, 0], ejes[1, 1]):
            e.axis("off")
        ejes[0, 1].text(0.0, 0.5, "\n".join(resultado.advertencias) or "sin métricas",
                        color=COLOR_TEXTO, fontsize=10, transform=ejes[0, 1].transAxes, wrap=True)
    else:
        sigma = resultado.sigma_mediana_px

        # --- 2: perfil de borde ---
        eje = ejes[0, 1]
        for curva in resultado.esf_curvas:
            eje.plot(resultado.esf_x * escala, curva, color=COLOR_CURVAS_FINAS, linewidth=0.6, alpha=0.25)
        eje.plot([], [], color=COLOR_CURVAS_FINAS, label="cada rayo válido")
        eje.plot(resultado.esf_x * escala, resultado.esf_media, color=COLOR_AZUL, linewidth=2.0, label="perfil medio alineado")
        eje.plot(resultado.esf_x * escala, 0.5 * (1 + erf(resultado.esf_x / (np.sqrt(2) * sigma))), color=COLOR_NARANJA,
                 linewidth=1.8, linestyle="--", label=f"ERF con sigma mediano ({sigma:.2f} px)")
        eje.set_xlim(-min(resultado.esf_x.max(), 6 * sigma + 8) * escala, min(resultado.esf_x.max(), 6 * sigma + 8) * escala)
        _estilo_eje(eje, "Perfil de borde (0 = interior, 1 = exterior)", f"distancia al borde [{unidad}]", "intensidad normalizada")
        eje.legend(loc="upper left", fontsize=8, framealpha=0.9)

        # --- 3: sigma contra ángulo ---
        eje = ejes[1, 0]
        validos = [r for r in resultado.rayos if r.valido]
        grados = np.degrees([r.angulo_rad for r in validos])
        sigmas = np.array([r.ajuste.sigma_px for r in validos])
        eje.scatter(grados, sigmas * escala, s=18, color=COLOR_AZUL, label="sigma de cada rayo válido", zorder=3)
        eje.axhline(sigma * escala, color=COLOR_TEXTO_SECUNDARIO, linestyle="--", linewidth=1.0, label=f"mediana {sigma:.2f} px")
        piso = (sigmas.min() - 0.15 * max(np.ptp(sigmas), 0.2)) * escala
        descartados = np.degrees([r.angulo_rad for r in resultado.rayos if not r.valido])
        eje.scatter(descartados, np.full(descartados.size, piso), s=30, marker="|", color=COLOR_NARANJA, linewidths=1.6,
                    label="rayo descartado", zorder=3, clip_on=False)
        eje.set_xlim(0, 360)
        eje.set_xticks(range(0, 361, 60))
        _estilo_eje(eje, "Sigma de cada rayo contra su ángulo", "ángulo del rayo [grados]", f"sigma [{unidad}]")
        eje.legend(loc="best", fontsize=8, framealpha=0.9)

        # --- 4: MTF ---
        eje = ejes[1, 1]
        f_max = min(0.5, 3.0 * resultado.mtf10_ciclos_px)
        f = np.linspace(0, f_max, 400)
        eje.plot(f / escala, np.exp(-2 * np.pi ** 2 * sigma ** 2 * f ** 2), color=COLOR_AZUL, linewidth=2.0,
                 label=f"gaussiana (MTF50 = {resultado.mtf50_ciclos_px / escala:.3g} ciclos/{unidad})")
        if resultado.frecuencias_mtf.size:
            usar = resultado.frecuencias_mtf <= f_max
            eje.plot(resultado.frecuencias_mtf[usar] / escala, resultado.mtf_no_param[usar], color=COLOR_NARANJA, linewidth=1.8,
                     linestyle="--", label=f"sin suponer gaussiana (MTF50 = {resultado.mtf50_no_param_ciclos_px / escala:.3g})")
        for nivel in (0.5, 0.1):
            eje.axhline(nivel, color=COLOR_CURVAS_FINAS, linestyle=":", linewidth=0.9)
        eje.set_xlim(0, f_max / escala)
        eje.set_ylim(0, 1.05)
        _estilo_eje(eje, "MTF del borde", f"frecuencia espacial [ciclos/{unidad}]", "MTF")
        eje.legend(loc="upper right", fontsize=8, framealpha=0.9)

    resumen = titulo
    if hay_metricas:
        resumen += (f"\nsigma = {resultado.sigma_mediana_px:.2f} ± {resultado.sigma_std_px:.2f} px (mediana ± dispersión entre rayos)   "
                    f"FWHM = {resultado.fwhm_px:.2f} px   w10-90 = {resultado.w1090_px:.2f} px   "
                    f"Michelson = {resultado.michelson_mediana:.3f}   Weber borde = {resultado.weber_borde_mediana:.3f}   "
                    f"Weber interior = {resultado.weber_interior:.3f}")
        if resultado.um_por_px is not None:
            resumen += f"\nFWHM = {resultado.fwhm_um:.3f} µm   w10-90 = {resultado.w1090_um:.3f} µm   escala {resultado.um_por_px:.5f} µm/px"
    figura.suptitle(resumen, fontsize=10, color=COLOR_TEXTO, x=0.01, ha="left", y=0.995)
    figura.tight_layout(rect=(0, 0, 1, 0.93))
    if ruta_salida is not None:
        figura.savefig(ruta_salida, dpi=110)
    if mostrar:
        plt.show()
    else:
        plt.close(figura)
    return figura


# %%
