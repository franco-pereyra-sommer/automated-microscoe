"""
Gráfico de la idea de mirar los colores de los píxeles: tono (H de HSV, 0..179 en OpenCV) de los píxeles de los
puntos violetas, de los glóbulos y del plasma, con la densidad normalizada (área = 1) de cada clase.

    python analizar_colores.py CARPETA_CON_PUNTOS --sin CARPETA_SIN_PUNTOS [--salida colores_puntos.png]

Los píxeles de los "puntos" son los de las detecciones del detector por contraste (deteccion_contraste.py), que
conviene haber revisado antes con calibrar_detector.py. Solo se cuentan los píxeles con saturación >= 40 (en un gris el
tono no está definido). El gráfico de la derecha muestra, foto por foto, dónde cae el tono mediano de los puntos frente
al de los glóbulos: si los rangos se superponen o se mueven de una foto a otra, un rango de color fijo no sirve.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibrar_detector import cargar
from deteccion_contraste import DetectorContraste, ParametrosContraste, estimar_mascara_fija

TINTA, TINTA_SEC, GRILLA = "#0b0b0b", "#52514e", "#d9d8d3"
C_PUNTO, C_GLOBULO = "#2a78d6", "#e87ba4"       # azul y magenta de la paleta de referencia
S_MIN = 40                                       # con saturación menor el tono no está definido (gris): se descarta


def tono_desenrollado(H: np.ndarray) -> np.ndarray:
    """H de OpenCV (0..179) sin el corte en 0: los rosados/rojizos que cruzan el 0 siguen en 180..190, no en 0..10."""
    h = H.astype(np.float32)
    return np.where(h < 60, h + 180, h)


def clases(img: np.ndarray, det: DetectorContraste):
    """(tono de los píxeles de puntos, tono del resto de los píxeles con color), ambos desenrollados."""
    h, w = img.shape[:2]
    puntos = np.zeros((h, w), np.uint8)
    for d in det.detectar(img):
        cv2.circle(puntos, (int(round(d.col)), int(round(d.fila))), max(2, int(d.diametro_px / 2)), 1, -1)
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 1.0), cv2.COLOR_BGR2HSV)
    con_color = hsv[..., 1] >= S_MIN
    H = tono_desenrollado(hsv[..., 0])
    return H[(puntos > 0) & con_color], H[(puntos == 0) & con_color]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("con_punto")
    ap.add_argument("--sin", required=True)
    ap.add_argument("--salida", default="colores_puntos.png")
    ap.add_argument("--titulo", default="")
    ap.add_argument("--umbral", type=float, default=55.0)
    ap.add_argument("--peso-azul", type=float, default=1.5)
    a = ap.parse_args()
    con, sin = cargar(a.con_punto), cargar(a.sin)
    p = ParametrosContraste(umbral=a.umbral, peso_azul=a.peso_azul)
    det = DetectorContraste(p, estimar_mascara_fija([i for _, i in con] + [i for _, i in sin], p))

    hp, hg, por_foto = [], [], []
    for k, (_, img) in enumerate(con):
        pp, gg = clases(img, det)
        hp.append(pp); hg.append(gg)
        if len(pp) >= 10:
            por_foto.append((k, np.median(pp), np.percentile(gg, 25), np.percentile(gg, 75)))
    for _, img in sin:
        hg.append(clases(img, det)[1])
    hp, hg = np.concatenate(hp), np.concatenate(hg)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRILLA, "text.color": TINTA, "axes.labelcolor": TINTA_SEC,
                         "xtick.color": TINTA_SEC, "ytick.color": TINTA_SEC})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    bins = np.arange(60, 191, 2)
    tex = {}
    for datos, color, nombre in ((hg, C_GLOBULO, "resto con color (glóbulos y plasma)"),
                                 (hp, C_PUNTO, "puntos violetas (detectados)")):
        dens, _ = np.histogram(datos, bins=bins, density=True)
        ax1.fill_between(bins[:-1] + 1, dens, color=color, alpha=0.25, linewidth=0, step="mid")
        ax1.step(bins[:-1] + 1, dens, color=color, linewidth=1.6, where="mid", label=f"{nombre}, n={len(datos):,}")
        tex[nombre] = (np.median(datos), np.percentile(datos, 5), np.percentile(datos, 95))
    ticks = list(range(60, 191, 20))
    ax1.set_xticks(ticks); ax1.set_xticklabels([str(t % 180 if t >= 180 else t) for t in ticks])
    ax1.set_xlabel("tono H (OpenCV 0–179; sobre 180 = pasó el 0, o sea rojizo)")
    ax1.set_ylabel("densidad de píxeles (normalizada, área = 1)")
    ax1.set_title((a.titulo + ": " if a.titulo else "") + "tono de los píxeles con color", loc="left", fontsize=11)
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.15)
    ax1.grid(axis="y", color=GRILLA, linewidth=0.6); ax1.set_axisbelow(True)
    for sp in ("top", "right"): ax1.spines[sp].set_visible(False)

    if por_foto:
        ks, med, lo, hi = zip(*por_foto)
        x = np.arange(len(ks))
        ax2.vlines(x, lo, hi, color=C_GLOBULO, linewidth=7, alpha=0.5, label="resto con color (percentil 25–75)")
        ax2.plot(x, med, "o", color=C_PUNTO, markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=1.5,
                 label="puntos violetas (mediana)")
        ax2.set_xticks(x); ax2.set_xticklabels([str(k) for k in ks])
        ax2.set_xlabel("foto de la carpeta 'con'"); ax2.set_ylabel("tono H (desenrollado)")
        ax2.set_title("Tono por foto", loc="left", fontsize=11)
        ax2.legend(frameon=False, fontsize=9, loc="upper left")
        ax2.set_ylim(min(lo + med) - 10, max(hi + med) + 35)
    ax2.grid(axis="y", color=GRILLA, linewidth=0.6); ax2.set_axisbelow(True)
    for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.salida, dpi=140)
    print(f"Guardado {Path(a.salida).resolve()}")
    for nombre, (m, lo5, hi5) in tex.items():
        print(f"  {nombre}: H mediana {m % 180:.0f}, percentil 5-95: {lo5 % 180:.0f}-{hi5 % 180:.0f}")
    if por_foto:
        print("  tono mediano de los puntos por foto:", [round(float(m % 180)) for m in med])


if __name__ == "__main__":
    main()
