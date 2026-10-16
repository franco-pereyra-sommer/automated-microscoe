"""
Calibración de los umbrales de color del detector a partir de fotos clasificadas
(mismo estilo que calibrar_umbral_rojo.py: conjuntos etiquetados y análisis min/max).

    python calibrar_detector.py CARPETA_CON_PUNTO --sin CARPETA_SIN_PUNTO [--salida CARPETA_DEBUG]

  CARPETA_CON_PUNTO : fotos de campos con al menos un punto violeta (completo, sin cortar por el borde)
  CARPETA_SIN_PUNTO : fotos de campos con glóbulos pero sin puntos violetas
  Usar fotos de la MISMA resolución con la que después se detecta (p. ej. QVGA): los umbrales de
  área están en píxeles.

Qué hace:
  1. Con los parámetros por defecto, muestra por imagen cuántos objetos detecta y su área, y guarda
     en --salida una copia con cada detección marcada para revisarla a ojo.
  2. Si hay --sin, recorre una grilla de (h_min, h_max, s_min) y busca los parámetros que detectan
     el punto en todas las fotos "con" y en ninguna "sin". Si varias combinaciones lo logran, elige la del
     centro de esa región (más robusta a cambios de iluminación) e imprime también el rango completo.

Sin posiciones verdaderas anotadas, "detectó algo" no garantiza que sea el punto correcto:
por eso conviene mirar las imágenes de --salida antes de confiar en los parámetros.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from deteccion_puntos import DetectorColor, ParametrosColor, dibujar_detecciones

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def cargar(carpeta: str) -> list[tuple[str, np.ndarray]]:
    out = []
    for ruta in sorted(Path(carpeta).iterdir()):
        if ruta.suffix.lower() in EXT:
            img = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
            if img is not None:
                out.append((ruta.name, img))
    if not out:
        raise SystemExit(f"No encontré imágenes en {carpeta}")
    return out


def area_mayor(det: DetectorColor, img: np.ndarray) -> float:
    """Área (px²) del objeto más grande que NO toca el borde; 0 si no hay ninguno."""
    return max((d.area_px for d in det.detectar(img) if not d.toca_borde), default=0.0)


def puntaje(con: list[float], sin: list[float]) -> tuple[float, float]:
    """(aciertos, margen): aciertos = fracción de 'con' detectadas - 2 x fracción de 'sin' con falsa detección;
    margen = (área mínima en 'con' - área máxima en 'sin') / área mediana en 'con'."""
    aciertos = np.mean([a > 0 for a in con]) - 2 * np.mean([a > 0 for a in sin])
    margen = (min(con) - max(sin)) / max(float(np.median(con)), 1.0)
    return float(aciertos), float(margen)


def resumen(nombre: str, v: list[float]) -> str:
    return f"{nombre}: min={min(v):.0f}  mediana={np.median(v):.0f}  max={max(v):.0f}  (px², 0 = sin detección)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("con_punto")
    ap.add_argument("--sin", dest="sin_punto")
    ap.add_argument("--salida", default="debug_detector")
    a = ap.parse_args()

    con = cargar(a.con_punto)
    sin = cargar(a.sin_punto) if a.sin_punto else []
    base = ParametrosColor()
    det = DetectorColor(base)
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)

    print(f"Parámetros por defecto: {base}\n")
    for etiqueta, conjunto in (("con_punto", con), ("sin_punto", sin)):
        for nombre, img in conjunto:
            dets = det.detectar(img)
            print(f"[{etiqueta}] {nombre}: {len(dets)} objeto(s)  áreas={[int(d.area_px) for d in dets]}  "
                  f"borde={[d.toca_borde for d in dets]}")
            cv2.imwrite(str(salida / f"{etiqueta}_{nombre}.png"), dibujar_detecciones(img, dets))
    c0 = [area_mayor(det, i) for _, i in con]
    print("\n" + resumen("con_punto", c0))
    if sin:
        s0 = [area_mayor(det, i) for _, i in sin]
        print(resumen("sin_punto", s0))
    print(f"Imágenes marcadas guardadas en {salida.resolve()}")

    if not sin:
        print("\nSin --sin no se puede buscar umbrales (se aflojarían hasta detectar todo).")
        return

    print("\nBuscando en la grilla (h_min, h_max, s_min)...")
    resultados = []
    for h_min, h_max, s_min in itertools.product(range(110, 146, 5), range(145, 176, 5), range(30, 161, 15)):
        d = DetectorColor(replace(base, h_min=h_min, h_max=h_max, s_min=s_min))
        c = [area_mayor(d, i) for _, i in con]
        s = [area_mayor(d, i) for _, i in sin]
        resultados.append((puntaje(c, s), (h_min, h_max, s_min), c, s))
    perfectos = [r for r in resultados if r[0][0] >= 1.0 and r[0][1] > 0]
    if perfectos:
        # Muchas combinaciones separan limpio: se elige la del CENTRO de esa región (la más robusta
        # a pequeños cambios de iluminación), no la de un borde.
        pts = np.array([r[1] for r in perfectos], dtype=float)
        rango = np.maximum(pts.max(0) - pts.min(0), 1.0)
        centro = np.median(pts, axis=0)
        k = int(np.argmin((((pts - centro) / rango) ** 2).sum(1)))
        (acierto, margen), (h0, h1, s0_), c, s = perfectos[k]
        print(f"  {len(perfectos)} de {len(resultados)} combinaciones separan limpio las fotos 'con' de las 'sin':")
        print(f"    h_min {int(pts[:, 0].min())}..{int(pts[:, 0].max())}   h_max {int(pts[:, 1].min())}..{int(pts[:, 1].max())}"
              f"   s_min {int(pts[:, 2].min())}..{int(pts[:, 2].max())}")
        print(f"  elegida (centro de esa región): h={h0}..{h1} s_min={s0_}  con[min/max]={min(c):.0f}/{max(c):.0f} px²  "
              f"sin[max]={max(s):.0f} px²")
    else:
        resultados.sort(key=lambda r: (r[0][0], r[0][1]), reverse=True)
        for (acierto, margen), (h0, h1, s0_), c, s in resultados[:5]:
            print(f"  h={h0}..{h1} s_min={s0_}: aciertos={acierto:+.2f} margen={margen:+.2f}  "
                  f"con[min/max]={min(c):.0f}/{max(c):.0f}  sin[max]={max(s):.0f}")
        (acierto, margen), (h0, h1, s0_), _, _ = resultados[0]
        print("AVISO: ninguna combinación separa limpio las fotos 'con' de las 'sin'. Revisar las imágenes marcadas "
              "(¿señuelos del mismo tono? ¿area_min/area_max mal puestos? ¿hace falta mirar también V o la textura?).")
    print("\nPara usar:")
    print(f"    DetectorColor(ParametrosColor(h_min={h0}, h_max={h1}, s_min={s0_}))")


if __name__ == "__main__":
    main()
