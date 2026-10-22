"""
Calibración del detector de puntos violetas a partir de fotos clasificadas en dos carpetas.

    python calibrar_detector.py CARPETA_CON_PUNTOS --sin CARPETA_SIN_PUNTOS [--total N]
                                [--salida CARPETA] [--metodo contraste|color]

  CARPETA_CON_PUNTOS : fotos de campos que tienen al menos un punto violeta
  CARPETA_SIN_PUNTOS : fotos de campos con glóbulos pero SIN puntos violetas
  --total N          : cuántos puntos violetas hay en total en CARPETA_CON_PUNTOS, contados como se ven en
                       las fotos (un punto que aparece en dos fotos superpuestas cuenta dos veces)
  --salida           : carpeta donde se guardan las fotos marcadas y la máscara de manchas fijas
  --metodo           : 'contraste' (por defecto; más oscuro y más azul que el entorno, robusto a cambios de
                       iluminación) o 'color' (rango de color HSV, el método anterior)

Usar fotos de la MISMA resolución con la que después se detecta (p. ej. QVGA): los umbrales de área
están en píxeles. Conviene que se hayan sacado con exposición, ganancia y balance de blancos fijos.

Qué imprime:
  1. Cuántas detecciones hay en las fotos SIN puntos (todas son falsas) y en las fotos CON puntos.
  2. Si se da --total: cuántas de las detecciones de 'con' se estima que son falsas (con la tasa de falsas
     medida en 'sin') y, por diferencia, cuántos de los puntos reales se detectaron. Es una ESTIMACIÓN a
     partir de conteos: no comprueba que las detecciones sean los puntos correctos. Para eso hay que mirar
     las fotos marcadas de --salida.
  3. Los parámetros elegidos y la línea de código para usarlos.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from deteccion_contraste import (DetectorContraste, ParametrosContraste, estimar_mascara_fija,
                                 guardar_mascara_fija)
from deteccion_puntos import DetectorColor, ParametrosColor, dibujar_detecciones

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
COSTO_FALSA = 0.25      # una falsa detección cuesta 1/4 de perder un punto real (una foto de más vs. un punto perdido)
TOLERANCIA_COSTO = 0.5  # se consideran "igual de buenos" los parámetros a menos de esto del mejor


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


@dataclass
class Estadistica:
    n_con: int             # detecciones en las fotos con puntos
    n_sin: int             # detecciones en las fotos sin puntos (todas falsas)
    fotos_con: int
    fotos_sin: int
    fotos_sin_con_falsa: int
    fotos_con_vacias: int  # fotos "con" donde no se detectó nada
    total: int | None      # puntos reales declarados en 'con'

    @property
    def tasa_falsas(self) -> float:
        """Falsas detecciones por foto, medidas en las fotos sin puntos."""
        return self.n_sin / max(self.fotos_sin, 1)

    @property
    def falsas_estimadas_en_con(self) -> float:
        return self.tasa_falsas * self.fotos_con

    @property
    def reales_estimados(self) -> float | None:
        if self.total is None:
            return None
        return float(np.clip(self.n_con - self.falsas_estimadas_en_con, 0, self.total))

    @property
    def costo(self) -> float:
        """Menor es mejor. Con --total: puntos reales que se estima que faltan + 1/4 por cada falsa estimada.
        Sin --total: fotos 'con' sin ninguna detección (se supone >= 1 punto por foto) + lo mismo."""
        perdidos = (self.total - self.reales_estimados) if self.total is not None else self.fotos_con_vacias
        return float(perdidos + COSTO_FALSA * (self.falsas_estimadas_en_con if self.total is not None else
                                               self.n_sin / max(self.fotos_sin, 1) * self.fotos_con))


def estadistica(c: list[int], s: list[int], total: int | None) -> Estadistica:
    return Estadistica(n_con=sum(c), n_sin=sum(s), fotos_con=len(c), fotos_sin=len(s),
                       fotos_sin_con_falsa=sum(1 for x in s if x > 0), fotos_con_vacias=sum(1 for x in c if x == 0),
                       total=total)


def mensaje(e: Estadistica) -> str:
    l = [f"En las imágenes SIN puntos violetas se encontraron {e.n_sin} detecciones "
         f"(en {e.fotos_sin_con_falsa} de {e.fotos_sin} fotos).",
         f"En las imágenes CON puntos violetas se encontraron {e.n_con} detecciones "
         f"({e.fotos_con_vacias} de {e.fotos_con} fotos quedaron sin ninguna)"
         + (f", y indicaste que hay {e.total} puntos." if e.total is not None else ".")]
    if e.total is not None:
        l.append(f"Estimación: unas {e.falsas_estimadas_en_con:.1f} de las detecciones de 'con' serían falsas "
                 f"(tasa medida en 'sin': {e.tasa_falsas:.2f} por foto), o sea que se habrían detectado ≈ "
                 f"{e.reales_estimados:.1f} de los {e.total} puntos ({100 * e.reales_estimados / e.total:.0f} %).")
        if e.reales_estimados < 0.9 * e.total:
            l.append("  Ni con los mejores parámetros se llega al total indicado. Si el total es correcto, mirá en las fotos marcadas "
                     "qué puntos faltan (¿desenfocados? ¿cortados por el borde? ¿mucho más claros que el resto?). Si no, "
                     "revisá el conteo: un total mayor que los puntos reales hace que el calibrador baje los umbrales para "
                     "'encontrarlos' y aparezcan falsas.")
        if e.n_con > e.total + e.falsas_estimadas_en_con + 1:
            l.append(f"  Hay más detecciones de las esperadas ({e.n_con} > {e.total} + {e.falsas_estimadas_en_con:.1f} falsas): "
                     "puede que 'con' tenga más puntos de los que contaste, o que el detector parta un punto en varios.")
    return "\n".join(l)


def centro_de_la_region(candidatos: list[tuple]) -> tuple:
    """De varios parámetros 'igual de buenos' elige el del centro de la región (más robusto a cambios chicos de
    iluminación que uno del borde). Cada candidato es (costo, params_tuple, ...)."""
    pts = np.array([c[1] for c in candidatos], dtype=float)
    rango = np.maximum(pts.max(0) - pts.min(0), 1e-9)
    centro = np.median(pts, axis=0)
    return candidatos[int(np.argmin((((pts - centro) / rango) ** 2).sum(1)))]


def guardar_marcadas(det, conjuntos, salida: Path) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    for etiqueta, conjunto in conjuntos:
        for nombre, img in conjunto:
            cv2.imwrite(str(salida / f"{etiqueta}_{nombre}.png"), dibujar_detecciones(img, det.detectar(img)))


# ------------------------------------------------------------------------------ método: contraste
def calibrar_contraste(con, sin, total, salida: Path) -> None:
    base = ParametrosContraste()
    imgs = [i for _, i in con] + [i for _, i in sin]
    mascara = estimar_mascara_fija(imgs, base)
    n_manchas, _, st, cen = cv2.connectedComponentsWithStats(mascara.astype(np.uint8), connectivity=8)
    if mascara.any():
        print(f"Manchas fijas (polvo/lente, en el mismo píxel en la mayoría de las fotos): {n_manchas - 1}")
        for c, s in zip(cen[1:], st[1:]):
            print(f"    en (col={c[0]:.0f}, fila={c[1]:.0f}), {int(s[cv2.CC_STAT_AREA])} px²  -> se ignoran")
    elif len(imgs) < 6:
        print("(Hay menos de 6 fotos en total: no se puede detectar manchas fijas.)")
    else:
        print("No se encontraron manchas fijas.")
    salida.mkdir(parents=True, exist_ok=True)
    guardar_mascara_fija(mascara, salida / "mascara_fija.png")

    # mapas (parte lenta) una sola vez por foto; el barrido de parámetros solo repite el umbral
    base_det = DetectorContraste(base, mascara)
    mapas_con = [base_det.mapas(i) for _, i in con]
    mapas_sin = [base_det.mapas(i) for _, i in sin]

    resultados = []
    for w, t in itertools.product((0.5, 1.0, 1.5, 2.0), range(20, 121, 5)):
        det = DetectorContraste(replace(base, peso_azul=w, umbral=float(t)), mascara)
        c = [len(det.detectar_desde_mapas(*m)) for m in mapas_con]
        s = [len(det.detectar_desde_mapas(*m)) for m in mapas_sin]
        resultados.append((estadistica(c, s, total), (w, t)))

    mejor = min(r[0].costo for r in resultados)
    buenos = [(e.costo, par, e) for e, par in resultados if e.costo <= mejor + TOLERANCIA_COSTO]
    costo, (w, t), e = centro_de_la_region(buenos)
    ws = sorted({b[1][0] for b in buenos})
    ts = sorted({b[1][1] for b in buenos})
    print(f"\nProbé {len(resultados)} combinaciones de (peso_azul, umbral). {len(buenos)} son igual de buenas: "
          f"peso_azul {ws[0]}..{ws[-1]}, umbral {ts[0]}..{ts[-1]}.")
    print(f"Elegida (centro de esa región): peso_azul={w}, umbral={t}\n")
    print(mensaje(e))

    elegido = DetectorContraste(replace(base, peso_azul=w, umbral=float(t)), mascara)
    guardar_marcadas(elegido, (("con", con), ("sin", sin)), salida)
    print(f"\nFotos marcadas y mascara_fija.png guardadas en {salida.resolve()}")
    print("Mirá las fotos marcadas: el conteo no garantiza que cada detección sea un punto de verdad.")
    print("\nPara usar:")
    print("    from deteccion_contraste import DetectorContraste, ParametrosContraste, cargar_mascara_fija")
    print(f"    detector = DetectorContraste(ParametrosContraste(umbral={t}, peso_azul={w}),")
    print(f"                                 cargar_mascara_fija(r\"{(salida / 'mascara_fija.png').resolve()}\"))")


# ------------------------------------------------------------------------------ método: color (HSV)
def area_mayor(det: DetectorColor, img: np.ndarray) -> float:
    return max((d.area_px for d in det.detectar(img) if not d.toca_borde), default=0.0)


def calibrar_color(con, sin, total, salida: Path) -> None:
    base = ParametrosColor()
    resultados = []
    for h_min, h_max, s_min in itertools.product(range(100, 146, 5), range(145, 176, 5), range(30, 161, 15)):
        det = DetectorColor(replace(base, h_min=h_min, h_max=h_max, s_min=s_min))
        c = [len(det.detectar(i)) for _, i in con]
        s = [len(det.detectar(i)) for _, i in sin]
        resultados.append((estadistica(c, s, total), (h_min, h_max, s_min)))
    mejor = min(r[0].costo for r in resultados)
    buenos = [(e.costo, par, e) for e, par in resultados if e.costo <= mejor + TOLERANCIA_COSTO]
    costo, (h0, h1, s0), e = centro_de_la_region(buenos)
    print(f"Probé {len(resultados)} combinaciones de (h_min, h_max, s_min); {len(buenos)} son igual de buenas.")
    print(f"Elegida (centro de esa región): h={h0}..{h1} s_min={s0}\n")
    print(mensaje(e))
    guardar_marcadas(DetectorColor(replace(base, h_min=h0, h_max=h1, s_min=s0)), (("con", con), ("sin", sin)), salida)
    print(f"\nFotos marcadas guardadas en {salida.resolve()}")
    print("\nPara usar:")
    print(f"    DetectorColor(ParametrosColor(h_min={h0}, h_max={h1}, s_min={s0}))")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("con_punto", help="carpeta con fotos que tienen puntos violetas")
    ap.add_argument("--sin", dest="sin_punto", required=True, help="carpeta con fotos sin puntos violetas")
    ap.add_argument("--total", type=int, default=None, help="cantidad total de puntos violetas en la carpeta 'con'")
    ap.add_argument("--salida", default="debug_detector")
    ap.add_argument("--metodo", choices=("contraste", "color"), default="contraste")
    a = ap.parse_args()

    con, sin = cargar(a.con_punto), cargar(a.sin_punto)
    print(f"{len(con)} fotos con puntos, {len(sin)} fotos sin puntos. Método: {a.metodo}\n")
    salida = Path(a.salida)
    if a.metodo == "contraste":
        calibrar_contraste(con, sin, a.total, salida)
    else:
        calibrar_color(con, sin, a.total, salida)


if __name__ == "__main__":
    main()
