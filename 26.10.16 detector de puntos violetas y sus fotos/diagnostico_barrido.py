"""
Imágenes de diagnóstico del barrido: para cada imagen de búsqueda se guardan DOS copias con el mismo nombre,

    <carpeta_salida>/diagnostico/crudas/    la imagen tal cual la dio la cámara (replicada ESCALA veces, píxel a píxel)
    <carpeta_salida>/diagnostico/marcadas/  la misma, con lo que detectó el programa y hacia dónde manda centrar

así se las puede abrir una al lado de la otra para comparar. Hay dos clases de imagen:

  campo_NNNN_...      cada campo visual de la espiral. Para cada detección: círculo (verde = se fotografía,
                      amarillo = ya se había fotografiado ese lugar, naranja = toca el borde y se ignora), el
                      número de la detección, una flecha celeste desde la detección hasta el centro de la imagen
                      (hacia dónde hay que llevarla) y cuánto hay que mover la impresora: dx (eje x = filas) y
                      dy (eje y = columnas), en µm.
  foto_NNNN_paso_K_.. cada verificación de centrado tras moverse al objeto K-ésimo: cruz magenta = dónde
                      debería estar el objeto si el movimiento y la escala fueran exactos; círculo azul =
                      radio donde se lo busca; círculo verde grueso = lo que se encontró; flecha celeste = lo que
                      le falta para quedar en el centro (residuo, µm); cartel "CENTRADO" o "CORREGIR".

Los nombres no llevan "(x, y, z)" a propósito, para que no se confundan con las fotos finales ante
extraer_coordenadas().
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from centrado import Calibracion, pixel_a_desplazamiento_um
from deteccion_puntos import Deteccion

ESCALA = 2

VERDE, NARANJA, AMARILLO = (60, 200, 60), (0, 150, 255), (0, 230, 230)      # BGR
CELESTE, MAGENTA, AZUL, BLANCO, ROJO = (255, 200, 0), (255, 0, 255), (255, 120, 40), (255, 255, 255), (60, 60, 255)
COLOR_ESTADO = {"foto": VERDE, "borde": NARANJA, "repetido": AMARILLO}
TEXTO_ESTADO = {"foto": "se fotografia", "borde": "toca el borde: se ignora", "repetido": "ya fotografiado: se omite"}


def _p(x: float, y: float) -> tuple[int, int]:
    """Coordenada de imagen (col, fila) -> píxel de la imagen ampliada. El centro del píxel [i, j] es (j, i)."""
    return int(round((x + 0.5) * ESCALA - 0.5)), int(round((y + 0.5) * ESCALA - 0.5))


def ampliar(imagen_bgr: np.ndarray) -> np.ndarray:
    """Replica cada píxel ESCALA x ESCALA veces (no inventa información)."""
    return cv2.resize(imagen_bgr, None, fx=ESCALA, fy=ESCALA, interpolation=cv2.INTER_NEAREST)


def _texto(img, txt, org, color=BLANCO, escala=0.45):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, escala, color, 1, cv2.LINE_AA)


def _centro(img, cal: Calibracion):
    c = _p(cal.centro_col, cal.centro_fila)
    cv2.drawMarker(img, c, BLANCO, cv2.MARKER_CROSS, 22, 1, cv2.LINE_AA)


def _flecha(img, cal: Calibracion, d: Deteccion):
    cv2.arrowedLine(img, _p(d.col, d.fila), _p(cal.centro_col, cal.centro_fila), CELESTE, 1, cv2.LINE_AA, tipLength=0.08)


def _leyenda(img, lineas: list[str], max_lineas: int = 9):
    """Texto en la esquina superior izquierda, sobre una franja oscura para que se lea sobre cualquier fondo."""
    if len(lineas) > max_lineas:
        lineas = lineas[:max_lineas - 1] + [f"... y {len(lineas) - max_lineas + 1} linea(s) mas"]
    escala, paso = 0.5, 19
    ancho = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)[0][0] for l in lineas) + 12
    alto = paso * len(lineas) + 8
    franja = img[:alto, :ancho]
    franja[:] = (franja * 0.3).astype(np.uint8)
    for k, l in enumerate(lineas):
        cv2.putText(img, l, (6, 18 + k * paso), cv2.FONT_HERSHEY_SIMPLEX, escala, BLANCO, 1, cv2.LINE_AA)


def anotar_campo(imagen_bgr: np.ndarray, cal: Calibracion, detecciones: list[Deteccion], estados: list[str],
                 titulo: str) -> np.ndarray:
    """Imagen ampliada con las detecciones de un campo. estados[k] in {'foto', 'borde', 'repetido'}."""
    img = ampliar(imagen_bgr)
    _centro(img, cal)
    for k, (d, est) in enumerate(zip(detecciones, estados), start=1):
        col = COLOR_ESTADO[est]
        r = max(4, int(round(d.diametro_px / 2 + 3)) * ESCALA)
        cv2.circle(img, _p(d.col, d.fila), r, col, 1, cv2.LINE_AA)
        cv2.drawMarker(img, _p(d.col, d.fila), col, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
        _flecha(img, cal, d)
        dx, dy = pixel_a_desplazamiento_um(cal, d.col, d.fila)
        px, py = _p(d.col, d.fila)
        _texto(img, f"#{k} dx={dx:+.0f} dy={dy:+.0f}um", (min(px + r + 3, img.shape[1] - 190), max(py - 4, 14)), col, 0.5)
    _leyenda(img, [titulo, f"{len(detecciones)} deteccion(es). Cruz blanca = centro de la imagen"]
             + [f"#{k}: {TEXTO_ESTADO[e]}, area {d.area_px:.0f}px2" for k, (d, e) in enumerate(zip(detecciones, estados), 1)])
    return img


def anotar_verificacion(imagen_bgr: np.ndarray, cal: Calibracion, candidatos: list[Deteccion],
                        esperado: tuple[float, float], radio_px: float, elegido: Deteccion | None,
                        residuo_um: tuple[float, float] | None, centrado: bool | None, titulo: str,
                        siguiente_mm: tuple[float, float] | None = None) -> np.ndarray:
    """Imagen ampliada de una verificación de centrado."""
    img = ampliar(imagen_bgr)
    _centro(img, cal)
    cv2.circle(img, _p(*esperado), int(round(radio_px * ESCALA)), AZUL, 1, cv2.LINE_AA)
    cv2.drawMarker(img, _p(*esperado), MAGENTA, cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)
    for d in candidatos:
        es = d is elegido
        cv2.circle(img, _p(d.col, d.fila), max(4, int(round(d.diametro_px / 2 + 3)) * ESCALA), VERDE if es else NARANJA,
                   2 if es else 1, cv2.LINE_AA)
    lin = [titulo, "magenta X = donde deberia estar; circulo azul = radio de busqueda"]
    if elegido is None:
        lin.append("NO SE ENCONTRO NADA cerca del lugar esperado: se saca la foto igual" if not candidatos else
                   "NO SE ENCONTRO NADA dentro del radio de busqueda: se saca la foto igual")
        col_cartel = ROJO
    else:
        _flecha(img, cal, elegido)
        rx, ry = residuo_um
        lin.append(f"encontrado en col={elegido.col:.1f} fila={elegido.fila:.1f}; residuo dx={rx:+.1f} dy={ry:+.1f} um "
                   f"(tolerancia +-{cal.paso_um / 2:.2f})")
        col_cartel = VERDE if centrado else ROJO
        lin.append("CENTRADO" if centrado else
                   ("CORREGIR: ir a x=%.4f y=%.4f" % siguiente_mm if siguiente_mm else "SIN CENTRAR (no quedan correcciones)"))
    _leyenda(img, lin)
    cv2.putText(img, lin[-1].split(":")[0], (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col_cartel, 2, cv2.LINE_AA)
    return img


class Diagnostico:
    """Guarda las imágenes (crudas y marcadas) en <carpeta>/diagnostico/{crudas,marcadas}. Si activo=False no hace nada."""

    def __init__(self, carpeta_salida: str | Path, activo: bool = True):
        self.activo = activo
        self.carpeta_crudas = Path(carpeta_salida) / "diagnostico" / "crudas"
        self.carpeta_marcadas = Path(carpeta_salida) / "diagnostico" / "marcadas"
        if activo:
            self.carpeta_crudas.mkdir(parents=True, exist_ok=True)
            self.carpeta_marcadas.mkdir(parents=True, exist_ok=True)
        self.n_campos = 0

    @staticmethod
    def _sufijo(pos_mm: tuple[float, float]) -> str:
        return f"x{pos_mm[0]:.4f}_y{pos_mm[1]:.4f}"

    def _guardar(self, nombre: str, cruda: np.ndarray, marcada: np.ndarray) -> None:
        for carpeta, img in ((self.carpeta_crudas, ampliar(cruda)), (self.carpeta_marcadas, marcada)):
            if not cv2.imwrite(str(carpeta / nombre), img):
                raise OSError(f"No pude escribir {carpeta / nombre}")

    def campo(self, imagen, cal, detecciones, estados, pos_mm) -> None:
        if not self.activo:
            return
        self.n_campos += 1
        titulo = f"campo {self.n_campos}  x={pos_mm[0]:.4f} y={pos_mm[1]:.4f} mm"
        self._guardar(f"campo_{self.n_campos:04d}_{self._sufijo(pos_mm)}.png", imagen,
                      anotar_campo(imagen, cal, detecciones, estados, titulo))

    def verificacion(self, imagen, cal, candidatos, esperado, radio_px, elegido, residuo_um, centrado,
                     n_foto: int, intento: int, pos_mm, siguiente_mm=None) -> None:
        if not self.activo:
            return
        titulo = f"foto {n_foto}  paso {intento}  x={pos_mm[0]:.4f} y={pos_mm[1]:.4f} mm"
        self._guardar(f"foto_{n_foto:04d}_paso_{intento}_{self._sufijo(pos_mm)}.png", imagen,
                      anotar_verificacion(imagen, cal, candidatos, esperado, radio_px, elegido, residuo_um, centrado,
                                          titulo, siguiente_mm))
