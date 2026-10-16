"""
Detección de puntos violetas en una imagen del microscopio.

Convención de píxeles (OpenCV / numpy):
    imagen[fila, col]      pixel_x = col,  pixel_y = fila
El centro del píxel [i, j] es la coordenada (fila=i, col=j). El centro geométrico de una
imagen de alto H y ancho W es, entonces, ((H-1)/2, (W-1)/2).

ATENCIÓN con los ejes: pixel_x (col) corresponde al eje Y de la impresora y pixel_y (fila)
al eje X de la impresora (ver centrado.py).

El barrido (barrido_puntos.py) solo necesita un objeto con el método
    detectar(imagen_bgr) -> list[Deteccion]
así que DetectorColor se puede reemplazar por otro detector (un modelo entrenado, un
detector de parásitos, etc.) sin tocar el resto del código.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Deteccion:
    col: float          # pixel_x del centro (subpíxel)
    fila: float         # pixel_y del centro (subpíxel)
    area_px: float
    diametro_px: float  # diámetro del círculo de igual área
    toca_borde: bool    # el objeto está cortado por el borde de la imagen (su centro está sesgado)

    @property
    def pixel_x(self) -> float:
        return self.col

    @property
    def pixel_y(self) -> float:
        return self.fila


class Detector(Protocol):
    def detectar(self, imagen_bgr: np.ndarray) -> list[Deteccion]: ...


@dataclass
class ParametrosColor:
    # OpenCV usa H en 0..179 (grados/2), S y V en 0..255. El violeta ronda H = 125..160.
    # Si h_min > h_max se interpreta como un rango que da la vuelta (p. ej. 170..10).
    h_min: int = 125
    h_max: int = 160
    s_min: int = 60
    v_min: int = 30
    v_max: int = 255
    # Tamaño aceptado del objeto, en píxeles² de la imagen en que se detecta. Referencia: con un campo de
    # 214 x 286 µm en QVGA (~0.9 µm/px), un glóbulo de ~8 µm mide ~9 px de diámetro (~64 px²).
    area_min_px: float = 20.0
    area_max_px: float = 2500.0
    blur_ksize: int = 3          # 0 = sin suavizado previo
    apertura_px: int = 3         # elimina motas sueltas
    cierre_px: int = 5           # rellena huecos dentro del objeto
    circularidad_min: float = 0.0  # 0 = no filtrar. Dos puntos superpuestos forman un objeto alargado
                                   # y se quiere tratarlos como uno solo, así que por defecto no se filtra.


class DetectorColor:
    """Segmenta por color (HSV) y devuelve el centroide de cada mancha conexa."""

    def __init__(self, params: ParametrosColor | None = None):
        self.p = params or ParametrosColor()

    def mascara(self, imagen_bgr: np.ndarray) -> np.ndarray:
        p = self.p
        img = imagen_bgr
        if p.blur_ksize and p.blur_ksize > 1:
            img = cv2.GaussianBlur(img, (p.blur_ksize, p.blur_ksize), 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if p.h_min <= p.h_max:
            m = cv2.inRange(hsv, (p.h_min, p.s_min, p.v_min), (p.h_max, 255, p.v_max))
        else:  # el rango de H da la vuelta
            m1 = cv2.inRange(hsv, (p.h_min, p.s_min, p.v_min), (179, 255, p.v_max))
            m2 = cv2.inRange(hsv, (0, p.s_min, p.v_min), (p.h_max, 255, p.v_max))
            m = cv2.bitwise_or(m1, m2)
        if p.apertura_px > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.apertura_px, p.apertura_px))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        if p.cierre_px > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.cierre_px, p.cierre_px))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        return m

    def detectar(self, imagen_bgr: np.ndarray) -> list[Deteccion]:
        p = self.p
        alto, ancho = imagen_bgr.shape[:2]
        mascara = self.mascara(imagen_bgr)
        n, etiquetas, stats, centroides = cv2.connectedComponentsWithStats(mascara, connectivity=8)
        salida: list[Deteccion] = []
        for i in range(1, n):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < p.area_min_px or area > p.area_max_px:
                continue
            if p.circularidad_min > 0:
                cnts, _ = cv2.findContours((etiquetas == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
                perim = cv2.arcLength(max(cnts, key=cv2.contourArea), True)
                if perim <= 0 or 4 * np.pi * area / perim**2 < p.circularidad_min:
                    continue
            x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            toca = bool(x0 == 0 or y0 == 0 or x0 + w >= ancho or y0 + h >= alto)
            salida.append(Deteccion(
                col=float(centroides[i][0]), fila=float(centroides[i][1]),
                area_px=area, diametro_px=float(2 * np.sqrt(area / np.pi)), toca_borde=toca))
        salida.sort(key=lambda d: -d.area_px)
        return salida


def dibujar_detecciones(imagen_bgr: np.ndarray, detecciones: list[Deteccion]) -> np.ndarray:
    """Copia de la imagen con el centro de la imagen (cruz blanca) y cada detección marcada."""
    out = imagen_bgr.copy()
    alto, ancho = out.shape[:2]
    cc, cf = int(round((ancho - 1) / 2)), int(round((alto - 1) / 2))
    cv2.drawMarker(out, (cc, cf), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
    for d in detecciones:
        color = (0, 165, 255) if d.toca_borde else (0, 255, 0)  # naranja = toca el borde
        cv2.circle(out, (int(round(d.col)), int(round(d.fila))), max(2, int(d.diametro_px / 2) + 2), color, 1)
        cv2.drawMarker(out, (int(round(d.col)), int(round(d.fila))), color, cv2.MARKER_CROSS, 8, 1)
    return out
