"""
Detector de puntos violetas por CONTRASTE LOCAL (más robusto que el color absoluto).

Por qué no alcanza con un rango de color (HSV): en las fotos de calibración el mismo tipo de punto
aparece con tonos y saturaciones muy distintos de una foto a otra (H de 110 a 146, S de 16 a 103),
porque cambian la iluminación, el balance de blancos y el foco. Lo que SÍ se mantiene siempre es que
el punto es más OSCURO y más AZUL que lo que tiene alrededor. Entonces, para cada píxel se mide

        dL = L* - L*_local          (negativo = más oscuro que su entorno)
        db = b* - b*_local          (negativo = más azul que su entorno)
        puntaje = -dL - peso_azul * db

donde "_local" es la mediana en una ventana de `fondo_px` píxeles (el fondo local de glóbulos y
plasma). Se detecta lo que supera `umbral`. Un cambio global de color no afecta, porque se resta el
entorno; y como dL y db se escalan por brillo_ref / L*_local, tampoco afecta un cambio de brillo (en
pruebas con fotos reales alteradas de 0.7x a 1.4x de brillo y con dominantes cálida/fría, sin la
normalización aparecían falsas detecciones y se perdían puntos).

Manchas fijas: el polvo del sensor o de la lente es un punto negro que está en el MISMO píxel en todas
las fotos. Es muy oscuro y neutro, pero igual puede superar el umbral. `estimar_mascara_fija` lo
detecta solo (lo que aparece en el mismo lugar en la mayoría de las fotos de calibración no es una
muestra) y el detector lo ignora.

Implementa el mismo protocolo que DetectorColor (detectar(imagen_bgr) -> list[Deteccion]), así que se
puede usar en barrido_puntos.barrer sin tocar nada más.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from deteccion_puntos import Deteccion


@dataclass
class ParametrosContraste:
    umbral: float = 50.0           # puntaje mínimo (unidades de L* y b* de OpenCV, 0..255)
    peso_azul: float = 1.0         # 0 = solo oscuridad; 1 = oscuro y azul pesan igual
    fondo_px: int = 41             # ventana (impar) de la mediana local; >= ~3 veces el diámetro del punto
    suavizado_sigma: float = 1.2   # px, antes de medir el contraste (reduce ruido y compresión JPEG)
    brillo_ref: float = 140.0      # el contraste se mide relativo al brillo local (L*_local) y se expresa como si el
                                   # entorno valiera brillo_ref: así un cambio de exposición o de luz no mueve el
                                   # umbral. 0 = contraste absoluto, sin normalizar.
    area_min_px: float = 15.0      # tamaño aceptado, en píxeles² de la imagen en que se detecta; depende
    area_max_px: float = 1500.0    # del aumento (con la lente nueva en QVGA un punto mide ~50-200 px²)
    apertura_px: int = 3           # elimina motas sueltas
    dilatacion_mascara_px: int = 4 # cuánto se agranda cada mancha fija al ignorarla


def _fondo_local(canal: np.ndarray, ventana: int) -> np.ndarray:
    if ventana % 2 == 0:
        ventana += 1
    return cv2.medianBlur(np.clip(canal, 0, 255).astype(np.uint8), ventana).astype(np.float32)


class DetectorContraste:
    """Puntos más oscuros y más azules que su entorno. `mascara_fija` (bool, alto x ancho) se ignora."""

    def __init__(self, params: ParametrosContraste | None = None, mascara_fija: np.ndarray | None = None):
        self.p = params or ParametrosContraste()
        self.mascara_fija = mascara_fija

    # ---------------------------------------------------------------- mapas
    def mapas(self, imagen_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(dL, db) por píxel: cuánto más oscuro y cuánto más azul que el entorno (negativo = sí)."""
        p = self.p
        lab = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        if p.suavizado_sigma > 0:
            lab = cv2.GaussianBlur(lab, (0, 0), p.suavizado_sigma)
        L, b = lab[..., 0], lab[..., 2]
        L_fondo = _fondo_local(L, p.fondo_px)
        dL, db = L - L_fondo, b - _fondo_local(b, p.fondo_px)
        if p.brillo_ref > 0:
            k = p.brillo_ref / np.maximum(L_fondo, 1.0)
            dL, db = dL * k, db * k
        return dL, db

    def puntaje(self, imagen_bgr: np.ndarray) -> np.ndarray:
        dL, db = self.mapas(imagen_bgr)
        return -dL - self.p.peso_azul * db

    # ---------------------------------------------------------------- candidatos
    def mascara_candidatos(self, imagen_bgr: np.ndarray, aplicar_fijas: bool = False) -> np.ndarray:
        """Máscara uint8 (0/255) de los píxeles con puntaje > umbral, ya limpiada con una apertura."""
        p = self.p
        m = (self.puntaje(imagen_bgr) > p.umbral).astype(np.uint8)
        if p.apertura_px > 1:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.apertura_px, p.apertura_px)))
        if aplicar_fijas and self.mascara_fija is not None:
            m[self._fijas_dilatadas(m.shape)] = 0
        return m * 255

    def _fijas_dilatadas(self, forma) -> np.ndarray:
        fija = np.asarray(self.mascara_fija, dtype=bool)
        if fija.shape != tuple(forma[:2]):
            raise ValueError(f"mascara_fija {fija.shape} no coincide con la imagen {tuple(forma[:2])}")
        d = self.p.dilatacion_mascara_px
        if d > 0:
            fija = cv2.dilate(fija.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))) > 0
        return fija

    def detectar(self, imagen_bgr: np.ndarray) -> list[Deteccion]:
        return self.detectar_desde_mapas(*self.mapas(imagen_bgr))

    def detectar_desde_mapas(self, dL: np.ndarray, db: np.ndarray) -> list[Deteccion]:
        """Igual que detectar(), pero a partir de mapas() ya calculados (el calibrador prueba muchos
        umbrales sobre las mismas imágenes y así no repite la parte lenta)."""
        p = self.p
        alto, ancho = dL.shape
        puntaje = -dL - p.peso_azul * db
        mascara = (puntaje > p.umbral).astype(np.uint8)
        if p.apertura_px > 1:
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.apertura_px, p.apertura_px)))
        fijas = self._fijas_dilatadas(mascara.shape) if self.mascara_fija is not None else None
        n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
        salida: list[Deteccion] = []
        for i in range(1, n):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < p.area_min_px or area > p.area_max_px:
                continue
            region = etiquetas == i
            if fijas is not None and fijas[region].mean() > 0.5:
                continue        # mancha fija (polvo): está en el mismo lugar en todas las fotos
            # centro ponderado por el puntaje: es más estable que el centroide geométrico del umbral
            peso = (puntaje[region] - p.umbral + 1.0).astype(np.float64)
            filas, cols = np.nonzero(region)
            x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            toca = bool(x0 == 0 or y0 == 0 or x0 + w >= ancho or y0 + h >= alto)
            salida.append(Deteccion(
                col=float((cols * peso).sum() / peso.sum()), fila=float((filas * peso).sum() / peso.sum()),
                area_px=area, diametro_px=float(2 * np.sqrt(area / np.pi)), toca_borde=toca))
        salida.sort(key=lambda d: -d.area_px)
        return salida


def guardar_mascara_fija(mascara: np.ndarray, ruta) -> None:
    """Guarda la máscara como PNG (blanco = mancha fija), para abrirla con cualquier visor."""
    if not cv2.imwrite(str(ruta), np.asarray(mascara, dtype=bool).astype(np.uint8) * 255):
        raise OSError(f"No pude escribir {ruta}")


def cargar_mascara_fija(ruta) -> np.ndarray:
    img = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(ruta)
    return img > 127


def estimar_mascara_fija(imagenes: list[np.ndarray], params: ParametrosContraste | None = None,
                         frac_min: float = 0.5, min_imagenes: int = 6, umbral: float = 30.0) -> np.ndarray:
    """Píxeles que son candidatos en al menos `frac_min` de las imágenes (polvo, manchas de la lente).

    Las imágenes tienen que ser de campos DISTINTOS (la platina en lugares distintos): así la muestra
    cambia de una a otra y lo que se repite en el mismo píxel es del equipo, no de la muestra.
    `umbral` es el del puntaje con que se buscan las manchas (más bajo que el de detección a propósito).
    Devuelve una máscara bool alto x ancho (todo False si hay menos de `min_imagenes` imágenes).
    """
    p = replace(params or ParametrosContraste(), umbral=umbral)   # umbral bajo: caza también el polvo tenue
    alto, ancho = imagenes[0].shape[:2]
    if len(imagenes) < min_imagenes:
        return np.zeros((alto, ancho), dtype=bool)
    det = DetectorContraste(p)
    cuenta = np.zeros((alto, ancho), dtype=np.int32)
    for img in imagenes:
        if img.shape[:2] != (alto, ancho):
            raise ValueError("todas las imágenes tienen que tener el mismo tamaño")
        # cada mancha se agranda un poco: el polvo cae en el mismo píxel, pero su borde "baila" por el foco
        m = cv2.dilate(det.mascara_candidatos(img), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cuenta += (m > 0)
    return cuenta >= max(2, int(np.ceil(frac_min * len(imagenes))))
