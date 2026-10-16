"""
Microscopio simulado para probar el barrido sin hardware.

Mundo: un plano con fondo de glóbulos rojos (rosados), algunos "señuelos" magenta y puntos violetas
en posiciones conocidas. La platina solo se mueve a múltiplos de 12.5 µm. La cámara ve una ventana
del mundo con la escala REAL (que se puede hacer distinta de la calibrada para probar robustez).

Convención de ejes idéntica a centrado.py: x de la impresora <-> filas, y <-> columnas,
con los signos de `cal_real`.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from centrado import PASO_MM, Calibracion, a_mm, a_pasos

BGR_VIOLETA = (165, 55, 125)   # H ~ 139, S ~ 170 en OpenCV
BGR_RBC = (165, 165, 232)      # rosado-rojizo, H ~ 0
BGR_SENUELO = (125, 90, 215)   # magenta-rosado: H ~ 172, fuera del rango violeta por defecto (125..160)


class MicroscopioSimulado:
    def __init__(self, cal_real: Calibracion, puntos_mm: list[tuple[float, float]],
                 pos_inicial_mm: tuple[float, float] = (150.0, 150.0), radio_um: float = 4.0,
                 lado_mundo_um: float = 1400.0, um_por_px_mundo: float = 0.45, semilla: int = 0,
                 ruido: float = 3.0, carpeta_fotos: str | None = None, n_senuelos: int = 25):
        self.cal = cal_real
        self.puntos_mm = list(puntos_mm)
        self.ruido = ruido
        self.rng = np.random.default_rng(semilla)
        self.ix, self.iy = a_pasos(pos_inicial_mm[0]), a_pasos(pos_inicial_mm[1])
        self.s_mundo = um_por_px_mundo
        self.origen_um = (pos_inicial_mm[0] * 1000 - lado_mundo_um / 2, pos_inicial_mm[1] * 1000 - lado_mundo_um / 2)
        n = int(lado_mundo_um / um_por_px_mundo)
        self.mundo = self._construir_mundo(n, radio_um, n_senuelos)
        self.carpeta_fotos = Path(carpeta_fotos) if carpeta_fotos else None
        if self.carpeta_fotos:
            self.carpeta_fotos.mkdir(parents=True, exist_ok=True)
        self.fotos: list[dict] = []     # lo que se fotografió: para verificar en los tests
        self.movimientos = 0

    # ---------------------------------------------------------------- mundo
    def _a_px_mundo(self, x_um: float, y_um: float) -> tuple[float, float]:
        return (x_um - self.origen_um[0]) / self.s_mundo, (y_um - self.origen_um[1]) / self.s_mundo  # (fila, col)

    def _construir_mundo(self, n: int, radio_um: float, n_senuelos: int) -> np.ndarray:
        rng = self.rng
        img = np.full((n, n, 3), (200, 200, 240), np.uint8)       # fondo claro rosado
        # glóbulos rojos de fondo
        r_rbc = int(3.6 / self.s_mundo)
        for _ in range(int(n * n / (r_rbc * r_rbc * 14))):
            f, c = int(rng.integers(0, n)), int(rng.integers(0, n))
            col = tuple(int(np.clip(v + rng.normal(0, 8), 0, 255)) for v in BGR_RBC)
            cv2.circle(img, (c, f), int(r_rbc * rng.uniform(0.85, 1.15)), col, -1, cv2.LINE_AA)
        # señuelos magenta (no deben detectarse)
        for _ in range(n_senuelos):
            f, c = int(rng.integers(0, n)), int(rng.integers(0, n))
            cv2.circle(img, (c, f), r_rbc, BGR_SENUELO, -1, cv2.LINE_AA)
        # puntos violetas, algo irregulares
        for (x_mm, y_mm) in self.puntos_mm:
            f, c = self._a_px_mundo(x_mm * 1000, y_mm * 1000)
            ejes = (int(radio_um / self.s_mundo * rng.uniform(0.9, 1.1)), int(radio_um / self.s_mundo * rng.uniform(0.8, 1.0)))
            cv2.ellipse(img, (int(round(c)), int(round(f))), ejes, float(rng.uniform(0, 180)), 0, 360,
                        BGR_VIOLETA, -1, cv2.LINE_AA)
        return cv2.GaussianBlur(img, (0, 0), 1.0)

    # ---------------------------------------------------------------- interfaz Microscopio
    def posicion_mm(self) -> tuple[float, float]:
        return a_mm(self.ix), a_mm(self.iy)

    def mover_a(self, x_mm: float, y_mm: float) -> None:
        nx, ny = a_pasos(x_mm), a_pasos(y_mm)
        if (nx, ny) != (self.ix, self.iy):
            self.movimientos += 1
        self.ix, self.iy = nx, ny

    def enfocar(self) -> None:
        pass    # el simulador siempre está enfocado

    def _render(self) -> np.ndarray:
        c = self.cal
        x_um, y_um = self.ix * PASO_MM * 1000, self.iy * PASO_MM * 1000
        R0, C0 = self._a_px_mundo(x_um, y_um)
        ax = c.signo_y * self.s_mundo / c.um_por_px_col      # columnas de la imagen <- eje y
        ay = c.signo_x * self.s_mundo / c.um_por_px_fila     # filas de la imagen <- eje x
        M = np.array([[ax, 0, c.centro_col - ax * C0], [0, ay, c.centro_fila - ay * R0]], np.float64)
        vista = cv2.warpAffine(self.mundo, M, (c.ancho_px, c.alto_px), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        ruido = self.rng.normal(0, self.ruido, vista.shape)
        return np.clip(vista.astype(np.float32) + ruido, 0, 255).astype(np.uint8)

    def capturar_campo(self) -> np.ndarray:
        return self._render()

    def capturar_foto(self, nombre: str) -> str:
        img = cv2.resize(self._render(), None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        ruta = ""
        if self.carpeta_fotos:
            ruta = str(self.carpeta_fotos / f"{nombre}.png")
            cv2.imwrite(ruta, img)
        self.fotos.append({"pasos": (self.ix, self.iy), "archivo": ruta})
        return ruta

    # ---------------------------------------------------------------- para los tests
    def offset_um(self, punto_mm: tuple[float, float]) -> tuple[float, float]:
        """Dónde está el punto respecto del centro de la cámara ahora, en µm (ejes de la impresora)."""
        return ((punto_mm[0] - self.posicion_mm()[0]) * 1000, (punto_mm[1] - self.posicion_mm()[1]) * 1000)
