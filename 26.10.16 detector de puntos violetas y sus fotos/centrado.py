"""
Centrado de un píxel de la imagen usando solo posiciones que la impresora puede alcanzar,
y registro de las posiciones donde ya se sacó una foto.

Esta es la "habilidad" del microscopio: no depende de qué se detecta (puntos violetas,
parásitos, ...). Entra un píxel (col, fila) de la imagen actual y sale la posición
alcanzable de la impresora que deja a ese píxel lo más cerca posible del centro.

Geometría
---------
La impresora solo se para en múltiplos de PASO_MM (0.0125 mm = 12.5 µm). Los puntos
alcanzables forman una red cuadrada de parámetro a = 12.5 µm. La celda primitiva
(Wigner-Seitz) de cada nodo es el cuadrado de lado a centrado en el nodo, así que
"el objeto cae dentro de la celda del nodo al que fui" equivale a "ese nodo es el más
cercano al objeto", y eso es exactamente redondear al nodo más cercano:

    pasos = round(posición_deseada / a)      ->  |residuo| <= a/2 = 6.25 µm por eje

Todo se maneja en pasos ENTEROS (ix, iy) para comparar posiciones sin errores de punto flotante.

Ejes (misma convención que claude_generar_mosaico_fiji.py, con invertir_filas = invertir_columnas = False)
----------------------------------------------------------------------------------------------------
    eje X de la impresora  <->  filas de la imagen    (x positivo -> la foto baja en el mosaico)
    eje Y de la impresora  <->  columnas de la imagen (y positivo -> la foto va a la derecha)
Con signo_x = signo_y = +1, un objeto ubicado por debajo del centro de la imagen se centra
moviendo la impresora hacia x positivo, y uno a la derecha del centro, hacia y positivo.
Si algún día se cambia el montaje de la cámara y esto deja de valer, se cambia el signo en
Calibracion (o se lo deja medir con calibrar_escala()).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PASO_MM = 0.0125  # mínimo desplazamiento de la impresora en x-y


# --------------------------------------------------------------------------- calibración
@dataclass(frozen=True)
class Calibracion:
    ancho_px: int
    alto_px: int
    um_por_px_fila: float   # µm por píxel a lo largo de las filas  (eje X de la impresora)
    um_por_px_col: float    # µm por píxel a lo largo de las columnas (eje Y de la impresora)
    signo_x: int = +1       # +1: un objeto por DEBAJO del centro (fila > centro) se centra moviendo x positivo
    signo_y: int = +1       # +1: un objeto a la DERECHA del centro (col > centro) se centra moviendo y positivo
    paso_mm: float = PASO_MM

    @property
    def centro_col(self) -> float:
        return (self.ancho_px - 1) / 2

    @property
    def centro_fila(self) -> float:
        return (self.alto_px - 1) / 2

    @property
    def paso_um(self) -> float:
        return self.paso_mm * 1000.0

    @classmethod
    def desde_fov_mm(cls, ancho_px: int, alto_px: int, fov_x_mm: float, fov_y_mm: float,
                     **kw) -> "Calibracion":
        """Mismos fov_x_mm / fov_y_mm que usa generar_mosaico_fiji: campo visual (mm) a lo largo del
        eje X de la impresora (= filas de la imagen) y del eje Y (= columnas). ancho_px x alto_px es la
        resolución de la imagen en la que se detecta (p. ej. 320 x 240), no la de las fotos finales."""
        return cls(ancho_px, alto_px, um_por_px_fila=fov_x_mm * 1000 / alto_px,
                   um_por_px_col=fov_y_mm * 1000 / ancho_px, **kw)


def pixel_a_desplazamiento_um(cal: Calibracion, col: float, fila: float) -> tuple[float, float]:
    """Cuánto hay que mover la impresora (dx, dy) en µm para llevar el píxel al centro."""
    dx = cal.signo_x * (fila - cal.centro_fila) * cal.um_por_px_fila
    dy = cal.signo_y * (col - cal.centro_col) * cal.um_por_px_col
    return dx, dy


# --------------------------------------------------------------------------- cuantización
def a_pasos(mm: float, paso_mm: float = PASO_MM) -> int:
    """mm -> pasos enteros (redondeo al nodo más cercano; los empates van hacia arriba)."""
    return int(math.floor(mm / paso_mm + 0.5))


def a_mm(pasos: int, paso_mm: float = PASO_MM) -> float:
    return round(pasos * paso_mm, 6)


def ajustar_a_grilla_mm(mm: float, paso_mm: float = PASO_MM) -> float:
    return a_mm(a_pasos(mm, paso_mm), paso_mm)


@dataclass(frozen=True)
class ObjetivoCentrado:
    pasos_x: int            # posición alcanzable de destino, en pasos de 12.5 µm
    pasos_y: int
    x_mm: float
    y_mm: float
    residuo_um: tuple[float, float]       # (rx, ry): dónde queda el objeto respecto del centro tras mover, ejes de la impresora
    px_esperado: tuple[float, float]      # (col, fila) donde debería aparecer el objeto tras mover
    dentro_de_celda: bool                 # |residuo| <= paso/2 en ambos ejes (siempre True tras redondear)


def centrar_pixel(cal: Calibracion, pos_actual_mm: tuple[float, float], col: float, fila: float
                  ) -> ObjetivoCentrado:
    """Posición alcanzable que deja el píxel (col, fila) de la imagen tomada en pos_actual_mm
    lo más cerca posible del centro de la próxima imagen."""
    dx_um, dy_um = pixel_a_desplazamiento_um(cal, col, fila)
    tx_mm = pos_actual_mm[0] + dx_um / 1000.0     # donde está el objeto, en mm de la impresora
    ty_mm = pos_actual_mm[1] + dy_um / 1000.0
    ix, iy = a_pasos(tx_mm, cal.paso_mm), a_pasos(ty_mm, cal.paso_mm)
    x_mm, y_mm = a_mm(ix, cal.paso_mm), a_mm(iy, cal.paso_mm)
    rx = (tx_mm - x_mm) * 1000.0                  # objeto menos cámara
    ry = (ty_mm - y_mm) * 1000.0
    px_col = cal.centro_col + cal.signo_y * ry / cal.um_por_px_col
    px_fila = cal.centro_fila + cal.signo_x * rx / cal.um_por_px_fila
    mitad = cal.paso_um / 2 + 1e-6
    return ObjetivoCentrado(ix, iy, x_mm, y_mm, (rx, ry), (px_col, px_fila),
                            abs(rx) <= mitad and abs(ry) <= mitad)


def residuo_medido_um(cal: Calibracion, col: float, fila: float) -> tuple[float, float]:
    """Con el objeto ya detectado en una imagen tomada en la posición final: cuánto le falta
    al centro, en µm (ejes de la impresora). Está centrado si ambos valores son <= paso/2."""
    return pixel_a_desplazamiento_um(cal, col, fila)


def esta_centrado(cal: Calibracion, col: float, fila: float) -> bool:
    rx, ry = residuo_medido_um(cal, col, fila)
    mitad = cal.paso_um / 2 + 1e-6
    return abs(rx) <= mitad and abs(ry) <= mitad


# --------------------------------------------------------------------------- registro
class RegistroVisitados:
    """Posiciones (en pasos enteros) donde la impresora ya se paró para sacar una foto.

    Una posición candidata se considera repetida si está a <= tol_pasos pasos (por eje) de
    alguna ya registrada. Con tol_pasos = 1 eso son 12.5 µm por eje, que es el error de
    12.5 µm que se pidió tolerar.
    """

    def __init__(self, tol_pasos: int = 1, paso_mm: float = PASO_MM):
        self.tol_pasos = tol_pasos
        self.paso_mm = paso_mm
        self._celdas: set[tuple[int, int]] = set()
        self.entradas: list[dict] = []

    def __len__(self) -> int:
        return len(self.entradas)

    def ya_visitada(self, ix: int, iy: int) -> bool:
        t = self.tol_pasos
        return any((ix + a, iy + b) in self._celdas for a in range(-t, t + 1) for b in range(-t, t + 1))

    def registrar(self, paradas: list[tuple[int, int]], **info) -> None:
        """paradas: todas las posiciones donde se frenó para centrar ESTE objeto (la inicial y las
        correcciones). Se registran todas para que una vista posterior del mismo objeto,
        calculada con un pequeño error distinto, también se reconozca como repetida."""
        for p in paradas:
            self._celdas.add((int(p[0]), int(p[1])))
        self.entradas.append({"paradas": [[int(a), int(b)] for a, b in paradas], **info})

    def guardar(self, ruta: str | Path) -> None:
        datos = {"paso_mm": self.paso_mm, "tol_pasos": self.tol_pasos, "entradas": self.entradas}
        Path(ruta).write_text(json.dumps(datos, indent=1), encoding="utf-8")

    @classmethod
    def cargar(cls, ruta: str | Path) -> "RegistroVisitados":
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        r = cls(datos["tol_pasos"], datos["paso_mm"])
        for e in datos["entradas"]:
            r.registrar([tuple(p) for p in e.pop("paradas")], **e)
        return r


# --------------------------------------------------------------------------- medir la escala
def calibrar_escala(micro, x_mm: float, y_mm: float, ancho_px: int, alto_px: int,
                    n_pasos: int = 4, respuesta_min: float = 0.05) -> Calibracion:
    """Mide um/px y el signo de cada eje moviendo la impresora una distancia CONOCIDA
    (n_pasos * 12.5 µm) y midiendo cuánto se corre el contenido de la imagen (correlación de fase).

    Conviene hacerlo en una zona con textura (glóbulos), con n_pasos tal que el corrimiento sea
    de ~1/4 del campo (con 214 x 286 µm en QVGA, 4 pasos = 50 µm ≈ 56 px) y repetirlo en 2-3 lugares.
    Con corrimientos de casi media imagen o más, la correlación de fase puede enganchar un pico
    falso; en ese caso la función avisa y hay que repetir con menos pasos.
    `micro` es cualquier objeto con mover_a(x,y) y capturar_campo() (ver barrido_puntos.Microscopio).
    """
    def gris(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return g.astype(np.float32)

    ventana = cv2.createHanningWindow((ancho_px, alto_px), cv2.CV_32F)
    mov_mm = n_pasos * PASO_MM
    micro.mover_a(x_mm, y_mm)
    base = gris(micro.capturar_campo())
    res = {}
    for eje in ("x", "y"):
        micro.mover_a(x_mm + (mov_mm if eje == "x" else 0.0), y_mm + (mov_mm if eje == "y" else 0.0))
        otra = gris(micro.capturar_campo())
        (sx, sy), respuesta = cv2.phaseCorrelate(base, otra, ventana)  # corrimiento (col, fila) del contenido
        if respuesta < respuesta_min:
            raise RuntimeError(f"Correlación de fase poco confiable en el eje {eje} "
                               f"(respuesta={respuesta:.3f}); probar otra zona o más pasos.")
        res[eje] = (sx, sy)
    micro.mover_a(x_mm, y_mm)
    # Al mover la cámara +d en un eje, el contenido se corre -signo*d/escala en ese eje de la imagen.
    corr_fila = res["x"][1]   # el eje X mueve el contenido en filas
    corr_col = res["y"][0]    # el eje Y mueve el contenido en columnas
    if abs(corr_fila) > 0.4 * alto_px or abs(corr_col) > 0.4 * ancho_px:
        raise RuntimeError(f"Corrimiento medido demasiado grande (filas {corr_fila:.0f} px, columnas {corr_col:.0f} px): "
                           "repetir con menos pasos.")
    um = mov_mm * 1000.0
    return Calibracion(ancho_px, alto_px,
                       um_por_px_fila=um / abs(corr_fila), um_por_px_col=um / abs(corr_col),
                       signo_x=-1 if corr_fila > 0 else +1,
                       signo_y=-1 if corr_col > 0 else +1)
