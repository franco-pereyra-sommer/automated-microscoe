"""
Adaptador entre el barrido de puntos violetas (barrido_puntos.Microscopio) y tu hardware:
TrimakerCOSMOS (impresora_cosmos.py) + OpenMVCamera (camara_openmv.py), reutilizando
busqueda_muestra.py (moverse_a_xy, autofocus, bytes_a_imagen).

Requiere que la raíz de env_labo7 esté en sys.path (como en tus otros scripts).

Cómo respeta tus reglas:
  * XY siempre por moverse_a_xy() (umbral DISTANCIA_XY_SIN_SUBIR_MM, hoy 0.4 mm). Si el destino está más lejos,
    se parte en saltos cortos y se reenfoca entre salto y salto, igual que hace tu barrido espiral
    tile a tile. Nunca se dispara un "viaje largo" que suba la platina y deje todo desenfocado.
  * Después de cada movimiento: time.sleep(espera_movimiento) (0.5 s, como espera_movimiento_alta
    en barrido_espiral). Es lo que garantiza que la platina frenó mientras M400 no ande.
  * Las fotos se guardan con "(x, y, z)" en el nombre, el formato de extraer_coordenadas(), así
    que sirven directo para armar un mosaico.
"""
from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path

from busqueda_muestra import (DISTANCIA_XY_SIN_SUBIR_MM, PicoNoConfirmado, autofocus,
                              bytes_a_imagen, moverse_a_xy, varianza_intensidad)
from centrado import PASO_MM, ajustar_a_grilla_mm


class MicroscopioCosmos:
    """Implementa barrido_puntos.Microscopio sobre TrimakerCOSMOS + OpenMVCamera.

    camara, impresora : OpenMVCamera (ya conectada) y TrimakerCOSMOS (con home() hecho).
    carpeta_fotos     : donde se guardan las fotos de alta calidad.
    z_arriba_seguro, z_min_seguridad : los mismos que usás en posicionar_en_muestra().
    kwargs_campo      : se pasan a camara.config_de_una() para las imágenes de búsqueda (QVGA a color).
                        Conviene FIJAR acá exposure_us, gain_db y whitebal: con balance de blancos
                        automático el color cambia de un campo a otro y los umbrales de color del
                        detector dejan de valer.
    kwargs_foto       : ídem para la foto de alta calidad.
    """

    def __init__(self, camara, impresora, carpeta_fotos, z_arriba_seguro: float, z_min_seguridad: float,
                 resolucion_campo="QVGA", pixformat_campo="RGB565",
                 resolucion_foto="WQXGA2", pixformat_foto="RGB565",
                 kwargs_campo: dict | None = None, kwargs_foto: dict | None = None,
                 feedrate_xy: int = 3000, espera_movimiento: float = 0.5,
                 autofoco: bool = True, pasos_arriba_autofoco: int = 3, pasos_abajo_autofoco: int = 3,
                 paso_z_autofoco: float = 0.0025, metrica_foco=varianza_intensidad,
                 enfocar_entre_saltos: bool = True):
        if z_min_seguridad >= z_arriba_seguro:
            raise ValueError("z_min_seguridad tiene que ser menor que z_arriba_seguro")
        self.camara = camara
        self.impresora = impresora
        self.carpeta_fotos = Path(carpeta_fotos)
        self.carpeta_fotos.mkdir(parents=True, exist_ok=True)
        self.z_arriba_seguro = z_arriba_seguro
        self.z_min_seguridad = z_min_seguridad
        self.resolucion_campo, self.pixformat_campo = resolucion_campo, pixformat_campo
        self.resolucion_foto, self.pixformat_foto = resolucion_foto, pixformat_foto
        self.kwargs_campo = dict(kwargs_campo or {})
        self.kwargs_foto = dict(kwargs_foto or {})
        self.feedrate_xy = feedrate_xy
        self.espera_movimiento = espera_movimiento
        self.autofoco = autofoco
        self.pasos_arriba_autofoco = pasos_arriba_autofoco
        self.pasos_abajo_autofoco = pasos_abajo_autofoco
        self.paso_z_autofoco = paso_z_autofoco
        self.metrica_foco = metrica_foco
        self.enfocar_entre_saltos = enfocar_entre_saltos
        self.autofocos_fallidos = 0
        self._modo: str | None = None     # "campo" | "foto": evita reconfigurar la cámara si no cambió

    # ------------------------------------------------------------------ cámara
    def invalidar_modo(self) -> None:
        """Llamalo si tocás la configuración de la cámara a mano: fuerza a reconfigurar en la próxima captura."""
        self._modo = None

    def _asegurar_modo(self, modo: str) -> None:
        if self._modo == modo:
            return
        if modo == "campo":
            self.camara.config_de_una(pixformat=self.pixformat_campo, resolucion=self.resolucion_campo,
                                      **self.kwargs_campo)
        elif modo == "foto":
            self.camara.config_de_una(pixformat=self.pixformat_foto, resolucion=self.resolucion_foto,
                                      **self.kwargs_foto)
        else:
            raise ValueError(modo)
        self._modo = modo

    # ------------------------------------------------------------------ interfaz Microscopio
    def posicion_mm(self) -> tuple[float, float]:
        x, y = self.impresora.posicion_x, self.impresora.posicion_y
        if x is None or y is None:
            raise RuntimeError("Llamá primero a impresora.home()")
        return float(x), float(y)

    def _saltos(self, x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
        """Puntos intermedios (sobre la grilla de 12.5 µm) tales que cada salto mida <= DISTANCIA_XY_SIN_SUBIR_MM,
        contando el redondeo a la grilla, para que moverse_a_xy() nunca lo trate como viaje largo."""
        distancia = math.hypot(x1 - x0, y1 - y0)
        if distancia < PASO_MM / 2:
            return []
        limite = DISTANCIA_XY_SIN_SUBIR_MM - 1e-6
        n = max(1, math.ceil(distancia / limite))
        while n <= 400:
            puntos = [(ajustar_a_grilla_mm(x0 + (x1 - x0) * k / n), ajustar_a_grilla_mm(y0 + (y1 - y0) * k / n))
                      for k in range(1, n + 1)]
            puntos[-1] = (x1, y1)
            previo, ok = (x0, y0), True
            for p in puntos:
                if math.hypot(p[0] - previo[0], p[1] - previo[1]) > limite:
                    ok = False
                    break
                previo = p
            if ok:
                return puntos
            n += 1
        raise RuntimeError("No pude partir el movimiento en saltos cortos")

    def mover_a(self, x_mm: float, y_mm: float) -> None:
        x1, y1 = ajustar_a_grilla_mm(x_mm), ajustar_a_grilla_mm(y_mm)
        x0, y0 = self.posicion_mm()
        saltos = self._saltos(x0, y0, x1, y1)
        for k, (xk, yk) in enumerate(saltos):
            viaje_largo = moverse_a_xy(self.impresora, xk, yk, self.z_arriba_seguro,
                                       feedrate_xy=self.feedrate_xy, espera_movimiento=self.espera_movimiento)
            if viaje_largo:   # no debería pasar: los saltos son <= DISTANCIA_XY_SIN_SUBIR_MM
                raise RuntimeError(f"moverse_a_xy() lo trató como viaje largo yendo a ({xk}, {yk}); "
                                   "la platina quedó levantada y sin enfocar.")
            xr, yr = self.posicion_mm()
            if abs(xr - xk) > PASO_MM / 2 or abs(yr - yk) > PASO_MM / 2:
                raise RuntimeError(f"La impresora no llegó a ({xk}, {yk}) y quedó en ({xr}, {yr}) "
                                   "(¿destino fuera de rango? desp_lin_abs cancela esas órdenes sin avisar).")
            if k < len(saltos) - 1 and self.enfocar_entre_saltos:
                self.enfocar()

    def enfocar(self) -> None:
        """Autofoco local alrededor del Z actual (el mismo autofocus() de tu barrido espiral).
        Deja la platina en el mejor Z. Si no confirma un máximo, avisa y deja el Z como estaba."""
        if not self.autofoco:
            return
        self._asegurar_modo("campo")
        try:
            autofocus(camara=self.camara, impresora=self.impresora, metrica=self.metrica_foco,
                      pasos_arriba=self.pasos_arriba_autofoco, pasos_abajo=self.pasos_abajo_autofoco,
                      paso_z=self.paso_z_autofoco, espera_movimiento=self.espera_movimiento,
                      graficamos=False, guardamos=False,
                      z_min_seguridad=self.z_min_seguridad, z_max_seguro=self.z_arriba_seguro)
        except PicoNoConfirmado as e:
            self.autofocos_fallidos += 1
            print(f"[AVISO] Autofoco sin pico confirmado en {self.posicion_mm()}; sigo con el Z actual. ({e})")

    def capturar_campo(self):
        """Imagen BGR de la búsqueda (QVGA a color)."""
        self._asegurar_modo("campo")
        imagen = bytes_a_imagen(self.camara.capturar())
        if imagen.ndim != 3:
            raise ValueError("El detector de color necesita una imagen a color: usá pixformat_campo='RGB565'.")
        return imagen

    def capturar_foto(self, nombre: str) -> str:
        """Foto de alta calidad. Se guarda como '<nombre> (x, y, z) - <fecha>.jpg'."""
        self._asegurar_modo("foto")
        image_data = self.camara.capturar()
        hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        x, y, z = self.impresora.pregunta_posicion_actual()
        archivo = self.camara.guardar_imagen(image_data=image_data, carpeta=self.carpeta_fotos,
                                             nombre=f"{nombre} ({x}, {y}, {z}) - {hora}", mostrar=False)
        return str(archivo)
