"""
Barrido en busca de puntos violetas (o de lo que devuelva el detector).

Lazo:  ir a un campo -> detectar -> para cada objeto: calcular la posición alcanzable que lo
centra -> si ya se sacó una foto ahí, saltear; si no, ir, verificar el centrado, sacar la foto
de alta calidad y registrar la posición.

Los objetivos de TODOS los objetos del campo se calculan antes de mover la impresora, porque
después del primer movimiento los píxeles de la imagen original ya no corresponden a la vista actual.
Una posición absoluta, en cambio, sigue valiendo.

Hardware: el barrido solo usa los métodos de `Microscopio`. El adaptador para TrimakerCOSMOS +
OpenMVCamera es adaptador_hardware.MicroscopioCosmos. mover_a tiene que volver recién cuando la
platina frenó (el adaptador lo garantiza con la espera por tiempo de tus scripts).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Protocol

import numpy as np

from centrado import (PASO_MM, Calibracion, RegistroVisitados, a_pasos, ajustar_a_grilla_mm,
                      centrar_pixel, esta_centrado, pixel_a_desplazamiento_um)
from deteccion_puntos import Detector


class Microscopio(Protocol):
    def posicion_mm(self) -> tuple[float, float]:
        """(x, y) actual de la impresora, en mm."""
    def mover_a(self, x_mm: float, y_mm: float) -> None:
        """Mueve a la posición absoluta (x, y) y vuelve cuando la platina ya frenó."""
    def enfocar(self) -> None:
        """Reenfoca en Z (el foco se pierde rápido al moverse). Puede ser un no-op."""
    def capturar_campo(self) -> np.ndarray:
        """Imagen BGR de baja resolución (la misma resolución con la que se calibró)."""
    def capturar_foto(self, nombre: str) -> str:
        """Foto de alta calidad; guarda el archivo y devuelve su ruta."""


@dataclass
class ConfigBarrido:
    carpeta_salida: str
    ignorar_toca_borde: bool = True     # un objeto cortado por el borde tiene el centro sesgado; se lo verá entero desde otro campo
    verificar: bool = True              # tras moverse, volver a detectar y confirmar que quedó en la celda
    max_correcciones: int = 2           # movimientos extra permitidos si no quedó centrado
    radio_busqueda_px: float = 40.0     # al verificar: distancia máxima al píxel esperado para aceptar la detección
    guardar_registro: bool = True       # escribe registro_visitados.json tras cada foto (permite retomar)


@dataclass
class ResultadoBarrido:
    campos_visitados: int = 0
    objetos_detectados: int = 0
    ignorados_por_borde: int = 0
    omitidos_por_repetidos: int = 0
    fotos: int = 0
    no_verificados: int = 0
    segundos: float = 0.0


def campos_espiral(x0_mm: float, y0_mm: float, paso_x_mm: float, paso_y_mm: float,
                   n_radial: int) -> Iterator[tuple[float, float]]:
    """Posiciones (x, y) de los campos, en espiral cuadrada alrededor de (x0, y0).
    Cubre (2*n_radial+1)^2 campos. Cada posición absoluta se ajusta a la grilla de 12.5 µm,
    así que pasos que no son múltiplos (0.07 mm = 5.6 pasos) no acumulan error."""
    i = j = 0
    di, dj = 0, 1
    largo = 1
    total = (2 * n_radial + 1) ** 2
    cuenta = 1
    yield ajustar_a_grilla_mm(x0_mm), ajustar_a_grilla_mm(y0_mm)
    while cuenta < total:
        for _ in range(2):
            for _ in range(largo):
                i, j = i + di, j + dj
                yield ajustar_a_grilla_mm(x0_mm + i * paso_x_mm), ajustar_a_grilla_mm(y0_mm + j * paso_y_mm)
                cuenta += 1
                if cuenta >= total:
                    return
            di, dj = dj, -di
        largo += 1


def _centrar_y_fotografar(micro: Microscopio, detector: Detector, cal: Calibracion, cfg: ConfigBarrido,
                          objetivo, n_foto: int, res: ResultadoBarrido):
    """Va al objetivo, verifica/corrige el centrado y saca la foto. Devuelve (paradas, info)."""
    paradas = []
    verificado: bool | None = None if not cfg.verificar else False
    residuo_um = None
    for intento in range(cfg.max_correcciones + 1):
        micro.mover_a(objetivo.x_mm, objetivo.y_mm)
        if intento == 0:
            micro.enfocar()     # una vez por objeto; las correcciones son de unos pocos pasos de 12.5 µm
        pos = micro.posicion_mm()
        paradas.append((a_pasos(pos[0], cal.paso_mm), a_pasos(pos[1], cal.paso_mm)))
        if not cfg.verificar:
            break
        # Buscar el objeto cerca del píxel donde debería estar
        cand = detector.detectar(micro.capturar_campo())
        ec, ef = objetivo.px_esperado
        cercanos = [d for d in cand if np.hypot(d.col - ec, d.fila - ef) <= cfg.radio_busqueda_px]
        if not cercanos:
            verificado = False
            break
        d = min(cercanos, key=lambda d: np.hypot(d.col - ec, d.fila - ef))
        residuo_um = pixel_a_desplazamiento_um(cal, d.col, d.fila)
        if esta_centrado(cal, d.col, d.fila):
            verificado = True
            break
        if intento == cfg.max_correcciones:
            break
        objetivo = centrar_pixel(cal, pos, d.col, d.fila)  # corrección: mover a la celda correcta
    pos_final = paradas[-1]
    nombre = f"punto_{n_foto:04d}_ix{pos_final[0]}_iy{pos_final[1]}"
    archivo = micro.capturar_foto(nombre)
    if verificado is False:
        res.no_verificados += 1
    return paradas, {"archivo": str(archivo), "verificado": verificado,
                     "residuo_um": None if residuo_um is None else [round(residuo_um[0], 2), round(residuo_um[1], 2)]}


def barrer(micro: Microscopio, detector: Detector, cal: Calibracion, campos: Iterable[tuple[float, float]],
           cfg: ConfigBarrido, registro: RegistroVisitados | None = None, log=print) -> tuple[ResultadoBarrido, RegistroVisitados]:
    registro = registro if registro is not None else RegistroVisitados(paso_mm=cal.paso_mm)
    Path(cfg.carpeta_salida).mkdir(parents=True, exist_ok=True)
    ruta_registro = Path(cfg.carpeta_salida) / "registro_visitados.json"
    res = ResultadoBarrido()
    t0 = time.time()

    for (xc, yc) in campos:
        micro.mover_a(xc, yc)
        micro.enfocar()
        pos_campo = micro.posicion_mm()          # posición REAL (ya cuantizada) donde se tomó la imagen
        dets = detector.detectar(micro.capturar_campo())
        res.campos_visitados += 1
        res.objetos_detectados += len(dets)

        # 1) objetivos de todos los objetos, calculados desde esta vista
        objetivos = []
        for d in dets:
            if cfg.ignorar_toca_borde and d.toca_borde:
                res.ignorados_por_borde += 1
                continue
            objetivos.append((centrar_pixel(cal, pos_campo, d.col, d.fila), d))

        # 2) visitarlos en orden de cercanía (vecino más próximo, partiendo de donde está la impresora)
        actual = (a_pasos(pos_campo[0], cal.paso_mm), a_pasos(pos_campo[1], cal.paso_mm))
        while objetivos:
            k = min(range(len(objetivos)),
                    key=lambda i: (objetivos[i][0].pasos_x - actual[0]) ** 2 + (objetivos[i][0].pasos_y - actual[1]) ** 2)
            obj, d = objetivos.pop(k)
            # El chequeo va acá (y no antes del bucle) para que dos objetos de un mismo campo cuyo
            # destino coincide también se detecten como repetidos.
            if registro.ya_visitada(obj.pasos_x, obj.pasos_y):
                res.omitidos_por_repetidos += 1
                continue
            paradas, info = _centrar_y_fotografar(micro, detector, cal, cfg, obj, res.fotos + 1, res)
            res.fotos += 1
            actual = paradas[-1]
            registro.registrar(paradas, campo_mm=[pos_campo[0], pos_campo[1]],
                               pixel_detectado=[round(d.col, 2), round(d.fila, 2)],
                               area_px=d.area_px, **info)
            if cfg.guardar_registro:
                registro.guardar(ruta_registro)
            log(f"foto {res.fotos}: pasos {paradas[-1]}  verificado={info['verificado']}")

    res.segundos = time.time() - t0
    return res, registro
