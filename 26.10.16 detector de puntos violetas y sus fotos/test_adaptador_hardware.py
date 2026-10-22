"""
Prueba del adaptador SIN hardware, pero con TU código real:
  * TrimakerCOSMOS (impresora_cosmos.py), con un puerto serie falso que imita a Repetier
  * OpenMVCamera (camara_openmv.py), con la captura reemplazada por imágenes del simulador
  * busqueda_muestra.py: moverse_a_xy(), autofocus() y bytes_a_imagen() reales

    python test_adaptador_hardware.py

Busca esos tres módulos en la raíz de env_labo7 (variable de entorno ENV_LABO7, o la ruta de abajo).
No mueve nada ni abre ningún puerto.
"""
from __future__ import annotations

import math
import os
import re
import sys
import tempfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

RAIZ = os.environ.get("ENV_LABO7", r"C:\BackUp\Facultad\Labo 7\env_labo7")
sys.path.insert(0, RAIZ)
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import serial
except ImportError:                       # sin pyserial: módulo mínimo para poder importar impresora_cosmos
    import types
    serial = types.ModuleType("serial")
    sys.modules["serial"] = serial

from adaptador_hardware import MicroscopioCosmos
from busqueda_muestra import DISTANCIA_XY_SIN_SUBIR_MM as LIMITE_SALTO
from barrido_puntos import ConfigBarrido, barrer, campos_espiral
from camara_openmv import OpenMVCamera
from centrado import PASO_MM, Calibracion, a_pasos
from deteccion_puntos import DetectorColor
from impresora_cosmos import TrimakerCOSMOS
from simulador_microscopio import MicroscopioSimulado

Z0, Z_ARRIBA, Z_MIN = 27.71, 27.9, 27.5
OFFSET = (-19.0, -3.0, 0.0)               # offset_x, offset_y, offset_z de TrimakerCOSMOS


def z_foco(x: float, y: float) -> float:
    """Plano de enfoque ligeramente inclinado: obliga al autofoco a seguirlo."""
    return Z0 + 0.02 * (x - 150.0) + 0.01 * (y - 150.0)


class SerialFalso:
    """Imita lo justo de Repetier: G90/G91, G1, M400, M114, G28. Cada G1 termina al instante."""
    instancia = None

    def __init__(self, port=None, baudrate=None, timeout=None, **kw):
        self.cola: list[str] = []
        self.modo_abs = True
        self.raw = [0.0, 0.0, 0.0]        # coordenadas internas de la impresora (con offsets)
        self.log: list[str] = []
        self.dtr = self.rts = False
        self.is_open = True
        self.al_mover = lambda: None
        SerialFalso.instancia = self

    @property
    def in_waiting(self) -> int:
        return len(self.cola)

    def readline(self) -> bytes:
        return (self.cola.pop(0) + "\n").encode() if self.cola else b""

    def reset_input_buffer(self) -> None:
        self.cola.clear()

    def close(self) -> None:
        self.is_open = False

    def write(self, datos: bytes) -> int:
        for linea in datos.decode().splitlines():
            self.log.append(linea)
            p = linea.split()
            if not p:
                continue
            if p[0] == "G90":
                self.modo_abs = True
            elif p[0] == "G91":
                self.modo_abs = False
            elif p[0] in ("G0", "G1"):
                for tok in p[1:]:
                    if tok[0] in "XYZ":
                        i, v = "XYZ".index(tok[0]), float(tok[1:])
                        self.raw[i] = v if self.modo_abs else self.raw[i] + v
                self.al_mover()
                self.cola.append("ok")
            elif p[0] == "G28":
                self.raw = [0.0, 0.0, 0.0]
                self.cola.append("ok")
            elif p[0] == "M400":
                self.cola.append("ok")
            elif p[0] == "M114":
                self.cola.append(f"X:{self.raw[0]:.4f} Y:{self.raw[1]:.4f} Z:{self.raw[2]:.4f} E:0.0000")
        return len(datos)


class CamaraFalsa(OpenMVCamera):
    """OpenMVCamera real (enums, config_de_una, guardar_imagen) con la captura servida por el simulador."""

    def __init__(self, sim: MicroscopioSimulado, serie: SerialFalso):
        super().__init__(puerto="FALSA")
        self.sim, self.serie = sim, serie
        self.framesize = None
        self.comandos: list[str] = []
        self.capturas: list[dict] = []

    def conectar(self):
        pass

    def _check_connection(self):
        pass

    def _enviar_comando(self, comando):
        self.comandos.append(comando)
        if comando.startswith("set_framesize"):
            self.framesize = comando.split()[1]
        return True

    def capturar(self):
        x, y = self.sim.posicion_mm()
        z = self.serie.raw[2] - OFFSET[2]
        img = self.sim.capturar_campo()
        sigma = 400.0 * abs(z - z_foco(x, y))        # desenfoque proporcional a la distancia al plano de foco
        if sigma > 0.05:
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        if self.framesize == "WQXGA2":
            img = cv2.resize(img, None, fx=2, fy=2)
        self.capturas.append({"framesize": self.framesize, "x": x, "y": y, "z": z, "dz": z - z_foco(x, y)})
        return bytes(cv2.imencode(".jpg", img)[1])


def armar(carpeta: str, puntos_mm, nominal: Calibracion, real: Calibracion | None = None, **kw):
    """Impresora real sobre serie falso + cámara real con imágenes simuladas + adaptador."""
    sim = MicroscopioSimulado(real or nominal, puntos_mm, pos_inicial_mm=(150.0, 150.0), semilla=11)
    with mock.patch("serial.Serial", SerialFalso, create=True):
        grua = TrimakerCOSMOS("COM-FALSO")
    serie = SerialFalso.instancia
    serie.raw = [150.0 + OFFSET[0], 150.0 + OFFSET[1], Z0 + OFFSET[2]]
    grua.posicion_x, grua.posicion_y, grua.posicion_z = 150.0, 150.0, Z0     # como dejaría home() + posicionar_en_muestra()

    def al_mover():
        sim.ix = a_pasos(round(serie.raw[0] - OFFSET[0], 6))
        sim.iy = a_pasos(round(serie.raw[1] - OFFSET[1], 6))
    serie.al_mover = al_mover
    cam = CamaraFalsa(sim, serie)
    micro = MicroscopioCosmos(cam, grua, Path(carpeta) / "fotos", z_arriba_seguro=Z_ARRIBA, z_min_seguridad=Z_MIN, **kw)
    return micro, grua, serie, cam, sim


NOMINAL = Calibracion.desde_fov_mm(320, 240, fov_x_mm=0.214, fov_y_mm=0.286)


def test_saltos():
    with tempfile.TemporaryDirectory() as t:
        micro, *_ = armar(t, [], NOMINAL)
        assert micro._saltos(150.0, 150.0, 150.0, 150.0) == []
        rng = np.random.default_rng(0)
        for _ in range(300):
            x1, y1 = 150 + rng.uniform(-0.6, 0.6), 150 + rng.uniform(-0.6, 0.6)
            x1, y1 = round(a_pasos(x1) * PASO_MM, 6), round(a_pasos(y1) * PASO_MM, 6)
            pts = micro._saltos(150.0, 150.0, x1, y1)
            previo = (150.0, 150.0)
            for p in pts:
                assert math.hypot(p[0] - previo[0], p[1] - previo[1]) <= LIMITE_SALTO, (previo, p)
                previo = p
            assert not pts or pts[-1] == (x1, y1)
    print(f"ok  saltos cortos (<= {LIMITE_SALTO} mm, terminan exactamente en el destino)")


def test_mover_y_gcode():
    with tempfile.TemporaryDirectory() as t:
        micro, grua, serie, cam, sim = armar(t, [], NOMINAL)
        micro.autofoco = False

        def ordenes_xy():
            return sum(1 for l in serie.log if l.startswith("G1 X") and " Y" in l)

        # 1) destino a menos del límite: UN solo salto
        with mock.patch("time.sleep"):
            micro.mover_a(150.1125, 150.2125)            # dista ~0.24 mm
        assert math.hypot(0.1125, 0.2125) < LIMITE_SALTO
        assert micro.posicion_mm() == (150.1125, 150.2125), micro.posicion_mm()
        assert ordenes_xy() == 1, ordenes_xy()

        # 2) destino más allá del límite: varios saltos, posición exacta en la grilla y Z sin subir
        antes = ordenes_xy()
        x1, y1 = 150.5, 150.6
        dist = math.hypot(x1 - 150.1125, y1 - 150.2125)
        assert dist > LIMITE_SALTO
        with mock.patch("time.sleep"):
            micro.mover_a(x1, y1)
        assert micro.posicion_mm() == (x1, y1), micro.posicion_mm()
        assert (sim.ix, sim.iy) == (a_pasos(x1), a_pasos(y1))      # la platina quedó donde se pidió
        n_saltos = ordenes_xy() - antes
        assert n_saltos >= math.ceil(dist / LIMITE_SALTO), n_saltos
        zs = [float(l.split("Z")[1].split()[0]) for l in serie.log if l.startswith("G1") and " Z" in l]
        assert max(zs) <= Z0 + 1e-9, "no debería haber subido a la altura de viaje"
    print(f"ok  mover_a: 1 orden XY para 0.24 mm; {n_saltos} para {dist:.2f} mm (límite {LIMITE_SALTO} mm); "
          "posición exacta en grilla, Z sin subir")


def test_calibrar_escala_con_adaptador():
    from centrado import calibrar_escala
    real = Calibracion(320, 240, um_por_px_fila=0.93, um_por_px_col=0.87)      # distinta de la supuesta
    with tempfile.TemporaryDirectory() as t:
        micro, grua, serie, cam, sim = armar(t, [], NOMINAL, real=real)
        micro.autofoco = False
        with mock.patch("time.sleep"):
            est = calibrar_escala(micro, 150.0, 150.0, 320, 240)
        assert micro.posicion_mm() == (150.0, 150.0)                            # vuelve al punto de partida
    assert (est.signo_x, est.signo_y) == (+1, +1)
    assert abs(est.um_por_px_fila / 0.93 - 1) < 0.03 and abs(est.um_por_px_col / 0.87 - 1) < 0.03, est
    print(f"ok  calibrar_escala a través del adaptador: medido ({est.um_por_px_fila:.3f}, {est.um_por_px_col:.3f}) µm/px "
          f"vs real (0.930, 0.870)")


def test_barrido_con_hardware_falso():
    puntos = [(150.0 + x / 1000, 150.0 + y / 1000) for x, y in [(30, 50), (-120, -80), (60, 190)]]
    with tempfile.TemporaryDirectory() as t:
        micro, grua, serie, cam, sim = armar(t, puntos, NOMINAL)
        cfg = ConfigBarrido(carpeta_salida=str(Path(t) / "salida"))
        campos = list(campos_espiral(150.0, 150.0, 0.107, 0.143, n_radial=1))
        with mock.patch("time.sleep"):
            res, reg = barrer(micro, DetectorColor(), NOMINAL, campos, cfg, log=lambda *_: None)

        fotos = sorted((Path(t) / "fotos").glob("*.jpg"))
        assert len(fotos) == len(puntos) == res.fotos, (len(fotos), res)
        patron = re.compile(r"\(([-+0-9.eE]+),\s*([-+0-9.eE]+),\s*([-+0-9.eE]+)\)")   # el de extraer_coordenadas()
        centrados = 0
        for f in fotos:
            m = patron.search(f.stem)
            assert m, f.name
            x, y, _ = map(float, m.groups())
            d = min(math.hypot((p[0] - x) * 1000, (p[1] - y) * 1000) for p in puntos)
            centrados += d <= 9
        assert centrados == len(puntos)
        # enfocado al sacar cada foto: el Z quedó a <= 1 paso de Z (2.5 µm) del plano de foco
        # (config_de_una descarta 2 frames antes de cada foto: por eso hay 3 capturas en alta por foto)
        altas = [c for c in cam.capturas if c["framesize"] == "WQXGA2"]
        assert len(altas) == 3 * len(puntos) and all(abs(c["dz"]) <= 0.0026 for c in altas), [c["dz"] for c in altas]
        assert micro.autofocos_fallidos == 0
        assert all(Z_MIN <= c["z"] <= Z_ARRIBA for c in cam.capturas)
        cambios = sum(1 for c in cam.comandos if c.startswith("set_framesize"))
        print(f"ok  barrido con TrimakerCOSMOS/OpenMVCamera/autofocus reales: {res.fotos} fotos, "
              f"repetidos omitidos={res.omitidos_por_repetidos}, no verificados={res.no_verificados}, "
              f"|dz| al fotografiar <= {max(abs(c['dz']) for c in altas) * 1000:.1f} µm, "
              f"cambios de configuración de cámara={cambios}, capturas={len(cam.capturas)}")


def test_camara_tocada_por_fuera():
    """Se volvió a correr foco_final_multimetrica (que deja la cámara en alta resolución) sin recrear el adaptador:
    antes el detector recibía una foto de 5 MP ("mascara_fija (240, 320) no coincide con la imagen (1944, 2592)")."""
    from deteccion_contraste import DetectorContraste, ParametrosContraste
    puntos = [(150.0 + x / 1000, 150.0 + y / 1000) for x, y in [(30, 50), (-120, -80)]]
    det = lambda: DetectorContraste(ParametrosContraste(area_min_px=15), np.zeros((240, 320), bool))
    # 1) barrer() reconfigura la cámara al empezar
    with tempfile.TemporaryDirectory() as t:
        micro, grua, serie, cam, sim = armar(t, puntos, NOMINAL)
        micro.autofoco = False
        assert micro.capturar_campo().shape[:2] == (240, 320)
        cam.config_de_una(pixformat="RGB565", resolucion="WQXGA2")          # a espaldas del adaptador
        cfg = ConfigBarrido(carpeta_salida=str(Path(t) / "salida"))
        with mock.patch("time.sleep"):
            res, _ = barrer(micro, det(), NOMINAL, list(campos_espiral(150.0, 150.0, 0.107, 0.143, n_radial=1)), cfg,
                            log=lambda *_: None)
        assert res.fotos == len(puntos), res
    # 2) aunque no se pase por barrer(): capturar_campo() detecta el tamaño distinto y se corrige sola
    with tempfile.TemporaryDirectory() as t:
        micro, grua, serie, cam, sim = armar(t, puntos, NOMINAL)
        micro.autofoco = False
        assert micro.capturar_campo().shape[:2] == (240, 320)
        cam.config_de_una(pixformat="RGB565", resolucion="WQXGA2")
        assert micro.capturar_campo().shape[:2] == (240, 320)
        assert micro.capturar_campo().shape[:2] == (240, 320)
        cambios = sum(c.startswith("set_framesize") for c in cam.comandos)
        assert cambios == 3, cam.comandos                                   # QVGA, WQXGA2 (a mano), QVGA otra vez
    print("ok  cámara reconfigurada por fuera del adaptador: el barrido y capturar_campo() vuelven a QVGA solos")


if __name__ == "__main__":
    test_saltos()
    test_mover_y_gcode()
    test_calibrar_escala_con_adaptador()
    test_barrido_con_hardware_falso()
    test_camara_tocada_por_fuera()
    print("\nTodas las pruebas del adaptador pasaron.")
