# %%

"""
Simuladores de hardware compartidos por las pruebas de esta carpeta.

Para qué sirven: probar código nuevo (funciones que mueven la platina o piden
fotos) SIN tocar el equipo. Antes de escribir un simulador nuevo en una
aplicación, mirar si alguno de estos alcanza (regla en REGLAS_CODIGO.txt).

Qué hay:
  - preparar_entorno(): agrega la raíz del repo al sys.path, fuerza el backend
    "Agg" de matplotlib (sin ventanas) y, si pyserial no está instalado,
    registra un módulo `serial` de mentira para poder importar los módulos base.
    Se ejecuta sola al importar este archivo.
  - sin_esperas(): contexto que anula time.sleep (los módulos base duermen
    segundos entre movimientos).
  - SerialFalso + crear_impresora_falsa(): la clase REAL TrimakerCOSMOS (límites,
    offsets, grilla de 12.5 / 2.5 µm, espera de fin de movimiento) corriendo
    sobre un puerto serie de mentira que guarda el G-code que se le manda.
  - SerialCamaraFalso: emula el firmware main_openmv.py (OK/ERR, info, capture)
    para probar la clase REAL OpenMVCamera.
  - CamaraFalsa: reemplaza a OpenMVCamera en las funciones que combinan
    cámara + impresora; capturar() devuelve un JPEG cuyo enfoque depende de la
    Z de la impresora (hay un plano de foco simulado).

Límites (lo que NO simulan): tiempos reales de movimiento, el comportamiento
de M400/M114 del firmware Repetier (ni cuántas líneas de posición emite G28:
ver SerialFalso), ruido y óptica reales. Que una prueba pase con estos
simuladores no reemplaza la prueba en el equipo.

Creado el 2026-10-27.
"""

import io
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parent.parent


def preparar_entorno():
    """Deja todo listo para importar los módulos base sin hardware."""

    if str(RAIZ) not in sys.path:
        sys.path.insert(0, str(RAIZ))

    import matplotlib
    matplotlib.use("Agg")

    try:
        import serial  # noqa: F401
    except ImportError:
        falso = types.ModuleType("serial")

        class Serial:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "pyserial no está instalado: este es el módulo `serial` "
                    "de mentira de tests/simuladores.py"
                )

        falso.Serial = Serial
        sys.modules["serial"] = falso


preparar_entorno()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


class Omitida(Exception):
    """Marca una prueba como omitida (correr_todo.py la cuenta aparte de las que fallan)."""


def omitir(motivo):
    raise Omitida(motivo)


@contextmanager
def sin_esperas():
    """Anula time.sleep mientras dura el bloque `with`."""
    with mock.patch("time.sleep", lambda segundos: None):
        yield


def a_jpeg(arreglo, modo):
    """Bytes JPEG de un array uint8. modo: "L" (gris 2D) o "RGB" (color 3D)."""
    buffer = io.BytesIO()
    Image.fromarray(arreglo, modo).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


# ======================================================================
# IMPRESORA
# ======================================================================

class SerialFalso:
    """
    Puerto serie de la impresora. Guarda cada comando que recibe (sin el
    salto de línea) en `escrituras` y contesta con líneas de posición, que es
    lo que TrimakerCOSMOS.esperar_fin_movimiento() espera para dar por
    terminado un movimiento:
      - M114 -> 1 línea.
      - G28  -> `lineas_por_g28` líneas (por defecto 2).

    Cuántas líneas emite G28 en el equipo real NO está confirmado: Franco
    sospecha que 2 (ver BITACORA.txt, pendiente "home() y el fin de movimiento").
    El valor por defecto acompaña al home() vigente, que lee dos. Si se mide en
    el equipo otro número, se cambia acá y en home() juntos: así
    test_impresora.py avisa si quedan desalineados (líneas sin leer en el buffer).
    """

    def __init__(self, lineas_por_g28=2):
        self.lineas_por_g28 = lineas_por_g28
        self.escrituras = []
        self._pendientes = []
        self.dtr = False
        self.rts = False
        self.abierto = True

    @property
    def in_waiting(self):
        return sum(len(linea) for linea in self._pendientes)

    def write(self, datos):
        comando = datos.decode().strip()
        self.escrituras.append(comando)
        if comando == "M114":
            self._pendientes.append(b"X:0.00 Y:0.00 Z:0.00 E:0.00\n")
        elif comando == "G28":
            for _ in range(self.lineas_por_g28):
                self._pendientes.append(b"X:0.00 Y:0.00 Z:0.00 E:0.00\n")

    def readline(self):
        return self._pendientes.pop(0) if self._pendientes else b""

    def reset_input_buffer(self):
        self._pendientes.clear()

    def close(self):
        self.abierto = False


def crear_impresora_falsa(hacer_home=True, lineas_por_g28=2):
    """
    TrimakerCOSMOS real conectada a un SerialFalso (queda en `impresora.ser`).
    Con hacer_home=True ya hizo home(): queda en (0, 0, 40) y `ser.escrituras`
    está vacío. `lineas_por_g28`: ver SerialFalso.
    """

    import impresora_cosmos

    serial_falso = SerialFalso(lineas_por_g28=lineas_por_g28)

    with sin_esperas(), mock.patch.object(
        impresora_cosmos.serial, "Serial", lambda **kwargs: serial_falso
    ):
        impresora = impresora_cosmos.TrimakerCOSMOS("COM_FALSO")

        # home() escucha 5 s de reloj real; acá no hay nada que escuchar.
        impresora.escuchar = lambda tiempo_maximo=10: []

        if hacer_home:
            impresora.home()

    serial_falso.escrituras.clear()

    return impresora


def movimientos_g1(impresora):
    """
    Los G1 que recibió la impresora, como lista de dicts {eje: valor}.
    Ej.: "G1 X81.0 Y97.0 F3000" -> {"X": 81.0, "Y": 97.0, "F": 3000.0}
    """

    movimientos = []

    for linea in impresora.ser.escrituras:
        if linea.startswith("G1 "):
            movimientos.append(
                {token[0]: float(token[1:]) for token in linea.split()[1:]}
            )

    return movimientos


# ======================================================================
# CÁMARA
# ======================================================================

class SerialCamaraFalso:
    """
    Puerto serie de la cámara, emulando el firmware main_openmv.py: OK a los
    set_*, ERR a lo desconocido, `info` con exposición/ganancias y `capture`
    con el protocolo de dos pasos (tamaño, "ok", bytes).
    """

    COMANDOS_OK = {
        "set_pixformat", "set_framesize", "set_quality",
        "set_exposure_auto", "set_gain_auto", "set_whitebal_auto",
    }

    def __init__(self, jpeg=None):
        self.escrituras = []
        self.is_open = True
        self._salida = bytearray()
        self._jpeg = jpeg if jpeg is not None else a_jpeg(
            np.full((24, 32), 128, np.uint8), "L"
        )

    def write(self, datos):
        comando = datos.decode().strip()
        self.escrituras.append(comando)

        if comando == "capture":
            self._salida += f"{len(self._jpeg)}\n".encode()
        elif comando == "ok":
            self._salida += self._jpeg
        elif comando == "info":
            self._salida += b"50000 25.0 (1.0, 1.0, 1.0)\n"
        elif comando.split()[0] in self.COMANDOS_OK:
            self._salida += b"OK\n"
        else:
            self._salida += b"ERR\n"

    def readline(self):
        fin = self._salida.find(b"\n")
        if fin < 0:
            fin = len(self._salida) - 1
        dato = bytes(self._salida[: fin + 1])
        del self._salida[: fin + 1]
        return dato

    def read(self, cantidad):
        dato = bytes(self._salida[:cantidad])
        del self._salida[:cantidad]
        return dato

    def reset_input_buffer(self):
        self._salida.clear()

    def close(self):
        self.is_open = False


class CamaraFalsa:
    """
    Reemplaza a OpenMVCamera en las funciones que combinan cámara + impresora.

    capturar() devuelve los bytes de un JPEG de una textura aleatoria que se
    borronea más cuanto más lejos está la Z de la impresora de `z_foco`:
    la métrica de enfoque tiene su máximo en `z_foco`.

    con_muestra=False simula un campo vacío (imagen clara y lisa).
    """

    def __init__(self, impresora, z_foco, con_muestra=True,
                 alto=240, ancho=320, semilla=0, px_borroneo_por_mm=120.0):
        self.impresora = impresora
        self.z_foco = z_foco
        self.con_muestra = con_muestra
        self.px_borroneo_por_mm = px_borroneo_por_mm

        self.pixformat = "GRAYSCALE"
        self.framesize = "QVGA"
        self.capturas = 0
        self.archivos_guardados = []
        self.llamadas_config_de_una = []

        generador = np.random.default_rng(semilla)
        self._textura = np.clip(
            generador.normal(128, 50, (alto, ancho)), 0, 255
        ).astype(np.float32)

    def _imagen(self):
        if not self.con_muestra:
            return np.full(self._textura.shape, 235, np.uint8)

        distancia = abs(float(self.impresora.posicion_z) - self.z_foco)
        sigma = max(self.px_borroneo_por_mm * distancia, 0.01)

        borrosa = cv2.GaussianBlur(self._textura, (0, 0), sigma)

        return np.clip(borrosa, 0, 255).astype(np.uint8)

    def capturar(self):
        self.capturas += 1
        gris = self._imagen()

        if self.pixformat == "GRAYSCALE":
            return a_jpeg(gris, "L")

        return a_jpeg(np.dstack([gris, gris, gris]), "RGB")

    def set_pixformat(self, formato):
        self.pixformat = str(getattr(formato, "value", formato)).upper()

    def set_framesize(self, resolucion):
        self.framesize = str(getattr(resolucion, "value", resolucion)).upper()

    def config_de_una(self, **kwargs):
        self.llamadas_config_de_una.append(kwargs)
        if "pixformat" in kwargs:
            self.set_pixformat(kwargs["pixformat"])
        if "resolucion" in kwargs:
            self.set_framesize(kwargs["resolucion"])

    def descartar_frames(self, cantidad=2):
        pass

    def guardar_imagen(self, image_data, carpeta, mostrar=False, nombre=None):
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        archivo = carpeta / f"{nombre or len(self.archivos_guardados)}.jpg"
        archivo.write_bytes(image_data)
        self.archivos_guardados.append(archivo)
        return archivo


# %%
