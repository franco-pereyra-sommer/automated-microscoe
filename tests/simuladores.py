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
    Z de la impresora (hay un plano de foco simulado). info() informa lo último
    que recibió config_de_una() (agregado el 2026-11-09, para 26.11.09).
  - CamaraFalsaConMuestra: una CamaraFalsa que VE una muestra (glóbulos y
    puntos violetas al azar): la imagen depende de la posición (x, y) de la
    impresora, con una escala µm/px y unos signos de ejes configurables. Sirve
    para probar el centrado. Opcionalmente devuelve cuadros viejos (como una
    cámara con el buffer sin vaciar) y una textura fina (`grano`, para escalas muy
    chicas). Agregada el 2026-10-29.

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
        self.llamadas_info = 0
        self.info_falla = False

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

    def config_de_una(self, dict_params, calidad=100, skip=2):
        """
        Igual que OpenMVCamera.config_de_una(): recibe el diccionario de parámetros y lo valida con la
        MISMA función que la cámara real (así una llamada con el formato viejo, o con un diccionario
        incompleto, falla acá como en el equipo). Guarda una copia de cada diccionario recibido, en orden,
        en `llamadas_config_de_una`.
        """
        import camara_openmv

        camara_openmv.validar_dict_params_camara(dict_params)
        self.llamadas_config_de_una.append(dict(dict_params))
        self.set_pixformat(dict_params["pixformat"])
        self.set_framesize(dict_params["resolucion"])

    def descartar_frames(self, cantidad=2):
        pass

    def info(self):
        """
        Igual que OpenMVCamera.info(): lo que la cámara informa de sí misma. Devuelve lo último que recibió
        config_de_una() (None = automático se informa como 18000 us / 35 dB / balance neutro) y cuenta las
        llamadas. Con `info_falla = True` levanta Exception, como la cámara real cuando no contesta.
        """
        self.llamadas_info += 1

        if self.info_falla:
            raise Exception("No se recibió información de la cámara")

        ultimo = self.llamadas_config_de_una[-1] if self.llamadas_config_de_una else {}
        whitebal = ultimo.get("whitebal") or (60.206, 60.206, 60.206)

        return {
            "exposure_us": int(ultimo.get("exposure_us") or 18000),
            "gain_db": float(ultimo.get("gain_db") or 35.0),
            "rgb_gain_db": tuple(float(canal) for canal in whitebal),
        }

    def guardar_imagen(self, image_data, carpeta, mostrar=False, nombre=None):
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        archivo = carpeta / f"{nombre or len(self.archivos_guardados)}.jpg"
        archivo.write_bytes(image_data)
        self.archivos_guardados.append(archivo)
        return archivo


class CamaraFalsaConMuestra(CamaraFalsa):
    """
    Cámara falsa que VE una muestra: capturar() devuelve un JPEG a color de lo
    que hay bajo la cámara en la posición (x, y) de la impresora
    (impresora.posicion_x / posicion_y). Mover la impresora corre la imagen
    como en el equipo, así que sirve para probar el centrado.

    La muestra es un plano con glóbulos rosados al azar y algunos puntos
    violetas, centrado en `origen_mm`. La escala (um_por_px_*) y los signos son
    los REALES del microscopio simulado y pueden ser distintos de los que supone
    el código que se prueba (así se comprueba que una escala mal puesta se nota).
    Ejes (PROYECTO_CONTEXTO.txt): x de la impresora = filas de la imagen, y =
    columnas; con signo +1, subir x lleva la cámara hacia abajo sobre la muestra.

    frames_de_retraso: capturar() devuelve el cuadro de hace N capturas (una
    cámara con cuadros viejos en el buffer). config_de_una() y descartar_frames()
    sacan cuadros como en el equipo (config_de_una descarta `skip`), que es lo
    que vacía ese buffer.

    grano: desvío (en niveles de gris) de una textura fina fija que se le suma a la muestra (0 = ninguna, como antes).
    Con escalas muy chicas (0.1 µm/px: un parche de 49 px mide 5 µm) los glóbulos solos dejan parches lisos y no
    hay de dónde agarrarse; el grano hace de la textura que tiene una muestra real. (Agregado el 2026-11-12.)

    La Z no influye: la imagen siempre está enfocada.
    """

    def __init__(self, impresora, origen_mm=(150.0, 150.0), um_por_px_fila=0.386, um_por_px_col=0.388,
                 signo_x=1, signo_y=1, alto=240, ancho=320, lado_um=500.0, um_por_px_mundo=0.25,
                 frames_de_retraso=0, ruido=2.0, semilla=0, grano=0.0):
        super().__init__(impresora, z_foco=0.0, alto=alto, ancho=ancho, semilla=semilla)
        self.origen_mm = origen_mm
        self.um_por_px_fila, self.um_por_px_col = um_por_px_fila, um_por_px_col
        self.signo_x, self.signo_y = signo_x, signo_y
        self.alto, self.ancho = alto, ancho
        self.um_por_px_mundo = um_por_px_mundo
        self.frames_de_retraso = frames_de_retraso
        self.grano = grano
        self.ruido = ruido
        self._ruido = np.random.default_rng(semilla + 1)
        self._cuadros = []
        self._mundo = self._construir_mundo(lado_um, semilla)

    def _construir_mundo(self, lado_um, semilla):
        n = int(lado_um / self.um_por_px_mundo)
        generador = np.random.default_rng(semilla)
        mundo = np.empty((n, n, 3), np.uint8)
        mundo[:] = (240, 205, 215)                                    # plasma claro (RGB)
        radio = max(2, int(3.6 / self.um_por_px_mundo))               # glóbulo de ~3.6 µm de radio
        for _ in range(int(n * n / (radio * radio * 14))):
            fila, col = (int(v) for v in generador.integers(0, n, 2))
            color = tuple(int(np.clip(v + generador.normal(0, 10), 0, 255)) for v in (225, 140, 165))
            cv2.circle(mundo, (col, fila), int(radio * generador.uniform(0.85, 1.15)), color, -1, cv2.LINE_AA)
            cv2.circle(mundo, (col, fila), max(1, radio // 3), (235, 175, 195), -1, cv2.LINE_AA)
        for _ in range(40):                                           # puntos violetas
            fila, col = (int(v) for v in generador.integers(0, n, 2))
            cv2.circle(mundo, (col, fila), max(2, int(3.0 / self.um_por_px_mundo)), (110, 40, 140), -1, cv2.LINE_AA)
        mundo = cv2.GaussianBlur(mundo, (0, 0), 1.0)

        if self.grano > 0:      # textura fina fija (generador aparte: los glóbulos y los puntos quedan como sin grano)
            fino = cv2.GaussianBlur(np.random.default_rng(semilla + 2).normal(0, 7 * self.grano, (n, n)).astype(np.float32),
                                    (0, 0), 2.0)                       # el desvío baja a ~grano después del suavizado
            mundo = np.clip(mundo.astype(np.float32) + fino[:, :, None], 0, 255).astype(np.uint8)

        return mundo

    def _vista_rgb(self):
        n = self._mundo.shape[0]
        fila_mundo = n / 2 + (float(self.impresora.posicion_x) - self.origen_mm[0]) * 1000.0 / self.um_por_px_mundo
        col_mundo = n / 2 + (float(self.impresora.posicion_y) - self.origen_mm[1]) * 1000.0 / self.um_por_px_mundo
        ay = self.signo_x * self.um_por_px_mundo / self.um_por_px_fila      # filas de imagen por fila del mundo
        ax = self.signo_y * self.um_por_px_mundo / self.um_por_px_col
        centro_fila, centro_col = (self.alto - 1) / 2, (self.ancho - 1) / 2
        matriz = np.array([[ax, 0, centro_col - ax * col_mundo], [0, ay, centro_fila - ay * fila_mundo]], np.float64)
        vista = cv2.warpAffine(self._mundo, matriz, (self.ancho, self.alto), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        con_ruido = vista.astype(np.float32) + self._ruido.normal(0, self.ruido, vista.shape)
        return np.clip(con_ruido, 0, 255).astype(np.uint8)

    def capturar(self):
        self.capturas += 1
        self._cuadros.append(self._vista_rgb())
        del self._cuadros[:-(self.frames_de_retraso + 1)]
        cuadro = self._cuadros[0]                  # el de hace N capturas (o el más viejo que haya)

        if self.pixformat == "GRAYSCALE":
            return a_jpeg(cv2.cvtColor(cuadro, cv2.COLOR_RGB2GRAY), "L")

        return a_jpeg(cuadro, "RGB")

    def descartar_frames(self, cantidad=2):
        for _ in range(cantidad):
            self.capturar()

    def config_de_una(self, dict_params, calidad=100, skip=2):
        super().config_de_una(dict_params, calidad=calidad, skip=skip)
        self.descartar_frames(skip)


# %%
