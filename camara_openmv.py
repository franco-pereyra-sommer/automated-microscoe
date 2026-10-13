# %%

"""
Control de la cámara OpenMV H7 PLUS sobre el protocolo serie de texto
que entiende "main_openmv.py" (el firmware corrido en la placa).

Antes vivía junto con TrimakerCOSMOS en "Main.py"; se separó en su
propio módulo porque las dos clases no tienen nada que ver entre sí.

También define PixFormat y Framesize: antes la configuración de la
cámara se pasaba como strings sueltos ("JPEG", "WQXGA2", ...) sin
ninguna validación del lado de la PC (la validación estaba comentada
en set_pixformat()), así que un typo como "JPG" en vez de "JPEG" viajaba
hasta la OpenMV y volvía como un "ERR" sin más detalle. Con estos dos
enums + _resolver_opcion(), un typo tira un ValueError inmediato y
legible, ACÁ, antes de mandar nada por el puerto serie.
"""

import enum
import io
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
from PIL import Image
import matplotlib.pyplot as plt


# ======================================================
# CONFIGURACIÓN DE CÁMARA, SIN STRINGS SUELTOS
# ======================================================

class PixFormat(enum.Enum):
    JPEG = "JPEG"
    RGB565 = "RGB565"
    GRAYSCALE = "GRAYSCALE"


class Framesize(enum.Enum):
    QQVGA = "QQVGA"
    QVGA = "QVGA"
    VGA = "VGA"
    HD = "HD"
    FHD = "FHD"
    WQXGA2 = "WQXGA2"
    K4 = "WQXGA2"  # alias de WQXGA2: "4K" no es un identificador
    # válido de Python, así que como NOMBRE de miembro usamos "K4";
    # el string literal "4K" se resuelve en _resolver_opcion() más
    # abajo, que es el que de verdad hace fácil escribirlo.


def _resolver_opcion(valor, enum_clase, nombre_parametro):
    """
    Acepta tanto un miembro del enum (por ej. PixFormat.RGB565) como
    un string (para no romper el código existente, que sigue pasando
    strings como "RGB565" o "GRAYSCALE"), incluido el caso especial
    "4K" -> WQXGA2.

    Deliberadamente NO se usa 'class PixFormat(str, enum.Enum)': ese
    mixin tiene comportamiento de __str__/__format__ que cambió entre
    versiones de Python (en algunas, f"{PixFormat.JPEG}" imprime
    "JPEG" y en otras "PixFormat.JPEG"), así que un Enum simple +
    este resolver es más predecible.

    Tira ValueError con un mensaje claro -- en vez del "ERR" mudo que
    devolvía la placa -- si el string no coincide con ninguna opción
    válida.
    """

    if isinstance(valor, enum_clase):
        return valor.value

    if isinstance(valor, str):

        valor_mayus = valor.strip().upper()

        if valor_mayus == "4K":
            valor_mayus = "WQXGA2"

        try:
            return enum_clase[valor_mayus].value
        except KeyError:
            opciones = sorted({m.value for m in enum_clase})
            raise ValueError(
                f"{nombre_parametro} inválido: {valor!r}. "
                f"Usá uno de {opciones}, o el miembro de "
                f"{enum_clase.__name__} correspondiente."
            )

    raise TypeError(
        f"{nombre_parametro} tiene que ser un string o un miembro de "
        f"{enum_clase.__name__}, no {type(valor).__name__}"
    )


class OpenMVCamera:

    def __init__(
        self,
        puerto="COM7",
        baudrate=115200,
        timeout=10
    ):
        self.puerto = puerto
        self.baudrate = baudrate
        self.timeout = timeout

        self.ser = None
        self.pixformat = None

    # ======================================================
    # CONEXIÓN
    # ======================================================

    def conectar(self):

        self.ser = serial.Serial(
            self.puerto,
            self.baudrate,
            timeout=self.timeout
        )

        self.pixformat = None

        time.sleep(2)

        self.ser.reset_input_buffer()

    def desconectar(self):

        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def _check_connection(self):

        if self.ser is None or not self.ser.is_open:
            raise Exception("La cámara no está conectada")

    # ======================================================
    # COMUNICACIÓN
    # ======================================================

    def _enviar_comando(self, comando):

        self._check_connection()

        self.ser.reset_input_buffer()

        self.ser.write(
            (comando + "\n").encode()
        )

        respuesta = self.ser.readline().decode(
            errors="ignore"
        ).strip()

        if respuesta == "ERR":
            raise Exception(
                f"OpenMV rechazó el comando: {comando}"
            )

        if respuesta != "OK":
            raise Exception(
                f"Respuesta inesperada de OpenMV: "
                f"{respuesta!r}"
            )

        return True

    # ======================================================
    # CAPTURA
    # ======================================================

    def capturar(self):

        self._check_connection()

        self.ser.reset_input_buffer()

        self.ser.write(b"capture\n")

        while True:

            line = self.ser.readline().decode(
                errors="ignore"
            ).strip()

            if line == "ERR":
                raise Exception(
                    "Error durante la captura en OpenMV"
                )

            if line.isdigit():
                size = int(line)
                break

        self.ser.write(b"ok\n")

        image_data = bytearray()

        while len(image_data) < size:

            chunk = self.ser.read(
                size - len(image_data)
            )

            if not chunk:
                raise Exception(
                    "Timeout recibiendo imagen"
                )

            image_data.extend(chunk)

        return bytes(image_data)

    # ======================================================
    # FORMATO
    # ======================================================

    def set_pixformat(self, formato):

        formato_resuelto = _resolver_opcion(formato, PixFormat, "formato")

        self._enviar_comando(
            f"set_pixformat {formato_resuelto}"
        )

        self.pixformat = formato_resuelto

    # ======================================================
    # DESCARTAR FRAMES
    # ======================================================
    # De uso libre: llamala después de cambiar pixformat,
    # framesize, exposición, ganancia o balance de blancos (o
    # cualquier combinación de esos cambios), para darle tiempo
    # al sensor de reacomodarse antes de la próxima captura que
    # realmente te importa. Cada frame descartado es una captura
    # real (no un time.sleep a ciegas), así que también sirve como
    # espera bloqueante genuina.

    def descartar_frames(self, cantidad=2):

        for _ in range(cantidad):
            self.capturar()

    # ======================================================
    # RESOLUCIÓN
    # ======================================================

    def set_framesize(self, resolucion):

        resolucion_resuelta = _resolver_opcion(resolucion, Framesize, "resolucion")

        self._enviar_comando(
            f"set_framesize {resolucion_resuelta}"
        )

    # ======================================================
    # CALIDAD JPEG
    # ======================================================

    def set_quality(self, quality):

        if not 0 <= quality <= 100:
            raise ValueError(
                "La calidad JPEG debe estar entre 0 y 100"
            )

        self._enviar_comando(
            f"set_quality {quality}"
        )

    # ======================================================
    # EXPOSICIÓN
    # ======================================================

    def set_exposure_auto(self, enable=True):

        self._enviar_comando(
            f"set_exposure_auto {int(enable)}"
        )

    def set_exposure(self, exposure_us):

        self._enviar_comando(
            f"set_exposure_auto 0 {exposure_us}"
        )

    # ======================================================
    # GANANCIA
    # ======================================================

    def set_gain_auto(self, enable=True):

        self._enviar_comando(
            f"set_gain_auto {int(enable)}"
        )

    def set_gain(self, gain_db):

        self._enviar_comando(
            f"set_gain_auto 0 {gain_db}"
        )

    # ======================================================
    # BALANCE DE BLANCOS
    # ======================================================

    def set_whitebal_auto(self, enable=True):

        self._enviar_comando(
            f"set_whitebal_auto {int(enable)}"
        )

    def set_whitebal(
        self,
        red,
        green,
        blue
    ):

        self._enviar_comando(
            f"set_whitebal_auto 0 "
            f"{red} {green} {blue}"
        )

    # ======================================================
    # INFORMACIÓN
    # ======================================================

    def info(self):

        self._check_connection()

        self.ser.reset_input_buffer()

        self.ser.write(b"info\n")

        line = self.ser.readline().decode(
            errors="ignore"
        ).strip()

        if not line:
            raise Exception(
                "No se recibió información de la cámara"
            )

        values = line.split()

        if len(values) != 3:
            raise Exception(
                f"Respuesta inesperada: {line!r}"
            )

        exposure_us = int(values[0])
        gain_db = float(values[1])

        rgb_gain_db = (
            float(values[2].strip("()").split(",")[0]),
            float(values[2].strip("()").split(",")[1]),
            float(values[2].strip("()").split(",")[2])
        )

        return {
            "exposure_us": exposure_us,
            "gain_db": gain_db,
            "rgb_gain_db": rgb_gain_db
        }

    # ======================================================
    # MOSTRAR
    # ======================================================

    def mostrar(self, image_data):

        img = Image.open(
            io.BytesIO(image_data)
        )

        imagen = np.array(img)

        fig, ax = plt.subplots()

        if imagen.ndim == 2:
            ax.imshow(imagen, cmap="gray")
        else:
            ax.imshow(imagen)

        ax.axis("off")

        plt.show()

    # ======================================================
    # GUARDAR
    # ======================================================

    def guardar_imagen(
        self,
        image_data,
        carpeta,
        mostrar=False,
        nombre=None
    ):

        carpeta = Path(carpeta)

        carpeta.mkdir(
            parents=True,
            exist_ok=True
        )

        if nombre is None:
            nombre = datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S"
            )

        archivo = carpeta / f"{nombre}.jpg"

        with open(archivo, "wb") as f:
            f.write(image_data)

        print(f"Imagen guardada en:\n{archivo}")

        if mostrar:
            self.mostrar(image_data)

        return archivo

    def guardar_bytes(
        self,
        image_data,
        carpeta,
        nombre=None
    ):

        return self.guardar_imagen(
            image_data,
            carpeta,
            mostrar=False,
            nombre=nombre
        )

    # ======================================================
    # CONFIGURACIONES "DE UNA" PARA LOS CASOS DE USO HABITUALES
    # ======================================================

    def config_de_una(
        self,
        pixformat="RGB565",
        resolucion="WQXGA2",
        exposure_us=None,
        gain_db=None,
        calidad=90
    ):

        self.set_pixformat(pixformat)
        self.set_framesize(resolucion)
        self.set_quality(calidad)

        if exposure_us is None:
            self.set_exposure_auto(True)
        else:
            self.set_exposure(exposure_us)

        if gain_db is None:
            self.set_gain_auto(True)
        else:
            self.set_gain(gain_db)

        self.descartar_frames(cantidad=1)

    def config_autofoco_local(
        self,
        pixformat=PixFormat.GRAYSCALE,
        exposure_us=None,
        gain_db=None,
    ):

        self.set_pixformat(pixformat)
        self.set_framesize(Framesize.QVGA)
        self.set_quality(100)

        if exposure_us is None:
            self.set_exposure_auto(True)
        else:
            self.set_exposure(exposure_us)

        if gain_db is None:
            self.set_gain_auto(True)
        else:
            self.set_gain(gain_db)

        self.descartar_frames(cantidad=1)


# %%
