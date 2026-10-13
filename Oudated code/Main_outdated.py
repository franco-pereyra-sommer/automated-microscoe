# %%


import numpy as np
import serial
import time
import cv2


class TrimakerCOSMOS:

    # =========================
    # INICIALIZACION
    # =========================
    def __init__(self,
                 puerto,
                 baudrate=250000,
                 clean_buffer = True,
                 devolveme_la_respuesta = False):
        
        # Configuracion particular de la impresora que usamos
        self.offset_x = -19
        self.offset_y = -3
        self.offset_z = 0

        self.posicion_x = None
        self.posicion_y = None
        self.posicion_z = None

        self.max_posicion_x = 270
        self.max_posicion_y = 198
        self.max_posicion_z = 150

        self.pasos_por_milimetro_eje_x = 80
        self.pasos_por_milimetro_eje_y = 80
        self.pasos_por_milimetro_eje_z = 400

        self.resolucion_x = 0.0125 # 1/80
        self.resolucion_y = 0.0125 
        self.resolucion_z = 0.0025 # 1/400

        self.ser = serial.Serial(
            port=puerto,
            baudrate=baudrate,
            timeout=1
        )

        # Evitar reset automatico
        self.ser.dtr = False
        self.ser.rts = False

        # Esperar boot
        time.sleep(5)

        print("Conectado a la impresora")

        if clean_buffer:

            self.ser.reset_input_buffer()

            if devolveme_la_respuesta:
                time.sleep(2)
                
                respuestas = []
        
                if devolveme_la_respuesta:
                    #Por si no queres los infinitos mensajes que te manda
        
                    while self.ser.in_waiting:
        
                        linea = self.ser.readline().decode(errors="ignore").strip()
        
                        if linea:
                            respuestas.append(linea)
        
                    for r in respuestas:
                        print(f"<<< {r}")




    # =========================
    # ESPERAR RESPUESTA
    # =========================
    def esperar_ok(self, tiempo_maximo=10):

        t0 = time.perf_counter()

        while time.perf_counter() - t0 < tiempo_maximo:

            if self.ser.in_waiting:

                linea = self.ser.readline().decode(errors="ignore").strip()

                if linea:

                    tiempo = time.perf_counter() - t0
                    print(f"{tiempo:.6f} s <<< {linea}")

                    if linea == "ok":
                        return tiempo

        return None

    

    def escuchar(self, tiempo_maximo=10):

        respuestas = []

        t0 = time.perf_counter()

        while time.perf_counter() - t0 < tiempo_maximo:

            if self.ser.in_waiting:

                linea = self.ser.readline().decode(errors="ignore").strip()

                if linea:

                    tiempo = time.perf_counter() - t0

                    respuestas.append((tiempo, linea))

                    print(f"{tiempo:.6f} s  <<< {linea}")

        return respuestas

    
    # =========================
    # ENVIAR GCODE
    # =========================
    def enviar_gcode(self, comando, espera=0.5, devolveme_la_respuesta = True):

        comando = comando.strip() + "\n"

        if devolveme_la_respuesta:

            print(f"\n>>> {comando.strip()}")

        self.ser.write(comando.encode())

        time.sleep(espera)

        respuestas = []

        if devolveme_la_respuesta:
            #Por si no queres los infinitos mensajes que te manda

            while self.ser.in_waiting:

                linea = self.ser.readline().decode(errors="ignore").strip()

                if linea:
                    respuestas.append(linea)

            for r in respuestas:
                print(f"<<< {r}")

            print(f"{comando = }")

        return respuestas



    # =========================
    # CAMBIAR A COORDENADAS RELATIVAS (SR fijo a la camara)
    # =========================
    def coord_rel(self, devolveme_la_respuesta=False):

        if devolveme_la_respuesta:
            print("\nSe hizo el cambio de cordenadas al sistema de referencia fijo a la camara (Cordenadas realtivas):\n")

        self.enviar_gcode("G91", devolveme_la_respuesta=devolveme_la_respuesta)



    # =========================
    # CAMBIAR A COORDENADAS ABSOLutaS (SR sobre la impresora)
    # =========================
    def coord_abs(self, devolveme_la_respuesta=False):

        if devolveme_la_respuesta:
            print("\nSe hizo el cambio de cordenadas al sistema de referencia fijo a la camara (Cordenadas absolutas):\n")

        self.enviar_gcode("G90", devolveme_la_respuesta=devolveme_la_respuesta)


    
    # =========================
    # ACATUALIZACION DE LA POSICION DE LA CAMARA
    # =========================
    def actualizar_posicion_actual(self, x=None, y=None, z=None, tipo_de_cord = None, devolveme_la_respuesta=True):
        # Habria que mandarlo a home para que se creen los atributos de posicion_x, posicion_y, posicion_z

        if self.posicion_x == None or self.posicion_y == None or self.posicion_z == None:
            print("Para que se pueda guardar la posicion (x, y, z) de la camara primero hay que hacer 'self.home()'")
            # # self.home() # Pero todavia no la defini a la funcion home(self)
            # self.enviar_gcode("G28")

            # # Setea la posicion de la camara 

            # self.posicion_x = 0
            # self.posicion_y = 0
            # self.posicion_z = 0

        if tipo_de_cord == "abs":

            if devolveme_la_respuesta:
                print(f"La camara empezo el movimiento en la posicion ({self.posicion_x}, {self.posicion_y}, {self.posicion_z})")

            self.posicion_x = np.round(x, 4)
            self.posicion_y = np.round(y, 4)
            self.posicion_z = np.round(z, 4)

            if devolveme_la_respuesta:
                print(f"\nLa camara termino el movimiento en la posicion ({self.posicion_x}, {self.posicion_y}, {self.posicion_z})")

        elif tipo_de_cord == "rel":
            if devolveme_la_respuesta:
                print(f"La camara empezo el movimiento en la posicion ({self.posicion_x} , {self.posicion_y}, {self.posicion_z})")

                print(f"La camara se va a mover ({x}, {y}, {z})")

            self.posicion_x = np.round(self.posicion_x + x, 4)
            self.posicion_y = np.round(self.posicion_y + y, 4)
            self.posicion_z = np.round(self.posicion_z + z, 4)  

            if devolveme_la_respuesta:

                print(f"\nLa camara termino el movimiento en la posicion ({self.posicion_x}, {self.posicion_y}, {self.posicion_z})")

        else:
            print("Tenes que poner el tipo de cordenadas 'rel' (de relativo, SR sobre la camara) o 'abs' (de absoluto, SR fijo a la impresora)")

    
    # =========================
    # PREGUNTAR LA POSICION DE LA CAMARA
    # =========================
    def pregunta_posicion_actual(self):

        print(f"La posicion actual de la camara es \n({self.posicion_x}, {self.posicion_y}, {self.posicion_z})")

        return self.posicion_x, self.posicion_y, self.posicion_z


    # =========================
    # FUNCION PARA HACER EL CAMBIO DE PASOS DEL MOTOR A MILIMETROS
    # =========================
    def pasos_a_mm(
        pasos_para_1_mm,
        cuantos_pasos_queres_hacer
        ):
        return cuantos_pasos_queres_hacer/pasos_para_1_mm


    # =========================
    # AJUSTAR MOVIMIENTO
    # =========================
    def ajustar_movimiento(self, valor, movimiento_en = "nada"):

        if movimiento_en == "x" or movimiento_en == "X":
            return self.resolucion_x * np.floor(valor / self.resolucion_x)

        if movimiento_en == "y" or movimiento_en == "Y":
            return self.resolucion_y * np.floor(valor / self.resolucion_y)

        if movimiento_en == "z" or movimiento_en == "Z":
            return self.resolucion_z * np.floor(valor / self.resolucion_z)



    # =========================
    # MOVER LINEALMENTE ABSOLUTO (SR fijo a la impresora)
    # =========================
    def desp_lin_abs(self, x=None, y=None, z=None, feedrate=3000, pasos_de_motor=False, devolveme_la_respuesta=False):

        if pasos_de_motor:
            x = self.pasos_a_mm(self.pasos_por_milimetro_eje_x, x)

            y = self.pasos_a_mm(self.pasos_por_milimetro_eje_y, y)

            z = self.pasos_a_mm(self.pasos_por_milimetro_eje_z, z)

        if x > self.max_posicion_x:
            
            print(f"Movimiento prohibido en x (se cancela la orden):\nPara el modo absoluto el maximo desplazamiento es {self.max_posicion_x}\nPara el modo relativo el maximo desplazamiento posible es {self.max_posicion_x-self.posicion_x}\n\n")
        
        if y > self.max_posicion_y:

            print(f"Movimiento prohibido en y (se cancela la orden):\nPara el modo absoluto el maximo desplazamiento es {self.max_posicion_y}\nPara el modo relativo el maximo desplazamiento posible es {self.max_posicion_y-self.posicion_y}\n\n")

        if z > self.max_posicion_z:

            print(f"Movimiento prohibido en z (se cancela la orden):\nPara el modo absoluto el maximo desplazamiento es {self.max_posicion_z}\nPara el modo relativo el maximo desplazamiento posible es {self.max_posicion_z-self.posicion_z}\n\n")

        if x <= self.max_posicion_x and x >= 0 and y <= self.max_posicion_y and y >= 0 and z <= self.max_posicion_z and z >= 0:

            x = self.ajustar_movimiento(x, "x")
            x = np.round(x, 4)
            # Se supone que a la impresora le mandas la distancia que se tiene que mover en milimetros, y tiene una precision de 12,5 micrones

            y = self.ajustar_movimiento(y, "y")
            y = np.round(y, 4)

            z = self.ajustar_movimiento(z, "z")
            z = np.round(z, 4)

            self.coord_abs(devolveme_la_respuesta=devolveme_la_respuesta)

            comando_eje_z = "G1"

            if z is not None:
                
                comando_eje_z += f" Z{z + self.offset_z}"

            comando_eje_z += f" F{feedrate}"

            self.enviar_gcode(comando_eje_z,devolveme_la_respuesta=devolveme_la_respuesta)

            comando_plano_xy = "G1"

            if x is not None:

                comando_plano_xy += f" X{x + self.offset_x}"

            if y is not None:

                comando_plano_xy += f" Y{y + self.offset_y}"

            comando_plano_xy += f" F{feedrate}"

            self.enviar_gcode(comando_plano_xy, devolveme_la_respuesta=devolveme_la_respuesta)

            self.actualizar_posicion_actual(x=x, y=y, z=z, tipo_de_cord="abs", devolveme_la_respuesta=devolveme_la_respuesta)

    

    # =========================
    # MOVER LINEALMENTE RELATIVO (SR sobre la camara)
    # =========================
    def desp_lin_rel(self, x=None, y=None, z=None, feedrate=3000, pasos_de_motor=False, devolveme_la_respuesta=False):

        if pasos_de_motor:
            x = self.pasos_a_mm(self.pasos_por_milimetro_eje_x, x)

            y = self.pasos_a_mm(self.pasos_por_milimetro_eje_y, y)

            z = self.pasos_a_mm(self.pasos_por_milimetro_eje_z, z)
        
        if self.posicion_x + x > self.max_posicion_x:

            print(f"Movimiento prohibido en x (se cancela la orden):\nPara el modo absoluto el maximo desplazamiento es {self.max_posicion_x}\nPara el modo relativo el maximo desplazamiento posible es {self.max_posicion_x-self.posicion_x}\n\n")
        
        if self.posicion_y + y > self.max_posicion_y:

            print(f"Movimiento prohibido en y (se cancela la orden):\nPara el modo absoluto el maximo desplazamiento es {self.max_posicion_y}\nPara el modo relativo el maximo desplazamiento posible es {self.max_posicion_y-self.posicion_y}\n\n")

        if self.posicion_z + z > self.max_posicion_z:

            print(f"Movimiento prohibido en z (se cancela la orden):\nPara el modo absoluto el maximo desplazamiento es {self.max_posicion_z}\nPara el modo relativo el maximo desplazamiento posible es {self.max_posicion_z-self.posicion_z}\n\n")

        if self.posicion_x + x <= self.max_posicion_x and self.posicion_x + x >= 0 and self.posicion_y + y <= self.max_posicion_y and self.posicion_y + y >= 0 and self.posicion_z + z <= self.max_posicion_z and self.posicion_z + z >= 0:

            x = self.ajustar_movimiento(x, "x")
            x = np.round(x, 4)
            # Se supone que a la impresora le mandas la distancia que se tiene que mover en milimetros, y tiene una presicion de 12,5 micrones

            y = self.ajustar_movimiento(y, "y")
            y = np.round(y, 4)

            z = self.ajustar_movimiento(z, "z")
            z = np.round(z, 4)

            self.coord_rel(devolveme_la_respuesta=devolveme_la_respuesta)

            comando_eje_z = f"G1 Z{z} F{feedrate}"

            self.enviar_gcode(comando_eje_z, devolveme_la_respuesta=devolveme_la_respuesta)

            comando_plano_xy = f"G1 X{x} Y{y} F{feedrate}"

            self.enviar_gcode(comando_plano_xy, devolveme_la_respuesta=devolveme_la_respuesta)

            self.actualizar_posicion_actual(x = x, y = y, z = z , tipo_de_cord="rel", devolveme_la_respuesta=devolveme_la_respuesta)


    
    # =========================
    # HOME
    # =========================
    def home(self, devolveme_la_respuesta=False):

        self.enviar_gcode("G28", devolveme_la_respuesta=devolveme_la_respuesta)

        # Setea la posicion de la camara 

        self.posicion_x = 0
        self.posicion_y = 0
        self.posicion_z = 0

        # Para que baje 20mm en Z

        # self.desp_lin_rel(0 + self.offset_x, 0 + self.offset_y, 20 + self.offset_z) 

        self.desp_lin_rel(0, 0, 40, devolveme_la_respuesta=devolveme_la_respuesta)


    
    # =========================
    # BAJA TODO LO POSIBLE LA CAMA EN Z
    # =========================
    def lower_z(self):

        self.desp_lin_rel(0, 0, self.max_posicion_z) 

        self.actualizar_posicion_actual(x=0, y=0, z=self.max_posicion_z, tipo_de_cord="abs")
        


    # =========================
    # CERRAR PUERTO
    # =========================
    def desconectar(self):

        self.ser.close()

        print("Puerto cerrado")




import serial
import time
import io

from pathlib import Path
from datetime import datetime

from PIL import Image
import matplotlib.pyplot as plt


class OpenMVCamera_Viejo:

    def __init__(self,
                 puerto="COM7",
                 baudrate=115200,
                 timeout=10):

        self.puerto = puerto
        self.baudrate = baudrate
        self.timeout = timeout

        self.ser = None

    # ==========================================
    # CONECTAR
    # ==========================================

    def conectar(self):

        self.ser = serial.Serial(
            self.puerto,
            self.baudrate,
            timeout=self.timeout
        )

        # Esperar boot OpenMV
        time.sleep(2)

        self.ser.reset_input_buffer()

    # ==========================================
    # DESCONECTAR
    # ==========================================

    def desconectar(self):

        if self.ser is not None:
            self.ser.close()

    # ==========================================
    # CAPTURAR IMAGEN
    # ==========================================

    def capturar_viejo(self):

        if self.ser is None:
            raise Exception("La cámara no está conectada")

        # Pedir captura
        self.ser.write(b"capture\n")

        # Leer tamaño
        line = self.ser.readline().decode(
            errors="ignore"
        ).strip()

        print("RX:", repr(line))

        if line == "ERR":
            raise Exception("Error en OpenMV")

        size = int(line)

        # ACK
        self.ser.write(b"ok\n")

        # Recibir JPG
        image_data = b""

        while len(image_data) < size:

            chunk = self.ser.read(size - len(image_data))

            if not chunk:
                break

            image_data += chunk

        return image_data
    
    def capturar(self):

        if self.ser is None:
            raise Exception("La cámara no está conectada")

        # limpiar basura previa
        self.ser.reset_input_buffer()

        # pedir foto
        self.ser.write(b"capture\n")

        # buscar un número válido
        while True:

            line = self.ser.readline().decode(
                errors="ignore"
            ).strip()

            print("RX:", repr(line))

            if line == "ERR":
                raise Exception("Error en OpenMV")

            if line.isdigit():
                size = int(line)
                break


        # ACK
        self.ser.write(b"ok\n")


        # recibir imagen
        image_data = b""

        while len(image_data) < size:

            chunk = self.ser.read(
                size - len(image_data)
            )

            if not chunk:
                raise Exception(
                    "Timeout recibiendo imagen"
                )

            image_data += chunk


        return image_data

    # ==========================================
    # MOSTRAR IMAGEN
    # ==========================================

    def mostrar(self, image_data):

        img = Image.open(io.BytesIO(image_data))

        plt.imshow(img)
        plt.axis("off")
        plt.show()

    # ==========================================
    # GUARDAR IMAGEN
    # ==========================================

    def guardar_imagen(self,
                       image_data,
                       carpeta,
                       mostrar=False,
                       nombre=None):
        
        carpeta = Path(carpeta)

        carpeta.mkdir(
            parents=True,
            exist_ok=True
        )

        # Nombre automático
        if nombre is None:

            nombre = datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S"
            )

        archivo = carpeta / f"{nombre}.jpg"

        # Guardar JPG
        with open(archivo, "wb") as f:
            f.write(image_data)

        print(f"Imagen guardada en:\n{archivo}")

        # Mostrar opcional
        if mostrar:
            self.mostrar(image_data)

        return archivo
    
    # ==========================================
    # GUARDAR IMAGEN (BYTES)
    # ==========================================

    
    def guardar_bytes(self,
                  image_data,
                  carpeta,
                  nombre=None):

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

        print(f"Guardada:\n{archivo}")

        return archivo



import serial
import time
import io

from pathlib import Path
from datetime import datetime

from PIL import Image
import matplotlib.pyplot as plt


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

    # ======================================================
    # CONEXIÓN
    # ======================================================

    def conectar(self):

        self.ser = serial.Serial(
            self.puerto,
            self.baudrate,
            timeout=self.timeout
        )

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

            # print("RX:", repr(line)) # Linea que devuelve muchas cosas cuando sacas muchas fotos

            if line == "ERR":
                raise Exception(
                    "Error durante la captura en OpenMV"
                )

            if line.isdigit():
                size = int(line)
                break

        # print("Tamaño recibido:", size)

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

        # print("Bytes recibidos:", len(image_data)) # Linea que devuelve muchas cosas cuando sacas muchas fotos
        # print("Primeros bytes:", image_data[:20]) # Linea que devuelve muchas cosas cuando sacas muchas fotos

        return bytes(image_data)

    def capturar_antes_del_cambio(self):

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

        formatos = {
            "JPEG",
            "RGB565",
            "GRAYSCALE"
        }

        formato = formato.upper()

        if formato not in formatos:
            raise ValueError(
                f"Formato inválido: {formato}. "
                f"Use: {formatos}"
            )

        self._enviar_comando(
            f"set_pixformat {formato}"
        )

    def jpeg(self):

        self.set_pixformat("JPEG")

    def rgb565(self):

        self.set_pixformat("RGB565")

    def grayscale(self):

        self.set_pixformat("GRAYSCALE")

    # ======================================================
    # RESOLUCIÓN
    # ======================================================

    def set_framesize(self, resolucion):

        resoluciones = {
            "QQVGA",
            "QVGA",
            "VGA",
            "HD",
            "FHD",
            "WQXGA2"
        }

        resolucion = resolucion.upper()

        if resolucion not in resoluciones:
            raise ValueError(
                f"Resolución inválida: {resolucion}. "
                f"Use: {resoluciones}"
            )

        self._enviar_comando(
            f"set_framesize {resolucion}"
        )

    # Alias más intuitivo

    def set_resolution(self, resolucion):

        self.set_framesize(resolucion)

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

        fig, ax = plt.subplots()

        ax.imshow(img)
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

    def config_analisis(
        self,
        resolucion="WQXGA2",
        exposure_us=None,
        gain_db=None,
        calidad=90
    ):

        self.set_pixformat("GRAYSCALE")
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


    def config_video(
        self,
        resolucion="HD",
        calidad=80,
        exposure_us=None,
        gain_db=None
    ):

        self.set_pixformat("JPEG")
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


# ==========================================
# USO
# ==========================================


# cam = OpenMVCamera("COM7")

# cam.conectar()

# # Guardar imagen sin mostrar

# cam.guardar_imagen(
#     carpeta=r"C:\Fotos",
#     mostrar=False
# )


# # Guardar y mostrar

# cam.guardar_imagen(
#     carpeta=r"C:\Fotos",
#     # nombre="parasito_001", # Con nombre personalizado
#     mostrar=True
# )


# # Al final

# cam.desconectar()


# # Capturar muchas imagenes

# imagenes = []

# for i in range(10):

#     imagenes.append(
#         cam.capturar()
#     )

#     time.sleep(0.1)


# # Guardarlas en formato Byte despues

# for i, img in enumerate(imagenes):

#     cam.guardar_bytes(
#         img,
#         carpeta=r"C:\Fotos",
#         nombre=f"foto_{i}"
#     )


# # Procesarlas con OpenCV

# import cv2
# import numpy as np

# data = np.frombuffer(
#     imagenes[0],
#     dtype=np.uint8
# )

# frame = cv2.imdecode(
#     data,
#     cv2.IMREAD_COLOR
# )



# %%
