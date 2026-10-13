# %%

"""
Control de la impresora Trimaker COSMOS (Repetier Firmware 0.92.3),
usada como plataforma de posicionamiento (x, y, z) para el microscopio.

Antes vivía junto con OpenMVCamera en "Main.py"; se separó en su propio
módulo porque las dos clases no tienen nada que ver entre sí.
"""

import re
import time

import numpy as np
import serial


class TrimakerCOSMOS:

    # =========================
    # INICIALIZACION
    # =========================
    def __init__(self,
                 puerto,
                 baudrate=250000,
                 clean_buffer=True,
                 devolveme_la_respuesta=False):

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

        self.resolucion_x = 0.0125  # 1/80
        self.resolucion_y = 0.0125
        self.resolucion_z = 0.0025  # 1/400

        self.ser = serial.Serial(
            port=puerto,
            baudrate=baudrate,
            timeout=0.1
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

                while self.ser.in_waiting:

                    linea = self.ser.readline().decode(errors="ignore").strip()

                    if linea:
                        respuestas.append(linea)

                for r in respuestas:
                    print(f"<<< {r}")

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
    # ACTUALIZACION DE LA POSICION DE LA CAMARA
    # =========================
    def actualizar_posicion_actual(self, x=None, y=None, z=None, tipo_de_cord=None, devolveme_la_respuesta=True):
        # Habria que mandarlo a home para que se creen los atributos de posicion_x, posicion_y, posicion_z

        if self.posicion_x is None or self.posicion_y is None or self.posicion_z is None:
            print("Para que se pueda guardar la posicion (x, y, z) de la camara primero hay que hacer 'self.home()'")

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
    def pregunta_posicion_actual(self, printear = False):
        if printear:
            
            print(f"La posicion actual de la camara es \n({self.posicion_x}, {self.posicion_y}, {self.posicion_z})")

        return self.posicion_x, self.posicion_y, self.posicion_z

    # =========================
    # FUNCION PARA HACER EL CAMBIO DE PASOS DEL MOTOR A MILIMETROS
    # =========================
    def pasos_a_mm(self, pasos_para_1_mm, cuantos_pasos_queres_hacer):
        # OJO: esta función le faltaba 'self' — tal como estaba definida,
        # llamarla como self.pasos_a_mm(...) (que es como se usa en
        # desp_lin_abs/desp_lin_rel con pasos_de_motor=True) tiraba
        # TypeError, porque Python pasa 'self' como primer argumento
        # posicional de más. No se nota en el uso normal porque
        # pasos_de_motor siempre quedó en False hasta ahora.

        return cuantos_pasos_queres_hacer / pasos_para_1_mm

    # =========================
    # AJUSTAR MOVIMIENTO
    # =========================
    def ajustar_movimiento(self, valor, movimiento_en="nada"):
        """
        Redondea 'valor' hacia abajo al múltiplo de la resolución del
        eje correspondiente (12.5 µm en x/y, 2.5 µm en z), que es la
        precisión real de los motores. El + 1e-6 es para que un valor
        que ya es un múltiplo exacto no se redondee hacia abajo por
        error de punto flotante (ej: 10.5 / 0.0025 podría dar
        4199.9999999 en vez de 4200.0).
        """

        if movimiento_en == "x" or movimiento_en == "X":
            return self.resolucion_x * np.floor(valor / self.resolucion_x + 1e-6)

        if movimiento_en == "y" or movimiento_en == "Y":
            return self.resolucion_y * np.floor(valor / self.resolucion_y + 1e-6)

        if movimiento_en == "z" or movimiento_en == "Z":
            return self.resolucion_z * np.floor(valor / self.resolucion_z + 1e-6)

    def es_linea_de_posicion(self, line):
        """True si la línea tiene el formato 'X:.. Y:.. Z:.. E:..' que manda M114."""
        return bool(re.search(r"X:-?\d+\.?\d*\s+Y:-?\d+\.?\d*\s+Z:-?\d+\.?\d*", line))

    def esperar_fin_movimiento(self, timeout=15):
        """
        Manda M400 + M114 (desde desp_lin_abs/desp_lin_rel/home, no acá
        adentro — ver el comentario en esos métodos) y bloquea hasta ver
        la línea de posición que M114 devuelve al terminar el movimiento.
        Devuelve el tiempo (en segundos) que tardó en aparecer esa línea,
        o None si se pasó del timeout.
        """

        t0 = time.time()

        while time.time() - t0 < timeout:
            if self.ser.in_waiting:
                line = self.ser.readline().decode(errors="ignore").strip()
                if self.es_linea_de_posicion(line):
                    return time.time() - t0

        return None

    # =========================
    # VALIDAR RANGO (usado por desp_lin_abs y desp_lin_rel)
    # =========================
    def _dentro_de_rango(self, x_absoluto, y_absoluto, z_absoluto):
        """
        Chequea que la posición ABSOLUTA de destino (x, y, z) esté
        dentro de los límites de la impresora, imprimiendo un aviso
        específico por cada eje que se pase. Devuelve True solo si
        los tres ejes están dentro de rango.
        """

        dentro_de_rango = True

        if not (0 <= x_absoluto <= self.max_posicion_x):
            print(
                f"Movimiento prohibido en x (se cancela la orden):\n"
                f"Para el modo absoluto el maximo desplazamiento es {self.max_posicion_x}\n"
                f"Para el modo relativo el maximo desplazamiento posible es {self.max_posicion_x - self.posicion_x}\n\n"
            )
            dentro_de_rango = False

        if not (0 <= y_absoluto <= self.max_posicion_y):
            print(
                f"Movimiento prohibido en y (se cancela la orden):\n"
                f"Para el modo absoluto el maximo desplazamiento es {self.max_posicion_y}\n"
                f"Para el modo relativo el maximo desplazamiento posible es {self.max_posicion_y - self.posicion_y}\n\n"
            )
            dentro_de_rango = False

        if not (0 <= z_absoluto <= self.max_posicion_z):
            print(
                f"Movimiento prohibido en z (se cancela la orden):\n"
                f"Para el modo absoluto el maximo desplazamiento es {self.max_posicion_z}\n"
                f"Para el modo relativo el maximo desplazamiento posible es {self.max_posicion_z - self.posicion_z}\n\n"
            )
            dentro_de_rango = False

        return dentro_de_rango

    # =========================
    # MOVER LINEALMENTE ABSOLUTO (SR fijo a la impresora)
    # =========================
    def desp_lin_abs(self, x=None, y=None, z=None, feedrate=3000, pasos_de_motor=False, devolveme_la_respuesta=False, devolveme_tiempos=False):
        # NOTA: x, y, z tienen que venir los tres, con un valor numérico
        # (no None) — a pesar del default, pasar alguno en None hace que
        # la comparación "x > self.max_posicion_x" de más abajo explote.

        if pasos_de_motor:
            x = self.pasos_a_mm(self.pasos_por_milimetro_eje_x, x)
            y = self.pasos_a_mm(self.pasos_por_milimetro_eje_y, y)
            z = self.pasos_a_mm(self.pasos_por_milimetro_eje_z, z)

        if not self._dentro_de_rango(x, y, z):
            return

        x = np.round(self.ajustar_movimiento(x, "x"), 4)
        # Se supone que a la impresora le mandas la distancia que se tiene que mover en milimetros, y tiene una precision de 12,5 micrones
        y = np.round(self.ajustar_movimiento(y, "y"), 4)
        z = np.round(self.ajustar_movimiento(z, "z"), 4)

        self.ser.write(b"G90\n")
        if devolveme_la_respuesta:
            print("\nSe hizo el cambio de cordenadas al sistema de referencia fijo a la impresora (Cordenadas absolutas):\n")

        comando_plano_xy = f"G1 X{x + self.offset_x} Y{y + self.offset_y} F{feedrate}\n"

        self.ser.write(comando_plano_xy.encode())
        self.ser.write(b"M400\n")
        self.ser.write(b"M114\n")

        tiempo_xy = self.esperar_fin_movimiento()

        comando_eje_z = f"G1 Z{z + self.offset_z} F{feedrate}\n"

        self.ser.write(comando_eje_z.encode())
        self.ser.write(b"M400\n")
        self.ser.write(b"M114\n")

        tiempo_z = self.esperar_fin_movimiento()

        self.actualizar_posicion_actual(x=x, y=y, z=z, tipo_de_cord="abs", devolveme_la_respuesta=devolveme_la_respuesta)

        if devolveme_tiempos:
            print(f"{tiempo_xy = }")
            print(f"{tiempo_z = }")
            return tiempo_xy, tiempo_z

    # =========================
    # MOVER LINEALMENTE RELATIVO (SR sobre la camara)
    # =========================
    def desp_lin_rel(self, x=None, y=None, z=None, feedrate=3000, pasos_de_motor=False, devolveme_la_respuesta=False, devolveme_tiempos=False):
        # Mismo comentario que en desp_lin_abs: x, y, z tienen que
        # venir los tres con valor numérico.

        if pasos_de_motor:
            x = self.pasos_a_mm(self.pasos_por_milimetro_eje_x, x)
            y = self.pasos_a_mm(self.pasos_por_milimetro_eje_y, y)
            z = self.pasos_a_mm(self.pasos_por_milimetro_eje_z, z)

        if not self._dentro_de_rango(self.posicion_x + x, self.posicion_y + y, self.posicion_z + z):
            return

        x = np.round(self.ajustar_movimiento(x, "x"), 4)
        # Se supone que a la impresora le mandas la distancia que se tiene que mover en milimetros, y tiene una presicion de 12,5 micrones
        y = np.round(self.ajustar_movimiento(y, "y"), 4)
        z = np.round(self.ajustar_movimiento(z, "z"), 4)

        self.ser.write(b"G91\n")
        if devolveme_la_respuesta:
            print("\nSe hizo el cambio de cordenadas al sistema de referencia fijo a la camara (Cordenadas realtivas):\n")

        comando_plano_xy = f"G1 X{x} Y{y} F{feedrate}\n"

        self.ser.write(comando_plano_xy.encode())
        self.ser.write(b"M400\n")
        self.ser.write(b"M114\n")

        tiempo_xy = self.esperar_fin_movimiento()

        comando_eje_z = f"G1 Z{z} F{feedrate}\n"

        self.ser.write(comando_eje_z.encode())
        self.ser.write(b"M400\n")
        self.ser.write(b"M114\n")

        tiempo_z = self.esperar_fin_movimiento()

        self.actualizar_posicion_actual(x=x, y=y, z=z, tipo_de_cord="rel", devolveme_la_respuesta=devolveme_la_respuesta)

        if devolveme_tiempos:
            print(f"{tiempo_xy = }")
            print(f"{tiempo_z = }")
            return tiempo_xy, tiempo_z

    # =========================
    # HOME
    # =========================
    def home(self, devolveme_la_respuesta=False, devolveme_timepos=False):
        # Setea la posicion de la camara

        self.posicion_x = 0
        self.posicion_y = 0
        self.posicion_z = 0

        self.escuchar(tiempo_maximo=5)

        self.ser.write(b"G28\n")
        self.ser.write(b"M400\n")

        tiempo_home = self.esperar_fin_movimiento()

        time.sleep(0.3)

        self.ser.write(b"G1 Z40 F3000\n")
        self.ser.write(b"M400\n")
        self.ser.write(b"M114\n")

        tiempo_bajar = self.esperar_fin_movimiento()

        self.actualizar_posicion_actual(x=0, y=0, z=40, tipo_de_cord="abs", devolveme_la_respuesta=devolveme_la_respuesta)

        if devolveme_timepos:
            print(f"{tiempo_home = }")
            print(f"{tiempo_bajar = }")
            return tiempo_home, tiempo_bajar

    # =========================
    # CERRAR PUERTO
    # =========================
    def desconectar(self):

        self.ser.close()

        print("Puerto cerrado")


# %%
