#%%



# Agrego la raiz del proyecto
import sys
import os

os.environ["PYTHONPATH"] = r"C:\BackUp\Facultad\Labo 7\env_labo7"

sys.path.insert(0, os.environ["PYTHONPATH"])


from pathlib import Path
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt

import time
from datetime import datetime

from Main import OpenMVCamera
from Main import TrimakerCOSMOS



# ============================================================
# MÉTRICAS
# ============================================================

def laplacian_variance(imagen):

    gray = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    return laplacian.var()


def tenengrad(imagen):

    gray = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    magnitud = sobel_x**2 + sobel_y**2

    return magnitud.mean()


def varianza_intensidad(imagen):

    gray = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    return gray.var()



def autofocus(
        camara,
        impresora,
        metrica,
        pasos_arriba=10,
        pasos_abajo=10,
        paso_z=0.0025,
        espera_movimiento=1,
        graficamos=True,
        guardamos=False,
        carpeta_guardado=None
        ):

    # ========================================================
    # POSICION INICIAL
    # ========================================================

    _, _, z_inicial = impresora.pregunta_posicion_actual()

    print(f"\nPosición Z inicial: {z_inicial} mm")


    # ========================================================
    # RESULTADOS
    # ========================================================

    posiciones_z = []
    valores_metrica = []


    # ========================================================
    # VOLVER A LA POSICION INICIAL
    # ========================================================

    if pasos_abajo > 0:

        impresora.desp_lin_rel(
            0,
            0,
            -pasos_abajo * paso_z
        )

        time.sleep(espera_movimiento)


    # ========================================================
    # BARRIDO EN Z
    # ========================================================

    cantidad_fotos = pasos_arriba + pasos_abajo + 1

    for paso in range(cantidad_fotos):

        # --------------------------------------------
        # Tomar fotografía
        # --------------------------------------------

        image_data = camara.capturar()

        _, _, z_actual = impresora.pregunta_posicion_actual()

        if guardamos:

            hora_actual = datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S"
            )

            camara.guardar_imagen(
                image_data=image_data,
                carpeta=carpeta_guardado,
                nombre=f"{hora_actual} ({impresora.posicion_x}, {impresora.posicion_y}, {z_actual})",
                mostrar=False
            )


        # --------------------------------------------
        # Convertir JPG -> imagen OpenCV
        # --------------------------------------------

        data = np.frombuffer(
            image_data,
            dtype=np.uint8
        )

        imagen = cv2.imdecode(
            data,
            cv2.IMREAD_COLOR
        )


        # --------------------------------------------
        # Calcular métrica
        # --------------------------------------------

        valor = metrica(imagen)


        # --------------------------------------------
        # Guardar posición y métrica
        # --------------------------------------------

        _, _, z_actual = impresora.pregunta_posicion_actual()

        posiciones_z.append(z_actual)
        valores_metrica.append(valor)


        print(
            f"Foto {paso + 1}/{cantidad_fotos} | "
            f"Z = {z_actual:.6f} mm | "
            f"Métrica = {valor:.6f}"
        )


        # --------------------------------------------
        # Mover al siguiente Z
        # --------------------------------------------

        if paso < cantidad_fotos - 1:

            impresora.desp_lin_rel(
                0,
                0,
                paso_z
            )

            time.sleep(espera_movimiento)


    # ========================================================
    # ENCONTRAR MEJOR POSICION
    # ========================================================

    indice_mejor = np.argmax(valores_metrica)

    mejor_z = posiciones_z[indice_mejor]
    mejor_valor = valores_metrica[indice_mejor]


    print("\n" + "=" * 50)
    print("RESULTADO DEL AUTOFOCUS")
    print("=" * 50)

    print(f"Mejor Z: {mejor_z:.6f} mm")
    print(f"Métrica: {mejor_valor:.6f}")


    # ========================================================
    # MOVER A LA MEJOR POSICION
    # ========================================================

    _, _, z_actual = impresora.pregunta_posicion_actual()

    desplazamiento = mejor_z - z_actual

    if desplazamiento != 0:

        impresora.desp_lin_rel(
            0,
            0,
            desplazamiento
        )

        time.sleep(espera_movimiento)


    # ========================================================
    # GRAFICAR LA METIRCA
    # ========================================================

    if graficamos:

        fig, ax = plt.subplots()

        ax.plot(
            posiciones_z,
            valores_metrica,
            marker="o"
        )

        ax.scatter(
            mejor_z,
            mejor_valor,
            s=80,
            label="Máximo"
        )

        ax.set_xlabel("Z [mm]")
        ax.set_ylabel(metrica.__name__)
        ax.set_title("Autofocus")
        ax.grid()
        ax.legend()

        plt.show()


    # ========================================================
    # DEVOLVER RESULTADOS
    # ========================================================

    return mejor_z, posiciones_z, valores_metrica


#%%



grua = TrimakerCOSMOS("COM11")

grua.home()



#%%



cam = OpenMVCamera("COM7")

cam.conectar()



#%%



mejor_z, posiciones_z, valores_metrica = autofocus(
    cam,
    grua,
    tenengrad,
    pasos_abajo=5,
    pasos_arriba=5,
    paso_z=0.0025,
    espera_movimiento=1,
    graficamos=True,
    guardamos=True,
    carpeta_guardado=r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.18 cadena de ordenes - nuevas piezas 3d - autoenfoque\Configuracion 10\Pasada 1"
    )



#%%