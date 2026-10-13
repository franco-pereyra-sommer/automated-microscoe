#%%



from pathlib import Path
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt

import time
from datetime import datetime

from PIL import Image
import io


# ============================================================
# EXTRAER COORDENADAS DEL NOMBRE
# ============================================================

def extraer_coordenadas(ruta):
    """
    Extrae x, y, z de un nombre como:

    2026-06-25_18-04-55 (150.4375, 100.0, 42.4375).jpg
    """

    nombre = ruta.stem

    patron = r"\(([-+0-9.eE]+),\s*([-+0-9.eE]+),\s*([-+0-9.eE]+)\)"

    coincidencia = re.search(patron, nombre)

    if coincidencia is None:
        raise ValueError(
            f"No se encontraron coordenadas en:\n{ruta.name}"
        )

    x, y, z = map(float, coincidencia.groups())

    return x, y, z


# ============================================================
# CARGAR IMAGEN
# ============================================================

def cargar_imagen(ruta):

    imagen = cv2.imread(str(ruta))

    if imagen is None:
        raise ValueError(
            f"No se pudo cargar:\n{ruta}"
        )

    return imagen


# ============================================================
# MÉTRICAS
# ============================================================

def intensidad_fondo(imagen):
    
    gray = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )
    
    return 255 - gray.mean()


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


# ============================================================
# ANALIZAR CARPETA
# ============================================================

def analizar_carpeta(carpeta):

    carpeta = Path(carpeta)

    archivos = sorted(
        carpeta.glob("*.jpg")
    )

    if len(archivos) == 0:
        raise ValueError(
            "No se encontraron imágenes JPG en la carpeta."
        )

    resultados = []

    for archivo in archivos:

        try:
            x, y, z = extraer_coordenadas(archivo)

            imagen = cargar_imagen(archivo)

            laplaciano = laplacian_variance(imagen)
            tenengrad_val = tenengrad(imagen)
            varianza = varianza_intensidad(imagen)

            resultados.append({
                "archivo": archivo.name,
                "x": x,
                "y": y,
                "z": z,
                "laplacian": laplaciano,
                "tenengrad": tenengrad_val,
                "varianza": varianza
            })

        except Exception as e:

            print(
                f"\nNo se pudo procesar {archivo.name}"
            )

            print(e)

    # Ordenar según Z
    resultados.sort(
        key=lambda resultado: resultado["z"]
    )

    return resultados


# ============================================================
# MOSTRAR MÁXIMOS
# ============================================================

def mostrar_maximos(resultados):

    metricas = {
        "Laplaciano": "laplacian",
        "Tenengrad": "tenengrad",
        "Varianza de intensidad": "varianza"
    }

    print("\n" + "=" * 60)
    print("MÁXIMOS")
    print("=" * 60)

    for nombre, clave in metricas.items():

        mejor = max(
            resultados,
            key=lambda resultado: resultado[clave]
        )

        print(f"\n{nombre}")

        print(
            f"  Métrica = {mejor[clave]:.6f}"
        )

        print(
            f"  Z       = {mejor['z']:.6f} mm"
        )

        print(
            f"  Archivo = {mejor['archivo']}"
        )


# ============================================================
# GRAFICAR
# ============================================================

def graficar_metricas(resultados):

    z = np.array([
        resultado["z"]
        for resultado in resultados
    ])

    laplacian = np.array([
        resultado["laplacian"]
        for resultado in resultados
    ])

    tenengrad_val = np.array([
        resultado["tenengrad"]
        for resultado in resultados
    ])

    varianza = np.array([
        resultado["varianza"]
        for resultado in resultados
    ])


    # --------------------------------------------------------
    # Laplaciano
    # --------------------------------------------------------

    fig, ax = plt.subplots()

    ax.plot(
        z,
        laplacian,
        marker="o"
    )

    ax.set_xlabel("Z [mm]")
    ax.set_ylabel("Varianza del Laplaciano")
    ax.set_title("Métrica de enfoque - Laplaciano")
    ax.grid()

    plt.show()


    # --------------------------------------------------------
    # Tenengrad
    # --------------------------------------------------------

    fig, ax = plt.subplots()

    ax.plot(
        z,
        tenengrad_val,
        marker="o"
    )

    ax.set_xlabel("Z [mm]")
    ax.set_ylabel("Tenengrad")
    ax.set_title("Métrica de enfoque - Tenengrad")
    ax.grid()

    plt.show()


    # --------------------------------------------------------
    # Varianza de intensidad
    # --------------------------------------------------------

    fig, ax = plt.subplots()

    ax.plot(
        z,
        varianza,
        marker="o"
    )

    ax.set_xlabel("Z [mm]")
    ax.set_ylabel("Varianza de intensidad")
    ax.set_title("Métrica de enfoque - Varianza de intensidad")
    ax.grid()

    plt.show()


def graficar_metricas_mas_imagenes_vieja(resultados, carpeta):

    metricas = {
        "Laplaciano": "laplacian",
        "Tenengrad": "tenengrad",
        "Varianza de intensidad": "varianza"
    }

    for nombre_metrica, clave in metricas.items():

        indice_maximo = np.argmax([
            resultado[clave]
            for resultado in resultados
        ])

        indices = [
            indice_maximo - 1,
            indice_maximo,
            indice_maximo + 1
        ]

        fig, ax = plt.subplots(
            1,
            3,
            figsize=(15, 5)
        )

        # fig.suptitle(f"Configuracion {configuracion} - Pasada {pasada}")

        for eje, indice in zip(ax, indices):

            # Si el máximo está en el primer o último lugar
            # no habría imagen anterior/posterior.
            if indice < 0 or indice >= len(resultados):
                eje.axis("off")
                continue

            resultado = resultados[indice]

            ruta_imagen = (
                Path(carpeta)
                / resultado["archivo"]
            )

            imagen = cv2.imread(
                str(ruta_imagen)
            )

            imagen = cv2.cvtColor(
                imagen,
                cv2.COLOR_BGR2RGB
            )

            eje.imshow(imagen)

            # El nombre tiene:
            # 2026-08-18_14-07-33 (...)
            # La hora ocupa los caracteres 11:19

            hora = resultado["archivo"][11:19]

            eje.set_title(hora)

            eje.axis("off")

        fig.suptitle(
            f"{nombre_metrica} - máximo",
            fontsize=14
        )

        plt.tight_layout()

        plt.show()


def grafico_del_maximo_de_la_metrica_con_imagenes(resultados, carpeta):

    metricas = {
        "Laplaciano": "laplacian",
        "Tenengrad": "tenengrad",
        "Varianza de intensidad": "varianza",
        # "Intensidad de fondo": "intensidad_fondo",
        "Fraccion rojo":"fraccion_rojo",
        "Saturacion Media":"saturacion_media"
    }

    for nombre_metrica, clave in metricas.items():

        indice_maximo = np.argmax([
            resultado[clave]
            for resultado in resultados
        ])

        indices = [
            indice_maximo - 1,
            indice_maximo,
            indice_maximo + 1
        ]

        fig, ax = plt.subplots(
            1,
            3,
            figsize=(15, 5)
        )

        # fig.suptitle(f"Configuracion {configuracion} - Pasada {pasada}")

        for eje, indice in zip(ax, indices):

            # Si el máximo está en el primer o último lugar
            # no habría imagen anterior/posterior.
            if indice < 0 or indice >= len(resultados):
                eje.axis("off")
                continue

            resultado = resultados[indice]

            ruta_imagen = (
                Path(carpeta)
                / resultado["archivo"]
            )

            imagen = cv2.imread(
                str(ruta_imagen)
            )

            imagen = cv2.cvtColor(
                imagen,
                cv2.COLOR_BGR2RGB
            )

            eje.imshow(imagen)

            eje.set_title(resultado["z"])

            eje.axis("off")

        fig.suptitle(
            f"{nombre_metrica} - máximo",
            fontsize=14
        )

        plt.tight_layout()

        plt.show()


def graficar_metricas_mas_imagenes(resultados, carpeta):

    metricas = {
        "Laplaciano": "laplacian",
        "Tenengrad": "tenengrad",
        "Varianza de intensidad": "varianza"
    }

    carpeta = Path(carpeta)

    configuracion = next(
        parte for parte in carpeta.parts
        if parte.lower().startswith("configuracion ")
    )

    pasada = next(
        parte for parte in carpeta.parts
        if parte.lower().startswith("pasada ")
    )

    for nombre_metrica, clave in metricas.items():

        indice_maximo = np.argmax([
            resultado[clave]
            for resultado in resultados
        ])

        indices = [
            indice_maximo - 1,
            indice_maximo,
            indice_maximo + 1
        ]

        fig, ax = plt.subplots(
            1,
            3,
            figsize=(15, 5)
        )

        fig.suptitle(
            f"{configuracion} - {pasada}"
        )

        for eje, indice in zip(ax, indices):

            if indice < 0 or indice >= len(resultados):
                eje.axis("off")
                continue

            resultado = resultados[indice]

            ruta_imagen = (
                carpeta
                / resultado["archivo"]
            )

            imagen = cv2.imread(
                str(ruta_imagen)
            )

            imagen = cv2.cvtColor(
                imagen,
                cv2.COLOR_BGR2RGB
            )

            eje.imshow(imagen)

            hora = resultado["archivo"][11:19]

            eje.set_title(hora)

            eje.axis("off")

        fig.suptitle(
            f"{nombre_metrica} - máximo\n"
            f"{configuracion} - {pasada}",
            fontsize=14
        )

        plt.tight_layout()
        plt.show()


def bytes_a_imagen(image_data):

    imagen = Image.open(
        io.BytesIO(image_data)
    )

    imagen = np.array(imagen)

    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(
            imagen,
            cv2.COLOR_RGB2BGR
        )

    return imagen



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
    autofocus_completo = False

    try:

        if pasos_abajo > 0:

            impresora.desp_lin_rel(
                0,
                0,
                -pasos_abajo * paso_z
            )
            time.sleep(1)
            # time.sleep(espera_movimiento)


        # ========================================================
        # BARRIDO EN Z
        # ========================================================

        cantidad_fotos = pasos_arriba + pasos_abajo + 1

        for paso in range(cantidad_fotos):

            # --------------------------------------------
            # Tomar fotografía
            # --------------------------------------------

            image_data = camara.capturar()

            # _, _, z_actual = impresora.pregunta_posicion_actual()

            if guardamos:

                hora_actual = datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S"
                )

                camara.guardar_imagen(
                    image_data=image_data,
                    carpeta=carpeta_guardado,
                    nombre=f"{hora_actual} ({impresora.posicion_x}, {impresora.posicion_y}, {impresora.posicion_z})",
                    mostrar=False
                )


            # --------------------------------------------
            # Convertir JPG -> imagen OpenCV
            # --------------------------------------------

            imagen = bytes_a_imagen(image_data)

            # --------------------------------------------
            # Calcular métrica
            # --------------------------------------------

            valor = metrica(imagen)


            # --------------------------------------------
            # Guardar posición y métrica
            # --------------------------------------------

            # _, _, z_actual = impresora.pregunta_posicion_actual()

            posiciones_z.append(impresora.posicion_z)
            valores_metrica.append(valor)


            # print(
            #     f"Foto {paso + 1}/{cantidad_fotos} | "
            #     f"Z = {z_actual:.6f} mm | "
            #     f"Métrica = {valor:.6f}"
            # )


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

        autofocus_completo = True

    finally:

        if not autofocus_completo:

            _, _, z_actual = impresora.pregunta_posicion_actual()

            desplazamiento = z_inicial - z_actual

            if desplazamiento != 0:

                impresora.desp_lin_rel(
                    0,
                    0,
                    desplazamiento
                )

                time.sleep(espera_movimiento)

            print(
                f"Autofocus interrumpido. "
                f"Z restaurado a {z_inicial:.6f} mm"
            )

        else:
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
    # DEVOLVER RESULTADOS
    # ========================================================

    return mejor_z, posiciones_z, valores_metrica


# Ejemplo de uso


# mejor_z, posiciones_z, valores_metrica = autofocus(
#     camara=cam,
#     impresora=grua,
#     metrica=tenengrad,
#     pasos_arriba=5,
#     pasos_abajo=5,
#     graficamos=True,
#     guardamos=True,
#     carpeta_guardado=r"C:\Fotos\autofocus"


# Funcion re colada, no la uso para enfoque, pero usa los dos dispositivos

def barrido_z_mas_guardar_imagenes(
    camara,
    impresora,
    x,
    y,
    z_max = 40,
    z_limites = [None],
    pasos_lista = [None],
    tiempo_espera_lista = [0.5],
    cam_pixformat = ["RGB565"],
    cam_framesize = ["QVGA"],
    carpeta_guardado = None
):
    """
    Te hace un barrido en z y te guarda las imagenes en la carpeta que le pedis

    z_limites: Porque existe una lista que contiene z's? -> Para poder decir por ejemplo: desde z=30 hasta z=29 movete con un paso de 100 micrones; desde z=29 hasta z=28 movete de a 50 micrones.
    z_limites: es una lista
    pasos_lista: es una lista
    """

    # La camara va al punto mas alejado (mientras mas chico es z mas cerca esta la muestra y la camara)
    impresora.desp_lin_abs(x,y,z_max)

    # time.sleep(10)


    ###
    # Advertencias
    ###

    for z_min, paso_barrido in zip(z_limites, pasos_lista):
    
        if z_max < z_min:
            raise ValueError(
                    f"z_min = {z_min} tiene que ser menor que z_max = {z_max}"
                    )

        if z_min == None:
            raise ValueError(
                f"z_min = {z_min}, no puede ser None. {z_limites = }"
                )

        if paso_barrido == None:
                raise ValueError(
                    f"paso = {paso_barrido}, no puede ser None. {z_limites = }"
                    )

    numero = 0

    for z_min, paso_barrido, tiempo_espera in zip(z_limites, pasos_lista, tiempo_espera_lista):

        z_min = np.round(z_min, 4)

        # zs = np.arange(z_min, impresora.posicion_z, paso_barrido)[::-1]

        # zs = np.linspace(z_min, impresora.posicion_z, int((impresora.posicion_z - z_min )/paso_barrido))[::-1]

        zs = np.arange(impresora.posicion_z, z_min, -paso_barrido)

        if numero > 0: 
            zs = zs[1:]
        
        print(f"{zs = }")

        for z_a_usar in zs:

            z_a_usar = np.round(z_a_usar, 4)

            impresora.desp_lin_abs(x,y,z_a_usar)

            time.sleep(tiempo_espera)

            for pixformat in cam_pixformat:

                if pixformat is not None:

                    camara.set_pixformat(pixformat)

                for framesize in cam_framesize:

                    camara.set_framesize(framesize) 

                    carpeta_guardado_final = rf"{carpeta_guardado}\Pixformat {pixformat}- resolucion {framesize}"

                    hora_actual = datetime.now().strftime(
                        "%Y-%m-%d_%H-%M-%S"
                    )

                    image_data = camara.capturar()

                    camara.guardar_imagen(
                        image_data=image_data,
                        carpeta=carpeta_guardado_final,
                        nombre=f"{hora_actual} ({impresora.posicion_x}, {impresora.posicion_y}, {impresora.posicion_z})",
                        mostrar=False
                    )

        numero = numero + 1 

    # Vuelve al punto de partida
    impresora.desp_lin_abs(x,y,z_max)


# Otra funcion colada que viene de claude_busqueda_muestra
def fraccion_pixeles_rojos(imagen_bgr, s_min=50, v_min=50, v_max=255):

    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)

    rango_bajo_1 = np.array([0, s_min, v_min])
    rango_alto_1 = np.array([10, 255, v_max])

    rango_bajo_2 = np.array([160, s_min, v_min])
    rango_alto_2 = np.array([180, 255, v_max])

    mascara_1 = cv2.inRange(hsv, rango_bajo_1, rango_alto_1)
    mascara_2 = cv2.inRange(hsv, rango_bajo_2, rango_alto_2)

    mascara = cv2.bitwise_or(mascara_1, mascara_2)

    return mascara.mean() / 255.0




# ============================================================
# METRICAS DE COLOR
# ============================================================

def medias_de_color(imagen_bgr):
    """
    Media de cada canal (R, G, B) y de la saturación (HSV).
    Sirve para ver, sin depender de un umbral fijo, cómo va
    apareciendo el color a medida que cambia Z.
    """

    b, g, r = cv2.split(imagen_bgr.astype(np.float64))

    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    saturacion_media = hsv[:, :, 1].mean()

    return {
        "media_r": r.mean(),
        "media_g": g.mean(),
        "media_b": b.mean(),
        "saturacion_media": saturacion_media,
    }


# ============================================================
# ANALIZAR CARPETA (TODAS LAS METRICAS)
# ============================================================

def analizar_carpeta_completo(carpeta):

    carpeta = Path(carpeta)

    archivos = sorted(carpeta.glob("*.jpg"))

    if len(archivos) == 0:
        raise ValueError(
            f"No se encontraron imágenes JPG en:\n{carpeta}"
        )

    resultados = []

    for archivo in archivos:

        try:
            x, y, z = extraer_coordenadas(archivo)

            imagen = cargar_imagen(archivo)

            fila = {
                "archivo": archivo.name,
                "x": x,
                "y": y,
                "z": z,
                "laplacian": laplacian_variance(imagen),
                "tenengrad": tenengrad(imagen),
                "varianza": varianza_intensidad(imagen),
                "intensidad_fondo": intensidad_fondo(imagen),
                "fraccion_rojo": fraccion_pixeles_rojos(imagen),
            }

            fila.update(medias_de_color(imagen))

            resultados.append(fila)

        except Exception as e:

            print(f"No se pudo procesar {archivo.name}: {e}")

    resultados.sort(key=lambda resultado: resultado["z"])

    return resultados


# ============================================================
# AGRUPAR POR (x, y)
# ============================================================
# Por si en la carpeta terminan mezclados barridos de distintas
# posiciones XY (por ejemplo, si corriste el barrido más de una
# vez apuntando a distintos campos visuales).

def agrupar_por_xy(resultados, decimales=3):

    grupos = {}

    for resultado in resultados:

        clave = (
            round(resultado["x"], decimales),
            round(resultado["y"], decimales),
        )

        grupos.setdefault(clave, []).append(resultado)

    return grupos


# ============================================================
# GRAFICAR TODAS LAS METRICAS VS Z
# ============================================================

def graficar_todas_las_metricas(resultados, titulo=""):

    metricas = {
        "Laplaciano": "laplacian",
        "Tenengrad": "tenengrad",
        "Varianza de intensidad": "varianza",
        "Intensidad de fondo": "intensidad_fondo",
        "Fracción de rojo": "fraccion_rojo",
        "Saturación media": "saturacion_media"
        # "Media canal R": "media_r",
        # "Media canal G": "media_g",
    }

    z = np.array([resultado["z"] for resultado in resultados])

    fig, ejes = plt.subplots(2, 4, figsize=(22, 9))
    ejes = ejes.flatten()

    for eje, (nombre, clave) in zip(ejes, metricas.items()):

        valores = np.array([resultado[clave] for resultado in resultados])

        eje.plot(z, valores, marker="o")
        eje.set_xlabel("Z [mm]")
        eje.set_ylabel(nombre)
        eje.set_title(nombre)
        eje.grid()

    fig.suptitle(titulo if titulo else "Métricas vs Z", fontsize=14)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # R, G, B juntos, para ver cómo se separan entre sí
    # --------------------------------------------------------

    fig, ax = plt.subplots()

    ax.plot(z, [r["media_r"] for r in resultados], marker="o", color="red", label="R")
    ax.plot(z, [r["media_g"] for r in resultados], marker="o", color="green", label="G")
    ax.plot(z, [r["media_b"] for r in resultados], marker="o", color="blue", label="B")

    ax.set_xlabel("Z [mm]")
    ax.set_ylabel("Media del canal")
    ax.set_title(f"Canales R/G/B vs Z\n{titulo}")
    ax.grid()
    ax.legend()

    plt.show()

    # --------------------------------------------------------
    # Dispersión varianza vs fracción de rojo, coloreado por Z
    # --------------------------------------------------------

    fig, ax = plt.subplots()

    dispersión = ax.scatter(
        [r["varianza"] for r in resultados],
        [r["fraccion_rojo"] for r in resultados],
        c=z,
        cmap="viridis",
    )

    ax.set_xlabel("Varianza de intensidad")
    ax.set_ylabel("Fracción de rojo")
    ax.set_title(f"Varianza vs Fracción de rojo (color = Z)\n{titulo}")
    ax.grid()

    fig.colorbar(dispersión, ax=ax, label="Z [mm]")

    plt.show()


# ============================================================
# EXPORTAR A CSV (OPCIONAL, PARA MIRAR LOS NUMEROS EN DETALLE)
# ============================================================

# def exportar_csv(resultados, ruta_csv):

#     if len(resultados) == 0:
#         return

#     columnas = list(resultados[0].keys())

#     with open(ruta_csv, "w", newline="", encoding="utf-8") as archivo_csv:

#         escritor = csv.DictWriter(archivo_csv, fieldnames=columnas)
#         escritor.writeheader()
#         escritor.writerows(resultados)

#     print(f"CSV guardado en:\n{ruta_csv}")



def graficar_todas_las_metricas_con_distintos_campos_visuales(
        resultados,
        titulo="",
        fig= None,
        ejes=None,
        zmin=None,
        zmax=None,
        label_de_mas=None,
        normalizacion_a_uno=False
        ):

    metricas = {
        "Laplaciano": "laplacian",
        "Tenengrad": "tenengrad",
        "Varianza de intensidad": "varianza",
        "Intensidad de fondo": "intensidad_fondo",
        "Fracción de rojo": "fraccion_rojo",
        "Saturación media": "saturacion_media"
        # "Media canal R": "media_r",
        # "Media canal G": "media_g",
    }

    z = np.array([resultado["z"] for resultado in resultados])

    for eje, (nombre, clave) in zip(ejes, metricas.items()):

        valores = np.array([resultado[clave] for resultado in resultados])

        if normalizacion_a_uno:
            valores = valores / np.max(valores)

        eje.plot(
            z,
            valores,
            marker=".",
            # label = f"x{resultados[0]["x"]}-y{resultados[0]["y"]} {label_de_mas}",
            label = f"{label_de_mas}",
            linewidth=1)
        eje.set_xlabel("Z [mm]")
        eje.set_ylabel(nombre)
        # eje.set_title(nombre)
        eje.grid(True)
        if zmin is not None and zmax is not None:
            eje.set_xlim(zmin, zmax)
        eje.legend()


#%%



if __name__ == "__main__":

    # ==========================================
    # PRUEBA/ APLICACION
    # ==========================================
    # C:\BackUp\Facultad\Labo 6\Fotos\26.06.25 Compuesto con obj y alguna lente + barrido z\sangre 2 lente mas grande otro con mas distancia entre el sensor y el objetivo
    # ruta = r"C:\Fotos\foto.jpg"

    ruta = r"C:\BackUp\Facultad\Labo 6\Fotos\26.06.25 Compuesto con obj y alguna lente + barrido z\sangre 2 lente mas grande otro con mas distancia entre el sensor y el objetivo\2026-06-25_18-04-55 (150.43749999999983, 100.0, 42.43749999999977).jpg"

    ruta = r"C:\BackUp\Facultad\Labo 6\Fotos\26.07.03 Combinaciones de lentes mas el objetivo\grilla y sangre\sangre tubo muy separada del sensor - super tubo 2 SUPER HD.bmp"

    imagen = cargar_imagen(ruta)

    print(
        "Varianza del Laplaciano:",
        laplacian_variance(imagen)
    )

    print(
        "Tenengrad:",
        tenengrad(imagen)
    )

    print(
        "Varianza de intensidad:",
        varianza_intensidad(imagen)
    )


    # Mostrar imagen

    gray = cv2.cvtColor(
            imagen,
            cv2.COLOR_BGR2GRAY
        )

    fig, ax = plt.subplots()

    ax.imshow(gray, cmap="gray")
    ax.set_title("Imagen")
    ax.axis("off")

    plt.show()



#%%



if __name__ == "__main__":

    carpeta = r"C:\BackUp\Facultad\Labo 6\Fotos\26.06.25 Compuesto con obj y alguna lente + barrido z\sangre 2 lente mas grande otro con mas distancia entre el sensor y el objetivo"

    resultados = analizar_carpeta(carpeta)

    mostrar_maximos(resultados)

    graficar_metricas(resultados)



# %%



if __name__ == "__main__":

    carpeta = r"C:\BackUp\Facultad\Labo 6\Fotos\26.06.23 Objetivo mas sensor + barrido grilla + resolucion en z\sangre 1"

    resultados = analizar_carpeta(carpeta)

    mostrar_maximos(resultados)

    graficar_metricas(resultados)



# %%



if __name__ == "__main__":

    carpeta = r"C:\BackUp\Facultad\Labo 6\Fotos\26.06.23 Objetivo mas sensor + barrido grilla + resolucion en z\lol"

    resultados = analizar_carpeta(carpeta)

    mostrar_maximos(resultados)

    graficar_metricas(resultados)



# %%



if __name__ == "__main__":

    carpeta = r"C:\BackUp\Facultad\Labo 6\Fotos\26.06.23 Objetivo mas sensor + barrido grilla + resolucion en z\pelo 4"

    resultados = analizar_carpeta(carpeta)

    mostrar_maximos(resultados)

    graficar_metricas(resultados)



# %%


# def autofocus_no_me_gusto_para_nada(carpeta, metrica):

#     # Analizar todas las imágenes de la carpeta
#     resultados = analizar_carpeta(carpeta)

#     if len(resultados) == 0:
#         raise ValueError(
#             "No hay imágenes válidas para realizar el autofocus."
#         )

#     # --------------------------------------------------------
#     # Obtener la métrica elegida para cada imagen
#     # --------------------------------------------------------

#     valores_metrica = []

#     for resultado in resultados:

#         imagen = cargar_imagen(
#             Path(carpeta) / resultado["archivo"]
#         )

#         valor = metrica(imagen)

#         valores_metrica.append(valor)

#     # --------------------------------------------------------
#     # Encontrar el máximo
#     # --------------------------------------------------------

#     indice_maximo = np.argmax(valores_metrica)

#     mejor_resultado = resultados[indice_maximo]

#     mejor_z = mejor_resultado["z"]
#     mejor_valor = valores_metrica[indice_maximo]

#     # --------------------------------------------------------
#     # Mostrar resultado
#     # --------------------------------------------------------

#     print("\n" + "=" * 60)
#     print("AUTOFOCUS")
#     print("=" * 60)

#     print(f"Métrica utilizada: {metrica.__name__}")
#     print(f"Mejor Z: {mejor_z:.6f} mm")
#     print(f"Valor de la métrica: {mejor_valor:.6f}")
#     print(f"Imagen: {mejor_resultado['archivo']}")

#     # --------------------------------------------------------
#     # Gráfico
#     # --------------------------------------------------------

#     z = np.array([
#         resultado["z"]
#         for resultado in resultados
#     ])

#     valores_metrica = np.array(valores_metrica)

#     fig, ax = plt.subplots()

#     ax.plot(
#         z,
#         valores_metrica,
#         marker="o"
#     )

#     ax.scatter(
#         mejor_z,
#         mejor_valor,
#         s=80,
#         label="Máximo"
#     )

#     ax.set_xlabel("Z [mm]")
#     ax.set_ylabel(metrica.__name__)
#     ax.set_title("Autofocus")
#     ax.grid()
#     ax.legend()

#     plt.show()

#     return mejor_z, mejor_valor
