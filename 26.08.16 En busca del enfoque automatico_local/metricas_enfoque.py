#%%



from pathlib import Path
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt

import time
from datetime import datetime


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


def autofocus_no_me_gusto_para_nada(carpeta, metrica):

    # Analizar todas las imágenes de la carpeta
    resultados = analizar_carpeta(carpeta)

    if len(resultados) == 0:
        raise ValueError(
            "No hay imágenes válidas para realizar el autofocus."
        )

    # --------------------------------------------------------
    # Obtener la métrica elegida para cada imagen
    # --------------------------------------------------------

    valores_metrica = []

    for resultado in resultados:

        imagen = cargar_imagen(
            Path(carpeta) / resultado["archivo"]
        )

        valor = metrica(imagen)

        valores_metrica.append(valor)

    # --------------------------------------------------------
    # Encontrar el máximo
    # --------------------------------------------------------

    indice_maximo = np.argmax(valores_metrica)

    mejor_resultado = resultados[indice_maximo]

    mejor_z = mejor_resultado["z"]
    mejor_valor = valores_metrica[indice_maximo]

    # --------------------------------------------------------
    # Mostrar resultado
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("AUTOFOCUS")
    print("=" * 60)

    print(f"Métrica utilizada: {metrica.__name__}")
    print(f"Mejor Z: {mejor_z:.6f} mm")
    print(f"Valor de la métrica: {mejor_valor:.6f}")
    print(f"Imagen: {mejor_resultado['archivo']}")

    # --------------------------------------------------------
    # Gráfico
    # --------------------------------------------------------

    z = np.array([
        resultado["z"]
        for resultado in resultados
    ])

    valores_metrica = np.array(valores_metrica)

    fig, ax = plt.subplots()

    ax.plot(
        z,
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

    return mejor_z, mejor_valor
