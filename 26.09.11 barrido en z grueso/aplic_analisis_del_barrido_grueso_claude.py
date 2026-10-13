#%%



# Agrego la raiz del proyecto
import sys
import os

os.environ["PYTHONPATH"] = r"C:\BackUp\Facultad\Labo 7\env_labo7"

sys.path.insert(0, os.environ["PYTHONPATH"])


from pathlib import Path
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt

from metricas_enfoque import (
    extraer_coordenadas,
    cargar_imagen,
    laplacian_variance,
    tenengrad,
    varianza_intensidad,
    intensidad_fondo,
)

# from busqueda_muestra import fraccion_pixeles_rojos

from metricas_enfoque import (
    fraccion_pixeles_rojos
)


# Estilo de graficos
import seaborn as sns
# from seaborn import axes_style
sns.reset_defaults()
sns.set_style(rc = {'axes.grid': True,'axes.axisbelow': True,'grid.color': '.8', 'grid.linestyle': '-'})
sns.set_context("talk", font_scale=1.0)




#%%



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

def exportar_csv(resultados, ruta_csv):

    if len(resultados) == 0:
        return

    columnas = list(resultados[0].keys())

    with open(ruta_csv, "w", newline="", encoding="utf-8") as archivo_csv:

        escritor = csv.DictWriter(archivo_csv, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(resultados)

    print(f"CSV guardado en:\n{ruta_csv}")



#%%



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ============================================================
    # RUTA A LA CARPETA DEL BARRIDO
    # ============================================================
    for resol in ["QVGA"]:#["FHD", "HD", "QVGA"]:

        carpetas = [
        rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 0 xy 150_150\Pixformat RGB565- resolucion {resol}",
        rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 0 xy 160_150\Pixformat RGB565- resolucion {resol}",
        # rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 0 xy 161_150\Pixformat RGB565- resolucion {resol}",
        # rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 0 xy 165_150\Pixformat RGB565- resolucion {resol}",
        # rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 1 xy 150_150\Pixformat RGB565- resolucion {resol}",
        # rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 1 xy 160_150\Pixformat RGB565- resolucion {resol}",
        # rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 1 xy 161_150\Pixformat RGB565- resolucion {resol}",
        # rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 2 xy 150_150\Pixformat RGB565- resolucion {resol}",
        # rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 3 xy 150_150\Pixformat RGB565- resolucion {resol}",
        # rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 4 xy 150_150\Pixformat RGB565- resolucion WQXGA2"
        ]

        for RUTA_CARPETA in carpetas:

        # RUTA_CARPETA = (
        #     r"C:\BackUp\Facultad\Labo 7\Fotos"
        #     r"\26.09.11 barrido gruso z en disitintos cv"
        #     r"\Pixformat RGB565\Resolucion HD"
        # )

            resultados = analizar_carpeta_completo(RUTA_CARPETA)

            print(f"Se procesaron {len(resultados)} imágenes.")

            grupos = agrupar_por_xy(resultados)

            if len(grupos) > 1:
                print(
                    f"\nATENCIÓN: se encontraron {len(grupos)} posiciones (x, y) "
                    f"distintas en la carpeta. Se grafica cada una por separado."
                )

            for (x, y), resultados_grupo in grupos.items():

                print(f"{RUTA_CARPETA = }")

                graficar_todas_las_metricas(
                    resultados_grupo,
                    titulo=f"x = {x}, y = {y}",
                )

            # exportar_csv(
            #     resultados,
            #     Path(RUTA_CARPETA) / "analisis_metricas.csv",
            # )



#%%



# ============================================================
# GRAFICAR TODAS LAS METRICAS VS Z
# ============================================================

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

    # plt.tight_layout()
    # plt.show()



#%%



if __name__ == "__main__":

    # ============================================================
    # RUTA A LA CARPETA DEL BARRIDO
    # ============================================================

    fig, ejes = plt.subplots(2, 3, figsize=(22, 9))
    ejes = ejes.flatten()

    for resol in ["QVGA"]:#["FHD", "HD", "QVGA"]:#

        carpetas = [
        rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 0 xy 150_150\Pixformat RGB565- resolucion {resol}",
        rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 0 xy 160_150\Pixformat RGB565- resolucion {resol}",
        # rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 0 xy 161_150\Pixformat RGB565- resolucion {resol}",
        # rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 0 xy 165_150\Pixformat RGB565- resolucion {resol}",
        # rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 1 xy 150_150\Pixformat RGB565- resolucion {resol}",
        # rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 1 xy 160_150\Pixformat RGB565- resolucion {resol}",
        # rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 1 xy 161_150\Pixformat RGB565- resolucion {resol}",
        # rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 2 xy 150_150\Pixformat RGB565- resolucion {resol}",
        # rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 3 xy 150_150\Pixformat RGB565- resolucion {resol}",
        # rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 4 xy 150_150\Pixformat RGB565- resolucion WQXGA2"
        ]

        for RUTA_CARPETA, label in zip(carpetas, ["Con globulos rojos", "Sin globulos rojos"]): #carpetas:#
            
            resultados = analizar_carpeta_completo(RUTA_CARPETA)

            # print(f"Se procesaron {len(resultados)} imágenes.")

            print(f"{RUTA_CARPETA = }")

            graficar_todas_las_metricas_con_distintos_campos_visuales(
                resultados,
                fig=fig,
                ejes=ejes,
                # zmin = 27.81,
                # zmax=27.89,
                label_de_mas = label
                # titulo=f"x = {x}, y = {y}",
            )


    fig.suptitle("Métricas vs Z", fontsize=14)
    plt.tight_layout()
    plt.show()



# %% Estoy viendo como esta armado



# RUTA_CARPETA = rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.18 barrido gruso z en disitintos cv\Pasada 0 xy 150_150\Pixformat RGB565- resolucion QVGA"
# resultados = analizar_carpeta_completo(RUTA_CARPETA)

# print(resultados[0]["x"])



# %%


if __name__ == "__main__":

    # ============================================================
    # RUTA A LA CARPETA DEL BARRIDO
    # ============================================================

    fig, ejes = plt.subplots(2, 3, figsize=(22, 9))
    ejes = ejes.flatten()

    for resol in ["VGA"]:#["FHD", "HD", "QVGA"]:#

        carpetas = [
        rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.25 preparado - barrido a un cv de gls\barrido grueso en z\Pasada 8.0 xy 150_150\Pixformat RGB565- resolucion {resol}",
        rf"C:\BackUp\Facultad\Labo 7\Fotos\26.09.25 preparado - barrido a un cv de gls\barrido grueso en z\Pasada 9.0 xy 150_150\Pixformat RGB565- resolucion {resol}",
        ]

        for RUTA_CARPETA, label in zip(carpetas, ["8.0V", "9.0V"]): #carpetas:#
            
            resultados = analizar_carpeta_completo(RUTA_CARPETA)

            # print(f"Se procesaron {len(resultados)} imágenes.")

            print(f"{RUTA_CARPETA = }")

            graficar_todas_las_metricas_con_distintos_campos_visuales(
                resultados,
                fig=fig,
                ejes=ejes,
                # zmin = 27.81,
                # zmax=27.89,
                label_de_mas = label,
                # titulo=f"x = {x}, y = {y}",
                normalizacion_a_uno=False
            )


    fig.suptitle("Métricas vs Z")
    plt.tight_layout()
    plt.show()



# %%




if __name__ == "__main__":

    # ============================================================
    # RUTA A LA CARPETA DEL BARRIDO
    # ============================================================

    fig, ejes = plt.subplots(2, 3, figsize=(22, 9))
    ejes = ejes.flatten()

    for resol in ["VGA"]:#["FHD", "HD", "QVGA"]:#

        carpetas = [
        rf"C:\BackUp\Facultad\Labo 7\Fotos\26.10.02 mosaico parasitos fijos pasada 4\0000 i0_j0",
        rf"C:\BackUp\Facultad\Labo 7\Fotos\26.10.02 mosaico parasitos fijos pasada 4\0002 i1_j1",
        ]

        for RUTA_CARPETA, label in zip(carpetas, ["0000 i0_j0", "0002 i1_j1"]): #carpetas:#
            
            resultados = analizar_carpeta_completo(RUTA_CARPETA)

            # print(f"Se procesaron {len(resultados)} imágenes.")

            print(f"{RUTA_CARPETA = }")

            graficar_todas_las_metricas_con_distintos_campos_visuales(
                resultados,
                fig=fig,
                ejes=ejes,
                # zmin = 27.81,
                # zmax=27.89,
                label_de_mas = label,
                # titulo=f"x = {x}, y = {y}",
                normalizacion_a_uno=False
            )


    fig.suptitle("Métricas vs Z")
    plt.tight_layout()
    plt.show()



# %%
