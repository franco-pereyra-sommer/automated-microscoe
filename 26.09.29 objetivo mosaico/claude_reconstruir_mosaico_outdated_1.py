#%%


# Agrego la raiz del proyecto
import sys
import os

os.environ["PYTHONPATH"] = r"C:\BackUp\Facultad\Labo 7\env_labo7"

sys.path.insert(0, os.environ["PYTHONPATH"])

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from metricas_enfoque import extraer_coordenadas, cargar_imagen


# ============================================================
# CARGAR TILES DE UNA CARPETA
# ============================================================

def cargar_tiles(carpeta, extension="*.jpg"):
    """
    Carga todas las imágenes de 'carpeta' (no recursivo, así que no
    entra a la subcarpeta de fotos de autofoco) y les extrae (x, y, z)
    del nombre de archivo con extraer_coordenadas() — la misma
    convención de nombre "... (x, y, z).jpg" que ya se usa en el
    resto del proyecto.
    """

    carpeta = Path(carpeta)

    archivos = sorted(carpeta.glob(extension))

    tiles = []

    for archivo in archivos:

        try:
            x, y, z = extraer_coordenadas(archivo)
        except ValueError:
            print(f"Se salteó (sin coordenadas en el nombre): {archivo.name}")
            continue

        imagen = cargar_imagen(archivo)

        tiles.append({
            "archivo": archivo,
            "x": x,
            "y": y,
            "z": z,
            "imagen": imagen,
        })

    if len(tiles) == 0:
        raise ValueError(f"No se encontraron tiles válidos en {carpeta}")

    return tiles


# ============================================================
# CONSTRUIR EL MOSAICO
# ============================================================

def construir_mosaico(
    tiles,
    fov_x_mm,
    fov_y_mm,
    invertir_x=False,
    invertir_y=True,
    usar_feather=True,
):
    """
    Ubica cada tile en un canvas único usando su (x, y) real en mm
    (conocido de antemano, no hace falta feature matching ni ninguna
    librería de stitching) y funde las zonas de superposición con
    una ventana de Hann (feather blending), para que no se note el
    borde entre fotos vecinas.

    'fov_x_mm'/'fov_y_mm': campo visual real (en mm) que cubre CADA
    tile, en la resolución con la que se sacaron estas fotos. Si el
    campo visual que tenés medido (140 µm x 100 µm) corresponde a
    otra resolución de cámara, hay que reescalarlo antes de llamar a
    esta función — la cámara puede no tener el mismo campo visual
    real a QVGA que a WQXGA2, según cómo lea el sensor en cada modo.

    'invertir_x'/'invertir_y': si al mirar el mosaico se ve
    espejado o con la orientación cambiada respecto a lo que barrió
    la platina, la causa más común es que el eje de la imagen (fila
    y columna) no vaya en la misma dirección que el eje físico del
    motor. Probá primero con los valores por defecto; si dos tiles
    que en la vida real están pegados aparecen invertidos en el
    mosaico, cambiá el flag del eje correspondiente.

    'usar_feather': con True, funde solapamientos con una ventana de
    Hann (2D), que da más peso al centro de cada tile y menos peso a
    sus bordes — así, en las zonas donde se superponen dos tiles, el
    resultado es un promedio pesado en vez de que una tape a la
    otra, y no se ve una costura dura en el borde. Con False, cada
    tile nuevo pisa directamente a la anterior en la zona de
    superposición (más simple, pero puede notarse la costura si hay
    vignetting o diferencias de brillo entre tiles).
    """

    alto_px, ancho_px = tiles[0]["imagen"].shape[:2]

    for tile in tiles:
        if tile["imagen"].shape[:2] != (alto_px, ancho_px):
            raise ValueError(
                f"Los tiles no tienen todos el mismo tamaño de imagen "
                f"({tile['archivo'].name} mide distinto) — ¿se mezclaron "
                f"fotos de resoluciones distintas?"
            )

    escala_x = ancho_px / fov_x_mm  # píxeles por mm, eje x
    escala_y = alto_px / fov_y_mm   # píxeles por mm, eje y

    xs = [tile["x"] for tile in tiles]
    ys = [tile["y"] for tile in tiles]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    ancho_canvas = int(round((x_max - x_min) * escala_x)) + ancho_px
    alto_canvas = int(round((y_max - y_min) * escala_y)) + alto_px

    es_color = tiles[0]["imagen"].ndim == 3
    canales = tiles[0]["imagen"].shape[2] if es_color else 1

    acumulado = np.zeros((alto_canvas, ancho_canvas, canales), dtype=np.float64)
    pesos = np.zeros((alto_canvas, ancho_canvas, 1), dtype=np.float64)

    if usar_feather:
        ventana_y = np.hanning(alto_px)
        ventana_x = np.hanning(ancho_px)
        ventana_2d = np.outer(ventana_y, ventana_x)
        # Piso chico en vez de 0 puro: en las esquinas del mosaico,
        # donde no hay superposición con ningún otro tile, un peso
        # exactamente 0 en el borde dejaría esa franja en negro.
        ventana_2d = np.clip(ventana_2d, 1e-3, None)
    else:
        ventana_2d = np.ones((alto_px, ancho_px))

    peso_tile = ventana_2d[:, :, None]

    for tile in tiles:

        imagen = tile["imagen"].astype(np.float64)
        if imagen.ndim == 2:
            imagen = imagen[:, :, None]

        dx_mm = tile["x"] - x_min
        dy_mm = tile["y"] - y_min

        px_x = dx_mm * escala_x
        px_y = dy_mm * escala_y

        if invertir_x:
            px_x = ancho_canvas - ancho_px - px_x
        if invertir_y:
            px_y = alto_canvas - alto_px - px_y

        px_x = int(round(px_x))
        px_y = int(round(px_y))

        acumulado[px_y:px_y + alto_px, px_x:px_x + ancho_px, :] += imagen * peso_tile
        pesos[px_y:px_y + alto_px, px_x:px_x + ancho_px, :] += peso_tile

    pesos_seguros = np.where(pesos == 0, 1, pesos)
    mosaico = acumulado / pesos_seguros
    mosaico = np.clip(mosaico, 0, 255).astype(np.uint8)

    if not es_color:
        mosaico = mosaico[:, :, 0]

    return mosaico


# ============================================================
# GUARDAR + MOSTRAR
# ============================================================

def guardar_mosaico(mosaico, ruta_salida):

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(ruta_salida), mosaico)

    print(f"Mosaico guardado en:\n{ruta_salida}")


def mostrar_mosaico(mosaico, titulo="Mosaico"):

    fig, ax = plt.subplots(figsize=(10, 10))

    if mosaico.ndim == 2:
        ax.imshow(mosaico, cmap="gray")
    else:
        ax.imshow(cv2.cvtColor(mosaico, cv2.COLOR_BGR2RGB))

    ax.set_title(titulo)
    ax.axis("off")

    plt.show()


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":

    carpeta = r"C:\BackUp\Facultad\Labo 7\Fotos\26.09.29 mosaico prueba - copia - copia"

    tiles = cargar_tiles(carpeta)

    print(f"{len(tiles)} tiles cargados de {carpeta}")

    mosaico = construir_mosaico(
        tiles,
        fov_x_mm=0.214,  # campo visual COMPLETO en x, no el paso entre tiles
        fov_y_mm=0.286,  # campo visual COMPLETO en y, no el paso entre tiles
        invertir_x=False,
        invertir_y=False,
        usar_feather=True,
    )

    # guardar_mosaico(mosaico, Path(carpeta) / "mosaico.jpg")
    mostrar_mosaico(mosaico)



#%%


print(f"{1944*50/453} = ")
print(f"{2592*62.5/520} = ")


# %%


for fov_x_mm in [0.21, 0.2105, 0.211, 0.2115, 0.212, 0.2125, 0.213, 0.2135, 0.214, 0.2145, 0.215, 0.2155]:

    for fov_y_mm in [0.282, 0.2825, 0.283, 0.2835, 0.284, 0.2845, 0.285, 0.2855, 0.286, 0.2865, 0.287, 0.2875, 0.288, 0.2885, 0.289, 0.2895, 0.29, 0.2905]:

        carpeta = r"C:\BackUp\Facultad\Labo 7\Fotos\26.09.29 mosaico prueba - copia"
        
        tiles = cargar_tiles(carpeta)
    
        print(f"{len(tiles)} tiles cargados de {carpeta}")
    
        mosaico = construir_mosaico(
            tiles,
            fov_x_mm=fov_x_mm,  # campo visual COMPLETO en x, no el paso entre tiles
            fov_y_mm=fov_y_mm,  # campo visual COMPLETO en y, no el paso entre tiles
            invertir_x=False,
            invertir_y=False,
            usar_feather=True,
        )
    
        # guardar_mosaico(mosaico, Path(carpeta) / "mosaico.jpg")
        mostrar_mosaico(mosaico)
# %%
