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
import pandas as pd

import time
from datetime import datetime

from metricas_enfoque import *



#%%



def obtener_metricas_carpeta(carpeta, tipo):
    
    datos = []
    
    for nombre in os.listdir(carpeta):
        
        ruta = os.path.join(carpeta, nombre)
        
        if not os.path.isfile(ruta):
            continue
        
        imagen = cv2.imread(ruta)
        
        if imagen is None:
            continue
        
        datos.append({
            "archivo": nombre,
            "tipo": tipo,
            "S_intensidad": intensidad_fondo(imagen),
            "S_bordes": tenengrad(imagen),
            "S_contraste": varianza_intensidad(imagen)
        })
    
    return datos


def cargar_datos_campos(carpeta_base):
    
    datos = []
    
    for tipo in [1, 2, 3]:
        
        carpeta = os.path.join(
            carpeta_base,
            f"Tipo {tipo}"
        )
        
        datos.extend(
            obtener_metricas_carpeta(
                carpeta,
                tipo
            )
        )
    
    return pd.DataFrame(datos)



#%%



carpeta_base = r"C:\BackUp\Facultad\Labo 7\Fotos\26.09.06 diferenciacion de los 3 tipos de campos visuales"

df = cargar_datos_campos(carpeta_base)

display(df)

#%%


# display(
#     df.groupby("tipo")[
#         ["S_intensidad", "S_bordes", "S_contraste"]
#     ].describe()
#     )

display(
    df.groupby("tipo")[
        ["S_contraste"]
    ].describe()
    )



#%% Estos graficos estan feos



import matplotlib.pyplot as plt


# ============================================================
# S_intensidad
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

for tipo in sorted(df["tipo"].unique()):
    
    datos = df[df["tipo"] == tipo]["S_intensidad"]
    
    ax.scatter(
        np.full(len(datos), tipo),
        datos,
        alpha=0.7,
        label=f"Tipo {tipo}"
    )

ax.set_xlabel("Tipo de campo visual")
ax.set_ylabel(r"$S_{\mathrm{intensidad}}$")
ax.set_title("Intensidad de las imágenes")
ax.set_xticks([1, 2, 3])
ax.grid(alpha=0.3)
ax.legend()

plt.show()


# ============================================================
# S_bordes (Tenengrad)
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

for tipo in sorted(df["tipo"].unique()):
    
    datos = df[df["tipo"] == tipo]["S_bordes"]
    
    ax.scatter(
        np.full(len(datos), tipo),
        datos,
        alpha=0.7,
        label=f"Tipo {tipo}"
    )

ax.set_xlabel("Tipo de campo visual")
ax.set_ylabel(r"$S_{\mathrm{bordes}}$ (Tenengrad)")
ax.set_title("Métrica de bordes — Tenengrad")
ax.set_xticks([1, 2, 3])
ax.grid(alpha=0.3)
ax.legend()

plt.show()


# ============================================================
# S_contraste
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

for tipo in sorted(df["tipo"].unique()):
    
    datos = df[df["tipo"] == tipo]["S_contraste"]
    
    ax.scatter(
        np.full(len(datos), tipo),
        datos,
        alpha=0.7,
        label=f"Tipo {tipo}"
    )

ax.set_xlabel("Tipo de campo visual")
ax.set_ylabel(r"$S_{\mathrm{contraste}}$")
ax.set_title("Varianza de intensidad")
ax.set_xticks([1, 2, 3])
ax.grid(alpha=0.3)
ax.legend()

plt.show()
# %%



fig, ax = plt.subplots(figsize=(8, 6))

datos = [
    df[df["tipo"] == tipo]["S_intensidad"].values
    for tipo in [1, 2, 3]
]

ax.boxplot(
    datos
)

ax.set_xlabel("Tipo de campo visual")
ax.set_ylabel(r"$S_{\mathrm{intensidad}}$")
ax.set_title("Intensidad de las imágenes")
ax.grid(axis="y", alpha=0.3)

plt.show()



fig, ax = plt.subplots(figsize=(8, 6))

datos = [
    df[df["tipo"] == tipo]["S_bordes"].values
    for tipo in [1, 2, 3]
]

ax.boxplot(
    datos
)

ax.set_xlabel("Tipo de campo visual")
ax.set_ylabel(r"$S_{\mathrm{bordes}}$ (Tenengrad)")
ax.set_title("Métrica de bordes — Tenengrad")
ax.grid(axis="y", alpha=0.3)

plt.show()



fig, ax = plt.subplots(figsize=(8, 6))

datos = [
    df[df["tipo"] == tipo]["S_contraste"].values
    for tipo in [1, 2, 3]
]

ax.boxplot(
    datos
)

ax.set_xlabel("Tipo de campo visual")
ax.set_ylabel(r"$S_{\mathrm{contraste}}$")
ax.set_title("Varianza de intensidad")
ax.grid(axis="y", alpha=0.3)

plt.show()



# %%
