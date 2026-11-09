#%%
# Análisis de una medición de 26.11.09 (correr_medicion.py): métricas de cada foto, gráficos y tabla resumen.
# Se corre celda por celda. Cada celda de abajo es independiente de las otras salvo la 1 (lee las fotos).

import os
import sys

import matplotlib.pyplot as plt

RAIZ = r"C:\BackUp\Facultad\Labo 7\env_labo7"
CARPETA_CODIGO = RAIZ + r"\26.11.09 medicion de aumentos y parametros de camara"

os.environ["PYTHONPATH"] = RAIZ
for ruta in (RAIZ, CARPETA_CODIGO):
    if ruta not in sys.path:
        sys.path.insert(0, ruta)

from analisis_de_fotos import (
    analizar_medicion,
    exportar_csv,
    graficar_metricas,
    graficar_relativo_a_la_linea_de_base,
    imprimir_resumen,
)


#%% PARÁMETROS

# La carpeta de UN (x, y) de UNA pasada: la que armó correr_medicion.py, con una subcarpeta por combinación de cámara
CARPETA_MEDICION = (
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.11.09 medicion de aumentos y parametros de camara"
    r"\muestra 1 xy 150_150 - pasada 1"
)

PASOS_LINEA_BASE = 5            # fotos que promedia la línea de base (los de buscar_foco_en_xy)
PASO_GRUESO_MM = 0.005          # el paso del descenso de buscar_foco_en_xy: de las fotos de 2.5 µm se saltea una de cada dos
AUMENTOS_CANDIDATOS = [0.1, 0.3, 1.0]   # líneas punteadas del gráfico relativo (1 + aumento)


#%% 1) MÉTRICAS DE TODAS LAS FOTOS (lee cada foto: tarda)

resultados_por_etiqueta = analizar_medicion(CARPETA_MEDICION)


#%% 2) MÉTRICAS Y PARÁMETROS DE LA CÁMARA VS Z

graficar_metricas(resultados_por_etiqueta, titulo=os.path.basename(CARPETA_MEDICION))
plt.show()


#%% 3) CADA MÉTRICA DIVIDIDA POR SU LÍNEA DE BASE (de acá se leen los x)

graficar_relativo_a_la_linea_de_base(
    resultados_por_etiqueta,
    aumentos_candidatos=AUMENTOS_CANDIDATOS,
    pasos_linea_base=PASOS_LINEA_BASE,
    paso_grueso=PASO_GRUESO_MM,
)
plt.show()


#%% 4) TABLA: LÍNEA DE BASE, RUIDO Y CUÁNTO SUBE EN EL FOCO

imprimir_resumen(resultados_por_etiqueta, pasos_linea_base=PASOS_LINEA_BASE, paso_grueso=PASO_GRUESO_MM)


#%% 5) CSV (opcional, para mirar los números en detalle)

# for etiqueta, resultados in resultados_por_etiqueta.items():
#     exportar_csv(resultados, os.path.join(CARPETA_MEDICION, f"analisis {etiqueta}.csv"))


#%%
