#%%



# Agrego la raiz del proyecto
import sys
import os

os.environ["PYTHONPATH"] = r"C:\BackUp\Facultad\Labo 7\env_labo7"

sys.path.insert(0, os.environ["PYTHONPATH"])

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


from metricas_enfoque import cargar_imagen, varianza_intensidad
from claude_busqueda_muestra  import fraccion_pixeles_rojos



#%%



# ============================================================
# RUTAS A LAS CARPETAS YA CLASIFICADAS A OJO
# ============================================================
# Ajustar acá si hace falta.

RUTAS_POR_TIPO = {
    "tipo_1": r"C:\BackUp\Facultad\Labo 7\Fotos\26.09.06 diferenciacion de los 3 tipos de campos visuales\Tipo 1",
    "tipo_2": r"C:\BackUp\Facultad\Labo 7\Fotos\26.09.06 diferenciacion de los 3 tipos de campos visuales\Tipo 2",
    "tipo_3": r"C:\BackUp\Facultad\Labo 7\Fotos\26.09.06 diferenciacion de los 3 tipos de campos visuales\Tipo 3",
}


# ============================================================
# MEDIR UNA CARPETA
# ============================================================

def medir_carpeta(carpeta, s_min=50, v_min=50):
    """
    Devuelve, para cada imagen de la carpeta, la fracción de rojo
    y la varianza de intensidad.
    """

    carpeta = Path(carpeta)

    archivos = sorted(carpeta.glob("*.jpg"))

    if len(archivos) == 0:
        raise ValueError(
            f"No se encontraron imágenes JPG en:\n{carpeta}"
        )

    fracciones_rojo = []
    varianzas = []
    nombres = []

    for archivo in archivos:

        try:
            imagen = cargar_imagen(archivo)

            fraccion = fraccion_pixeles_rojos(
                imagen, s_min=s_min, v_min=v_min
            )

            varianza = varianza_intensidad(imagen)

            fracciones_rojo.append(fraccion)
            varianzas.append(varianza)
            nombres.append(archivo.name)

        except Exception as e:

            print(f"No se pudo procesar {archivo.name}: {e}")

    return {
        "nombres": nombres,
        "fraccion_rojo": np.array(fracciones_rojo),
        "varianza": np.array(varianzas),
    }


# ============================================================
# REPORTE DE ESTADÍSTICAS POR TIPO
# ============================================================

def reportar_estadisticas(mediciones_por_tipo):

    print("\n" + "=" * 70)
    print("ESTADÍSTICAS POR TIPO")
    print("=" * 70)

    for tipo, datos in mediciones_por_tipo.items():

        fr = datos["fraccion_rojo"]
        var = datos["varianza"]

        print(f"\n{tipo}  (n = {len(fr)})")

        print(
            f"  Fracción de rojo   -> "
            f"min = {fr.min():.5f} | "
            f"max = {fr.max():.5f} | "
            f"media = {fr.mean():.5f} | "
            f"std = {fr.std():.5f}"
        )

        print(
            f"  Varianza intensidad -> "
            f"min = {var.min():.3f} | "
            f"max = {var.max():.3f} | "
            f"media = {var.mean():.3f} | "
            f"std = {var.std():.3f}"
        )


# ============================================================
# SUGERIR UMBRAL: TIPO 1 (SIN MUESTRA) vs TIPO 2+3 (CON MUESTRA)
# ============================================================

def sugerir_umbral(mediciones_por_tipo, clave="fraccion_rojo"):
    """
    Junta tipo_2 y tipo_3 como "contiene muestra" (para esta búsqueda
    no interesa diferenciarlos) y compara contra tipo_1 ("sin muestra"
    o demasiado lejos del plano ideal).

    Sugiere el punto medio entre el máximo de tipo_1 y el mínimo de
    tipo_2+tipo_3, igual que hiciste vos a mano con la varianza.
    """

    sin_muestra = mediciones_por_tipo["tipo_1"][clave]

    con_muestra = np.concatenate([
        mediciones_por_tipo["tipo_2"][clave],
        mediciones_por_tipo["tipo_3"][clave],
    ])

    max_sin_muestra = sin_muestra.max()
    min_con_muestra = con_muestra.min()

    print(f"\n--- Separación usando '{clave}' ---")
    print(f"Máximo en tipo_1 (sin muestra):        {max_sin_muestra:.5f}")
    print(f"Mínimo en tipo_2 + tipo_3 (con muestra): {min_con_muestra:.5f}")

    if max_sin_muestra < min_con_muestra:

        umbral_sugerido = (max_sin_muestra + min_con_muestra) / 2

        print(f"¡Separación limpia! Umbral sugerido = {umbral_sugerido:.5f}")

    else:

        print(
            "ATENCIÓN: hay superposición entre los grupos con esta métrica "
            "sola (el máximo de tipo_1 es mayor al mínimo de tipo_2+3). "
            "Puede que haga falta combinar fracción de rojo + varianza, "
            "o ajustar s_min/v_min de fraccion_pixeles_rojos()."
        )

        umbral_sugerido = None

    return umbral_sugerido


# ============================================================
# GRAFICAR LA DISTRIBUCIÓN (fracción de rojo vs varianza)
# ============================================================

def graficar_dispersión(mediciones_por_tipo):

    fig, ax = plt.subplots()

    colores = {"tipo_1": "gray", "tipo_2": "orange", "tipo_3": "red"}

    for tipo, datos in mediciones_por_tipo.items():

        ax.scatter(
            datos["varianza"],
            datos["fraccion_rojo"],
            label=tipo,
            color=colores.get(tipo),
            alpha=0.7,
        )

    ax.set_xlabel("Varianza de intensidad")
    ax.set_ylabel("Fracción de píxeles rojos")
    ax.set_title("Separación entre tipos de campo visual")
    ax.legend()
    ax.grid()

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    mediciones_por_tipo = {
        tipo: medir_carpeta(carpeta)
        for tipo, carpeta in RUTAS_POR_TIPO.items()
    }

    reportar_estadisticas(mediciones_por_tipo)

    sugerir_umbral(mediciones_por_tipo, clave="fraccion_rojo")
    sugerir_umbral(mediciones_por_tipo, clave="varianza")

    graficar_dispersión(mediciones_por_tipo)


#%%
