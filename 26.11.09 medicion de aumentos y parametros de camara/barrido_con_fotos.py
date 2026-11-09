# %%

"""
Barrido en Z que guarda una foto por altura (aplicación 26.11.09 medicion de aumentos y parametros de camara).

Hace lo mismo que el descenso de buscar_foco_en_xy() (mismo viaje a (x, y), misma espera, la cámara
configurada UNA vez con un diccionario de parámetros), pero sin detectar nada: baja de z_arriba_seguro a
z_min_seguridad y guarda cada foto. Así se mide, con las fotos guardadas, qué aumentos ("x") y qué
parámetros de cámara le sirven a la detección (ver analisis_de_fotos.py).

Reemplaza a barrido_z_mas_guardar_imagenes() de 26.09.11 (vivía en "Oudated code/metricas_enfoque.py"). Diferencias:
  - la cámara se configura UNA vez por barrido, no en cada altura: con exposición o ganancia en
    automático, el sensor se va acomodando mientras baja, como en la búsqueda real;
  - los parámetros se pasan en un diccionario (REGLAS_CODIGO.txt, sección 8) y se revisan ANTES de mover;
  - respeta z_min_seguridad y el viaje a (x, y) pasa por moverse_a_xy();
  - guarda al lado de las fotos lo que la cámara informa de sí misma en cada una (info_camara.csv).

Los nombres de las fotos siguen siendo "<hora> (x, y, z).jpg", así que el análisis de 26.09.11 las entiende.
"""

import csv
import time
from datetime import datetime
from pathlib import Path

import impresora_cosmos
from busqueda_muestra import RiesgoDeColision
from camara_openmv import validar_dict_params_camara


NOMBRE_CSV_INFO_CAMARA = "info_camara.csv"
COLUMNAS_INFO_CAMARA = ["archivo", "z", "exposure_us", "gain_db", "rgb_r_db", "rgb_g_db", "rgb_b_db"]


def _como_texto(valor):
    return "auto" if valor is None else f"{valor:g}"


def nombre_de_carpeta_de_parametros(dict_params):
    """
    Nombre de la carpeta de un barrido, a partir de los parámetros de la cámara (una carpeta por combinación).

    Ej.: "Pixformat RGB565 - resolucion QVGA - Exposure 15000 - Gain auto - Whitebal 66.2224-61.4303-60.206".
    analisis_de_fotos.py arma las etiquetas de los gráficos con las partes que cambian entre carpetas.
    """

    validar_dict_params_camara(dict_params, "dict_params")

    pixformat = getattr(dict_params["pixformat"], "value", dict_params["pixformat"])
    resolucion = getattr(dict_params["resolucion"], "value", dict_params["resolucion"])
    whitebal = dict_params["whitebal"]

    texto_whitebal = "auto" if whitebal is None else "-".join(f"{canal:g}" for canal in whitebal)

    return (
        f"Pixformat {pixformat} - resolucion {resolucion} - "
        f"Exposure {_como_texto(dict_params['exposure_us'])} - "
        f"Gain {_como_texto(dict_params['gain_db'])} - "
        f"Whitebal {texto_whitebal}"
    )


def barrido_con_fotos(
    camara,
    impresora,
    x,
    y,
    z_arriba_seguro,
    z_min_seguridad,
    dict_params_camara,
    carpeta_guardado,
    paso_z=0.0025,
    z_seguro_xy=40.0,
    feedrate_xy=3000,
    espera_movimiento=0.5,
    guardar_info_camara=True,
    devolveme_la_respuesta=False,
):
    """
    Lleva la platina a (x, y), baja en Z de a 'paso_z' sacando una foto por altura y las guarda en 'carpeta_guardado'.

    Pasos:
    1) moverse_a_xy() a (x, y) con 'z_seguro_xy' como altura de viaje, y de ahí a 'z_arriba_seguro'
       (que tiene que estar LEJOS del foco: de ahí sale la línea de base de la detección).
    2) Configura la cámara UNA vez con 'dict_params_camara' y baja hasta 'z_min_seguridad' (inclusive):
       foto, guardado y, si 'guardar_info_camara', lo que informa la cámara (exposición, ganancia, balance de
       blancos: son los valores REALES, también con el modo automático).
    3) Vuelve a 'z_arriba_seguro' (la platina no queda cerca de la muestra).

    Se revisa todo ANTES de mover: diccionario de cámara, que la carpeta no tenga ya fotos (no se mezclan
    barridos), que z_min_seguridad < z_arriba_seguro <= z_seguro_xy y que la platina no esté ya por debajo de
    z_min_seguridad. Nunca manda una altura menor a z_min_seguridad (que ya tiene que traer el margen respecto
    de la distancia de trabajo del objetivo, 300 um); si pasara, levanta RiesgoDeColision.

    Si la cámara no contesta el info de una foto, el barrido sigue con esa fila vacía y avisa al final.

    Devuelve {"carpeta": Path, "archivos": [Path, ...], "z": [mm, ...], "fotos_sin_info": cantidad}.
    """

    validar_dict_params_camara(dict_params_camara, "dict_params_camara")

    if paso_z <= 0:
        raise ValueError(f"'paso_z' tiene que ser positivo, no {paso_z}")

    if not z_min_seguridad < z_arriba_seguro:
        raise ValueError(
            f"z_min_seguridad ({z_min_seguridad} mm) tiene que ser menor que z_arriba_seguro ({z_arriba_seguro} mm)"
        )

    if z_seguro_xy < z_arriba_seguro:
        raise ValueError(
            f"z_seguro_xy ({z_seguro_xy} mm, la altura de los viajes en XY) no puede ser menor que "
            f"z_arriba_seguro ({z_arriba_seguro} mm)"
        )

    if z_seguro_xy > impresora.max_posicion_z:
        raise ValueError(f"z_seguro_xy = {z_seguro_xy} mm pasa el máximo de la impresora ({impresora.max_posicion_z} mm)")

    carpeta = Path(carpeta_guardado)

    if carpeta.exists() and any(carpeta.glob("*.jpg")):
        raise FileExistsError(
            f"Ya hay fotos en {carpeta}: no se mezclan barridos. Cambiá la carpeta (por ejemplo, el número de pasada)."
        )

    _, _, z_actual = impresora.pregunta_posicion_actual()

    if z_actual is None:
        raise RuntimeError("La posición de la impresora no está inicializada. Llamá primero a impresora.home()")

    if z_actual < z_min_seguridad:
        raise RiesgoDeColision(
            f"La posición actual en Z ({z_actual} mm) ya está por debajo del límite de seguridad "
            f"({z_min_seguridad} mm). Se aborta antes de mover nada."
        )

    # --------------------------------------------------------
    # 1) Ir a (x, y) y a la altura desde la que empieza el barrido
    # --------------------------------------------------------

    impresora_cosmos.moverse_a_xy(
        impresora,
        x,
        y,
        z_seguro=z_seguro_xy,
        feedrate_xy=feedrate_xy,
        espera_movimiento=espera_movimiento,
        devolveme_la_respuesta=devolveme_la_respuesta,
    )

    # desp_lin_000() cancela SIN mover si el destino sale del rango: no bajar en otro lado como si nada.
    if (
        abs(impresora.posicion_x - x) > impresora.resolucion_x + 1e-9
        or abs(impresora.posicion_y - y) > impresora.resolucion_y + 1e-9
    ):
        raise RuntimeError(
            f"La platina no llegó a (x, y) = ({x}, {y}) mm: está en "
            f"({impresora.posicion_x}, {impresora.posicion_y}) mm. No se baja en Z."
        )

    _ir_a_z(impresora, z_arriba_seguro, z_min_seguridad, espera_movimiento, devolveme_la_respuesta)

    # --------------------------------------------------------
    # 2) Bajar sacando fotos
    # --------------------------------------------------------

    camara.config_de_una(dict_params_camara)

    carpeta.mkdir(parents=True, exist_ok=True)

    archivos = []
    alturas = []
    fotos_sin_info = 0
    z = impresora.posicion_z

    with open(carpeta / NOMBRE_CSV_INFO_CAMARA, "w", newline="", encoding="utf-8") as archivo_csv:

        escritor = csv.DictWriter(archivo_csv, fieldnames=COLUMNAS_INFO_CAMARA)
        escritor.writeheader()

        while True:

            image_data = camara.capturar()

            nombre = (
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                f"({float(impresora.posicion_x)}, {float(impresora.posicion_y)}, {float(impresora.posicion_z)})"
            )

            archivo = camara.guardar_imagen(image_data=image_data, carpeta=carpeta, nombre=nombre, mostrar=False)

            archivos.append(Path(archivo))
            alturas.append(float(impresora.posicion_z))

            if guardar_info_camara:

                fila = {"archivo": Path(archivo).name, "z": float(impresora.posicion_z)}

                try:
                    info = camara.info()
                    fila.update({
                        "exposure_us": info["exposure_us"],
                        "gain_db": info["gain_db"],
                        "rgb_r_db": info["rgb_gain_db"][0],
                        "rgb_g_db": info["rgb_gain_db"][1],
                        "rgb_b_db": info["rgb_gain_db"][2],
                    })
                except Exception as motivo:
                    fotos_sin_info += 1
                    print(f"  (la cámara no informó sus parámetros en Z = {impresora.posicion_z}: {motivo})")

                escritor.writerow(fila)
                archivo_csv.flush()

            if z - z_min_seguridad <= 1e-9:
                break

            paso = -min(paso_z, z - z_min_seguridad)

            impresora.desp_lin_cam(0, 0, paso, devolveme_la_respuesta=devolveme_la_respuesta)
            time.sleep(espera_movimiento)

            z = impresora.posicion_z

            if z < z_min_seguridad - 1e-6:
                raise RiesgoDeColision(
                    f"La platina quedó en Z = {z} mm, por debajo de z_min_seguridad ({z_min_seguridad} mm)."
                )

    # --------------------------------------------------------
    # 3) Volver arriba
    # --------------------------------------------------------

    _ir_a_z(impresora, z_arriba_seguro, z_min_seguridad, espera_movimiento, devolveme_la_respuesta)

    if fotos_sin_info:
        print(f"ATENCIÓN: la cámara no informó sus parámetros en {fotos_sin_info} de {len(archivos)} fotos.")

    return {
        "carpeta": carpeta,
        "archivos": archivos,
        "z": alturas,
        "fotos_sin_info": fotos_sin_info,
    }


def _ir_a_z(impresora, z_destino, z_min_seguridad, espera_movimiento, devolveme_la_respuesta):
    """Lleva la platina a z_destino (en el mismo x, y) con un desplazamiento relativo; nunca por debajo de z_min_seguridad."""

    if z_destino < z_min_seguridad:
        raise RiesgoDeColision(f"Z = {z_destino} mm está por debajo de z_min_seguridad ({z_min_seguridad} mm)")

    _, _, z_actual = impresora.pregunta_posicion_actual()

    if abs(z_destino - z_actual) > 1e-9:
        impresora.desp_lin_cam(0, 0, z_destino - z_actual, devolveme_la_respuesta=devolveme_la_respuesta)
        time.sleep(espera_movimiento)


# %%
