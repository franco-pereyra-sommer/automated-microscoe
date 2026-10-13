#%%


import time
from pathlib import Path
from datetime import datetime

# from metricas_enfoque import varianza_intensidad
# from busqueda_muestra import posicionar_en_muestra, FUNCIONES_METRICA
from busqueda_muestra import *


# ============================================================
# GENERADOR DE OFFSETS EN ESPIRAL CUADRADA
# ============================================================

def generar_offsets_espiral(pasos_radiales):
    """
    Genera offsets (i, j) enteros que recorren, en orden de espiral
    (empezando en el centro y expandiéndose hacia afuera), una
    grilla cuadrada de lado (2*pasos_radiales + 1) centrada en
    (0, 0).

    'pasos_radiales' es el tope del barrido: cuántos "anillos"
    alrededor del punto inicial se recorren como máximo. Con
    pasos_radiales=2, por ejemplo, se recorren 25 posiciones (una
    grilla de 5x5 centrada en el punto inicial).

    Cada offset (i, j) se multiplica después por el paso en x/y
    (en mm) para obtener el desplazamiento real. Se devuelven en el
    orden en que la espiral los visita, así que si en algún momento
    se quiere cortar el barrido antes de tiempo (por ejemplo, por un
    error de cámara), lo ya capturado sigue siendo una espiral
    completa y simétrica hasta ese anillo.
    """

    if pasos_radiales < 0:
        raise ValueError("pasos_radiales tiene que ser >= 0")

    offsets = [(0, 0)]

    if pasos_radiales == 0:
        return offsets

    i, j = 0, 0
    paso = 1

    # Se generan un par de vueltas de más (margen) y se recorta con
    # el filtro de abajo, en vez de calcular a mano en qué punto
    # exacto de la vuelta termina el último anillo.
    while paso <= 2 * pasos_radiales + 2:

        for _ in range(paso):  # derecha
            i += 1
            if max(abs(i), abs(j)) <= pasos_radiales:
                offsets.append((i, j))

        for _ in range(paso):  # arriba
            j += 1
            if max(abs(i), abs(j)) <= pasos_radiales:
                offsets.append((i, j))

        paso += 1

        for _ in range(paso):  # izquierda
            i -= 1
            if max(abs(i), abs(j)) <= pasos_radiales:
                offsets.append((i, j))

        for _ in range(paso):  # abajo
            j -= 1
            if max(abs(i), abs(j)) <= pasos_radiales:
                offsets.append((i, j))

        paso += 1

    return offsets


# ============================================================
# ORQUESTADOR: HOME -> POSICIONAR EN MUESTRA -> BARRIDO ESPIRAL
# ============================================================

def barrido_espiral_muestra(
    camara,
    impresora,
    x_ini,
    y_ini,
    z_arriba_seguro,
    z_min_seguridad,
    carpeta_salida,
    pasos_radiales_maximo=3,
    paso_fino = 0.0025,
    paso_x_mm=0.05,
    paso_y_mm=0.075,
    resolucion_busqueda="QVGA",
    resolucion_alta="WQXGA2",
    pixformat_busqueda="GRAYSCALE",
    pixformat_alta="RGB565",
    metrica_foco_final=varianza_intensidad,
    pasos_arriba_foco_local=3,
    pasos_abajo_foco_local=3,
    guardar_fotos_autofoco_local=True,
    feedrate_xy=3000,
    espera_movimiento=0.3,
    espera_movimiento_alta=0.5,
    devolveme_la_respuesta=False,
    kwargs_posicionar=None,
):
    """
    Secuencia completa:

    1) impresora.home()
    2) posicionar_en_muestra() en (x_ini, y_ini): verificación +
       barrido grueso + confirmación + foco final, todo con
       'resolucion_busqueda' (QVGA por defecto) y la métrica
       'metrica_foco_final' (varianza de intensidad por defecto).
       Si 'guardar_fotos_autofoco' es True, las fotos de ese foco
       final quedan en 'carpeta_salida/(x, y, z)/'.
    3) Ya con el plano de foco encontrado, se cambia a
       'resolucion_alta' (WQXGA2 por defecto — la resolución más
       alta que soporta la cámara) y se hace un barrido en espiral
       en XY alrededor de (x_ini, y_ini), manteniendo Z fijo en el
       plano de foco. El paso en cada eje es 'paso_x_mm'/'paso_y_mm'
       (por defecto, la mitad del campo visual real en cada eje —
       100 µm en x / 140 µm en y de campo visual completo → 0.05 mm
       / 0.07 mm de paso — para que campos visuales consecutivos se
       superpongan al 50%). 'pasos_radiales_maximo' pone el tope de
       anillos del barrido (ver generar_offsets_espiral()).

    OJO: 'paso_x_mm'/'paso_y_mm' es el PASO entre tiles consecutivos
    (la mitad del campo visual, para el 50% de superposición), no el
    campo visual completo — ese valor completo (0.10 mm / 0.14 mm)
    es el que después hace falta pasarle a construir_mosaico(), en
    reconstruir_mosaico.py, para calcular la escala de píxeles por mm.

    Cada foto del barrido espiral se guarda directamente en
    'carpeta_salida', con nombre:

        {índice} i{i}_j{j} ({x}, {y}, {z}).jpg

    El "(x, y, z)" del nombre es el que después usa
    metricas_enfoque.extraer_coordenadas() para reconstruir el
    mosaico (ver reconstruir_mosaico.py) — no cambiar ese formato
    sin actualizar también esa función.

    Devuelve un diccionario con el resultado de posicionar_en_muestra()
    más la lista de tiles del barrido espiral (uno por posición, con
    su (x, y, z) real, sus índices (i, j) de grilla y la ruta del
    archivo guardado).

    OJO: esta función asume que el plano de foco no cambia de forma
    apreciable dentro del rango barrido (no vuelve a enfocar en cada
    posición). Si el barrido es muy grande y la muestra no es
    perfectamente plana, capaz haga falta reenfocar cada tantas
    posiciones — no está implementado acá.
    """

    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    if kwargs_posicionar is None:
        kwargs_posicionar = {}

    if isinstance(metrica_foco_final, str):
        metrica_foco_final_funcion = FUNCIONES_METRICA[metrica_foco_final]
    else:
        metrica_foco_final_funcion = metrica_foco_final

    # --------------------------------------------------------
    # 1) Home
    # --------------------------------------------------------

    impresora.home(devolveme_la_respuesta=devolveme_la_respuesta)
    time.sleep(espera_movimiento)

    # --------------------------------------------------------
    # 2) Posicionar en la muestra + foco final
    # --------------------------------------------------------

    resultado_posicionamiento = posicionar_en_muestra(
        camara=camara,
        impresora=impresora,
        x=x_ini,
        y=y_ini,
        z_arriba_seguro=z_arriba_seguro,
        z_min_seguridad=z_min_seguridad,
        resolucion_global=resolucion_busqueda,
        resolucion_local=resolucion_busqueda,
        pixformat_global=pixformat_busqueda,
        hacer_foco_final=False,
        metrica_foco_final=metrica_foco_final_funcion,
        pasos_arriba_foco_final=pasos_arriba_foco_local,
        pasos_abajo_foco_final=pasos_abajo_foco_local,
        guardar_fotos_foco_final=guardar_fotos_autofoco_local,
        carpeta_fotos_foco_final=carpeta_salida,
        feedrate_xy=feedrate_xy,
        espera_movimiento=espera_movimiento,
        devolveme_la_respuesta=devolveme_la_respuesta,
        **kwargs_posicionar,
    )

    x_foco = resultado_posicionamiento["x"]
    y_foco = resultado_posicionamiento["y"]

    z_foco = resultado_posicionamiento["z_foco_final"]
    if z_foco is None:
        z_foco = resultado_posicionamiento["z_confirmado"]

    print(
        f"\nPlano de foco encontrado en "
        f"({x_foco}, {y_foco}, {z_foco:.4f}). "
        f"Arranca el barrido espiral."
    )

    # --------------------------------------------------------
    # 3) Cambiar a resolución alta y barrer en espiral
    # --------------------------------------------------------

    # if pixformat_alta is not None:
    #     camara.set_pixformat(pixformat_alta)

    # camara.set_framesize(resolucion_alta)
    # camara.descartar_frames()  # darle tiempo al sensor a reacomodarse


    metrica_final = (
        metrica_foco_final
        or FUNCIONES_METRICA[resultado_posicionamiento["metrica_confirmada"]]
    )

    z_confirmado = resultado_posicionamiento["z_confirmado"]

    margen_disponible_abajo = z_confirmado - z_min_seguridad
    pasos_abajo_reales = max(
        min(pasos_abajo_foco_local, int(margen_disponible_abajo // paso_fino)),
        0,
    )

    offsets = generar_offsets_espiral(pasos_radiales_maximo)

    tiles = []

    for indice, (i, j) in enumerate(offsets):

        camara.config_autofoco_local()

        carpeta_guardado_autofoco_local = (
            rf"{carpeta_salida}"
            rf"\{indice:04d} i{i}_j{j}"
        )

        mejor_z, posiciones_z_final, valores_final = autofocus(
            camara=camara,
            impresora=impresora,
            metrica=metrica_final,
            pasos_arriba=pasos_arriba_foco_local,
            pasos_abajo=pasos_abajo_reales,
            paso_z=paso_fino,
            espera_movimiento=espera_movimiento,
            graficamos=False,
            guardamos=guardar_fotos_autofoco_local,
            carpeta_guardado=carpeta_guardado_autofoco_local
        )

        x_objetivo = x_foco + i * paso_x_mm
        y_objetivo = y_foco + j * paso_y_mm

        impresora.desp_lin_abs(
            x=x_objetivo, y=y_objetivo, z=mejor_z,
            feedrate=feedrate_xy,
            devolveme_la_respuesta=devolveme_la_respuesta,
        )
        time.sleep(espera_movimiento_alta)

        camara.config_de_una(
            pixformat=pixformat_alta,
            resolucion=resolucion_alta
        )

        image_data = camara.capturar()

        hora_actual = datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S"
                )

        nombre = (
            f"{indice:04d} i{i}_j{j} "
            f"({impresora.posicion_x}, "
            f"{impresora.posicion_y}, "
            f"{impresora.posicion_z})"
            f" - {hora_actual}"
        )

        archivo = camara.guardar_imagen(
            image_data=image_data,
            carpeta=carpeta_salida,
            nombre=nombre,
            mostrar=False,
        )

        print(
            f"[{indice + 1}/{len(offsets)}] "
            f"(i={i}, j={j}) -> "
            f"({impresora.posicion_x}, {impresora.posicion_y}, {impresora.posicion_z})"
        )

        tiles.append({
            "indice": indice,
            "i": i,
            "j": j,
            "x": impresora.posicion_x,
            "y": impresora.posicion_y,
            "z": impresora.posicion_z,
            "archivo": archivo,
        })

    resultado_posicionamiento["tiles_espiral"] = tiles
    resultado_posicionamiento["carpeta_salida"] = carpeta_salida

    return resultado_posicionamiento


# ============================================================
# EJEMPLO DE USO
# ============================================================

# resultado = barrido_espiral_muestra(
#     camara=cam,
#     impresora=grua,
#     x_ini=150.0,
#     y_ini=100.0,
#     z_arriba_seguro=30.0,
#     z_min_seguridad=42.0,
#     carpeta_salida=r"C:\Fotos\26.09.27 mosaico prueba",
#     pasos_radiales_maximo=3,
# )
#
# print(f"{len(resultado['tiles_espiral'])} fotos guardadas en "
#       f"{resultado['carpeta_salida']}")


#%%
