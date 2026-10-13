#%%



# Agrego la raiz del proyecto
import sys
import os

os.environ["PYTHONPATH"] = r"C:\BackUp\Facultad\Labo 7\env_labo7"

sys.path.insert(0, os.environ["PYTHONPATH"])

import time
from pathlib import Path
from datetime import datetime

import numpy as np

# from metricas_enfoque import varianza_intensidad
# from busqueda_muestra import posicionar_en_muestra, FUNCIONES_METRICA
from busqueda_muestra import *

from camara_openmv import OpenMVCamera
from impresora_cosmos import TrimakerCOSMOS


# ============================================================
# AJUSTE A LA GRILLA DE LA IMPRESORA (XY)
# ============================================================
# La impresora solo puede moverse en múltiplos de 0.0125 mm en x-y.
# TrimakerCOSMOS.ajustar_movimiento() redondea HACIA ABAJO
# (np.floor(valor / resolucion)), y por ruido de punto flotante un
# valor que "debería" ser múltiplo exacto (ej: 149.95) puede dar
# 11999.999999999998 y caer un paso entero (12.5 µm) más abajo.
# Para que eso no pase, todas las coordenadas XY del barrido se
# pasan primero por _a_grilla() y se les suma un epsilon de 1 nm
# (_EPS_MM): no cambia nada si algún día ajustar_movimiento() pasa
# a usar np.round, pero evita el escalón espurio con floor.

PASO_MINIMO_XY_MM = 0.0125
# _EPS_MM = 1e-6
_EPS_MM = 0


def _a_grilla(valor_mm, paso_minimo=PASO_MINIMO_XY_MM):
    """Redondea 'valor_mm' al múltiplo más cercano de 'paso_minimo'."""
    return float(np.round(np.round(valor_mm / paso_minimo) * paso_minimo, 4))


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
    paso_grueso=0.01,
    paso_fino=0.0025,
    paso_x_mm=0.05,
    paso_y_mm=0.0625,
    resolucion_busqueda="QVGA",
    resolucion_alta="WQXGA2",
    # pixformat_busqueda=OpenMVCamera.GRAYSCALE,
    # pixformat_alta=OpenMVCamera.RGB565,
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
       barrido grueso + confirmación, con 'resolucion_busqueda'
       (QVGA por defecto). NO hace el foco final (hacer_foco_final=
       False): el foco fino se hace después, en cada tile.
    3) Barrido en espiral en XY alrededor de (x_ini, y_ini). Para
       CADA tile, en este orden:
         a) moverse en XY a la posición del tile (Z queda donde
            estaba, que es el mejor Z del tile anterior),
         b) camara.config_autofoco_local() y autofocus() ahí mismo
            (el foco se pierde rápido al moverse, así que hay que
            enfocar EN el tile nuevo, no en el anterior). Deja la
            platina en el mejor Z encontrado,
         c) camara.config_de_una() con pixformat_alta/resolucion_alta
            y captura de la foto de calidad alta,
         d) se guarda la foto en 'carpeta_salida'.
       Las fotos de cada autofoco (si guardar_fotos_autofoco_local)
       quedan en la subcarpeta 'carpeta_salida/{índice} i{i}_j{j}/'.

    Pasos XY: 'paso_x_mm'/'paso_y_mm' es el PASO entre tiles
    consecutivos (no el campo visual completo). Como la impresora
    solo se mueve en múltiplos de 0.0125 mm, ambos pasos (y x_ini/
    y_ini) se redondean al múltiplo más cercano (se avisa por
    pantalla si eso los cambia). Por eso el default en y es 0.0625
    (5 pasos, ≈45% de un campo visual de 0.14 mm) y no 0.07 (que
    caería a 0.0625 o 0.075 según la posición). El campo visual
    completo (ej: 0.10 mm / 0.14 mm) es el que después hace falta
    pasarle a construir_mosaico(), en reconstruir_mosaico.py.

    Seguridad en Z: 'pasos_abajo' de cada autofoco se acota en cada
    tile según el Z ACTUAL y 'z_min_seguridad' (no una sola vez al
    principio), porque el Z va derivando de un tile a otro. Se
    cumple z_min_seguridad < z_arriba_seguro (mayor Z = más lejos de
    la muestra = más seguro).

    Cada foto del barrido espiral se guarda directamente en
    'carpeta_salida', con nombre:

        {índice} i{i}_j{j} ({x}, {y}, {z}) - {fecha_hora}.jpg

    El "(x, y, z)" del nombre es el que después usa
    metricas_enfoque.extraer_coordenadas() para reconstruir el
    mosaico (ver reconstruir_mosaico.py) — no cambiar ese formato
    sin actualizar también esa función.

    Devuelve un diccionario con el resultado de posicionar_en_muestra()
    más la lista de tiles del barrido espiral (uno por posición, con
    su (x, y, z) real, sus índices (i, j) de grilla, la ruta del
    archivo guardado y si el autofoco quedó en un borde de su rango).
    """

    if z_min_seguridad >= z_arriba_seguro:
        raise ValueError(
            "z_min_seguridad tiene que ser menor que z_arriba_seguro"
        )

    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    if kwargs_posicionar is None:
        kwargs_posicionar = {}

    if isinstance(metrica_foco_final, str):
        metrica_foco_final_funcion = FUNCIONES_METRICA[metrica_foco_final]
    else:
        metrica_foco_final_funcion = metrica_foco_final

    # --------------------------------------------------------
    # Pasos y punto inicial sobre la grilla de la impresora
    # --------------------------------------------------------

    paso_x_grilla = _a_grilla(paso_x_mm)
    paso_y_grilla = _a_grilla(paso_y_mm)

    if paso_x_grilla <= 0 or paso_y_grilla <= 0:
        raise ValueError(
            f"paso_x_mm/paso_y_mm tienen que ser >= {PASO_MINIMO_XY_MM} mm"
        )

    if paso_x_grilla != paso_x_mm or paso_y_grilla != paso_y_mm:
        print(
            f"[AVISO] Los pasos se ajustaron a múltiplos de "
            f"{PASO_MINIMO_XY_MM} mm: "
            f"x {paso_x_mm} -> {paso_x_grilla} | "
            f"y {paso_y_mm} -> {paso_y_grilla}"
        )

    x_ini_grilla = _a_grilla(x_ini)
    y_ini_grilla = _a_grilla(y_ini)

    if x_ini_grilla != x_ini or y_ini_grilla != y_ini:
        print(
            f"[AVISO] El punto inicial se ajustó a la grilla de la "
            f"impresora: ({x_ini}, {y_ini}) -> "
            f"({x_ini_grilla}, {y_ini_grilla})"
        )

    # --------------------------------------------------------
    # 1) Home
    # --------------------------------------------------------

    impresora.home(devolveme_la_respuesta=devolveme_la_respuesta)
    time.sleep(espera_movimiento)

    # --------------------------------------------------------
    # 2) Posicionar en la muestra (sin foco final)
    # --------------------------------------------------------

    resultado_posicionamiento = posicionar_en_muestra(
        camara=camara,
        impresora=impresora,
        x=x_ini_grilla + _EPS_MM,
        y=y_ini_grilla + _EPS_MM,
        z_arriba_seguro=z_arriba_seguro,
        z_min_seguridad=z_min_seguridad,
        resolucion_global=resolucion_busqueda,
        resolucion_local=resolucion_busqueda,
        pixformat_global=pixformat_busqueda,
        hacer_foco_final=False,
        feedrate_xy=feedrate_xy,
        espera_movimiento=espera_movimiento,
        devolveme_la_respuesta=devolveme_la_respuesta,
        paso_grueso=paso_grueso,
        **kwargs_posicionar,
    )

    x_foco = x_ini_grilla
    y_foco = y_ini_grilla

    z_foco = resultado_posicionamiento["z_confirmado"]

    print(
        f"\nMuestra confirmada en "
        f"({x_foco}, {y_foco}, {z_foco:.4f}). "
        f"Arranca el barrido espiral."
    )

    # --------------------------------------------------------
    # 3) Barrido espiral: mover -> autofoco -> foto en alta
    # --------------------------------------------------------

    offsets = generar_offsets_espiral(pasos_radiales_maximo)

    tiles = []

    for indice, (i, j) in enumerate(offsets):

        # ---- a) mover en XY (Z queda donde está) ------------------

        x_objetivo = _a_grilla(x_foco + i * paso_x_grilla)
        y_objetivo = _a_grilla(y_foco + j * paso_y_grilla)

        _, _, z_actual = impresora.pregunta_posicion_actual()

        impresora.desp_lin_abs(
            x=x_objetivo + _EPS_MM,
            y=y_objetivo + _EPS_MM,
            z=z_actual,
            feedrate=feedrate_xy,
            devolveme_la_respuesta=devolveme_la_respuesta,
        )
        time.sleep(espera_movimiento_alta)

        # Chequeo: la posición que registró la impresora tiene que
        # coincidir con la pedida (si no, algún redondeo se comió un
        # paso de 12.5 µm y el mosaico va a salir corrido).
        if (
            abs(impresora.posicion_x - x_objetivo) > PASO_MINIMO_XY_MM / 2
            or abs(impresora.posicion_y - y_objetivo) > PASO_MINIMO_XY_MM / 2
        ):
            print(
                f"[AVISO] Tile {indice}: se pidió "
                f"({x_objetivo}, {y_objetivo}) pero la impresora quedó en "
                f"({impresora.posicion_x}, {impresora.posicion_y})"
            )

        # ---- b) autofoco EN el tile nuevo -----------------------

        camara.config_autofoco_local()

        _, _, z_antes_autofoco = impresora.pregunta_posicion_actual()

        margen_disponible_abajo = z_antes_autofoco - z_min_seguridad

        pasos_abajo_reales = max(
            min(
                pasos_abajo_foco_local,
                int(margen_disponible_abajo // paso_fino),
            ),
            0,
        )

        carpeta_autofoco_tile = carpeta_salida / f"{indice:04d} i{i}_j{j}"

        mejor_z, posiciones_z_final, valores_final = autofocus(
            camara=camara,
            impresora=impresora,
            metrica=metrica_foco_final_funcion,
            pasos_arriba=pasos_arriba_foco_local,
            pasos_abajo=pasos_abajo_reales,
            paso_z=paso_fino,
            espera_movimiento=espera_movimiento,
            graficamos=False,
            guardamos=guardar_fotos_autofoco_local,
            carpeta_guardado=carpeta_autofoco_tile,
        )

        # Si el máximo cae en un extremo del rango barrido, el foco
        # real probablemente está más allá: el próximo tile arranca
        # de este Z y el error se va acumulando.
        indice_mejor = int(np.argmax(valores_final))
        en_borde = indice_mejor in (0, len(valores_final) - 1)

        if en_borde:
            print(
                f"[AVISO] Tile {indice}: el autofoco quedó en el borde de "
                f"su rango (Z = {mejor_z:.4f}). Ampliá pasos_arriba/"
                f"pasos_abajo_foco_local."
            )

        # autofocus() ya deja la platina en mejor_z.

        # ---- c) foto de calidad alta ----------------------------

        camara.config_de_una(
            pixformat=pixformat_alta,
            resolucion=resolucion_alta,
        )

        image_data = camara.capturar()

        hora_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
            "autofoco_en_borde": en_borde,
        })

    resultado_posicionamiento["tiles_espiral"] = tiles
    resultado_posicionamiento["carpeta_salida"] = carpeta_salida

    return resultado_posicionamiento



#%%



grua = TrimakerCOSMOS("COM11")

grua.home()

print(f"{grua.pregunta_posicion_actual() = }")

# grua.desp_lin_abs(150,150,27.71)



#%%



cam = OpenMVCamera("COM7")

cam.conectar()

cam.mostrar(cam.capturar())



#%%



resultado = posicionar_en_muestra(
    cam,
    grua,
    x=150,
    y=153,
    z_arriba_seguro=27.9,
    z_min_seguridad=27.5,
    resolucion_global="QVGA",
    pixformat_global="GRAYSCALE",
    paso_grueso=0.005,
    paso_fino=0.0025,
    criterios_deteccion=None,
    margen_confirmacion=0.15,
    pasos_confirmacion=6,
    verificar_presencia=False,
    alturas_verificacion=3,
    umbral_intensidad_fondo=152.5,
    feedrate_xy=3000,
    espera_movimiento=0.3,
    hacer_foco_final=True,
    metrica_foco_final=None,
    pasos_arriba_foco_final=6,
    pasos_abajo_foco_final=6,
    guardar_fotos_foco_final=True,
    carpeta_fotos_foco_final=rf"C:\BackUp\Facultad\Labo 7\Fotos\26.10.02 ver fotos 5",
    foco_final_multimetrica_activado=True,
    pixformat_color_foco_final="RGB565",
    resolucion_color_foco_final="WQXGA2",
    devolveme_la_respuesta=False
)



#%%


def autofocus(
        camara,
        impresora,
        metrica,
        pasos_arriba=10,
        pasos_abajo=10,
        paso_z=0.0025,
        espera_movimiento=0.3,
        graficamos=True,
        guardamos=False,
        carpeta_guardado=None,
        vecinos_para_confirmar=3,
        pasos_extension_maxima=10,
        z_min_seguridad=None,
        z_max_seguro=None,
        ):
    """
    Barre en Z alrededor de la posición actual y devuelve la altura
    donde 'metrica' es máxima.

    Confirmación del máximo (puramente posicional, no exige que la
    métrica venga subiendo/bajando de forma monótona — puede tener
    ruido foto a foto y seguir teniendo un máximo real): se acepta
    como real sólo si, dentro de la lista de valores barridos, tiene
    al menos 'vecinos_para_confirmar' fotos a su izquierda Y a su
    derecha.

    Si el barrido inicial ('pasos_abajo' + 'pasos_arriba') no alcanza
    para eso, se extiende de a un paso por vez, en la dirección que
    haga falta (puede ser hacia arriba, hacia abajo, o ambas dentro
    de un mismo llamado), evaluando la métrica después de cada paso
    nuevo y cortando la extensión apenas el máximo queda confirmado
    (para no dar pasos de más).

    Si se agota 'pasos_extension_maxima', o la extensión cruzaría
    'z_min_seguridad' (hacia abajo) o 'z_max_seguro' (hacia arriba),
    sin lograr confirmar un máximo, se levanta PicoNoConfirmado.

    'z_min_seguridad'/'z_max_seguro' son opcionales (default None,
    sin límite) para no romper compatibilidad con llamadas que ya
    existen y no los pasan, pero conviene pasarlos siempre que haya
    riesgo real de choque con la muestra.
    """

    # ========================================================
    # POSICION INICIAL
    # ========================================================

    _, _, z_inicial = impresora.pregunta_posicion_actual()

    print(f"\nPosición Z inicial: {z_inicial} mm")

    # --------------------------------------------------------
    # No bajar/subir más de lo seguro en el barrido inicial
    # --------------------------------------------------------

    if z_min_seguridad is not None and pasos_abajo > 0:

        pasos_abajo_max = int((z_inicial - z_min_seguridad) / paso_z)

        if pasos_abajo > pasos_abajo_max:
            print(
                f"Aviso: pasos_abajo se recorta de {pasos_abajo} a "
                f"{max(pasos_abajo_max, 0)} para no cruzar "
                f"z_min_seguridad = {z_min_seguridad} mm."
            )
            pasos_abajo = max(pasos_abajo_max, 0)

    if z_max_seguro is not None and pasos_arriba > 0:

        pasos_arriba_max = int((z_max_seguro - z_inicial) / paso_z)

        if pasos_arriba > pasos_arriba_max:
            print(
                f"Aviso: pasos_arriba se recorta de {pasos_arriba} a "
                f"{max(pasos_arriba_max, 0)} para no cruzar "
                f"z_max_seguro = {z_max_seguro} mm."
            )
            pasos_arriba = max(pasos_arriba_max, 0)

    # ========================================================
    # RESULTADOS
    # ========================================================

    posiciones_z = []
    valores_metrica = []

    autofocus_completo = False

    try:

        if pasos_abajo > 0:

            impresora.desp_lin_rel(
                0,
                0,
                -pasos_abajo * paso_z
            )
            time.sleep(1)

        # =============
        # BARRIDO INICIAL EN Z
        # =============

        cantidad_fotos = pasos_arriba + pasos_abajo + 1

        for paso in range(cantidad_fotos):

            # --------------------------------------------
            # Tomar fotografía
            # --------------------------------------------

            image_data = camara.capturar()

            if guardamos:

                hora_actual = datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S"
                )

                camara.guardar_imagen(
                    image_data=image_data,
                    carpeta=carpeta_guardado,
                    nombre=f"{hora_actual} ({np.round(impresora.posicion_x, 4)}, {np.round(impresora.posicion_y, 4)}, {np.round(impresora.posicion_z, 4)})",
                    mostrar=False
                )

            # --------------------------------------------
            # Convertir a imagen OpenCV y calcular métrica
            # --------------------------------------------

            imagen = bytes_a_imagen(image_data)

            valor = metrica(imagen)

            posiciones_z.append(impresora.posicion_z)
            valores_metrica.append(valor)

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

        # ===========
        # CONFIRMAR EL MÁXIMO (extendiendo el barrido si hace falta)
        # ===========

        pasos_extendidos = 0

        while True:

            indx_max = int(np.argmax(valores_metrica))

            vecinos_izquierda = indx_max
            vecinos_derecha = len(valores_metrica) - 1 - indx_max

            print(f"{vecinos_izquierda = }")
            print(f"{vecinos_derecha = }")

            if (
                vecinos_izquierda >= vecinos_para_confirmar
                and vecinos_derecha >= vecinos_para_confirmar
            ):
                print(f"{vecinos_izquierda >= vecinos_para_confirmar = }")
                print(f"{vecinos_derecha >= vecinos_para_confirmar = }")
                break

            if pasos_extendidos >= pasos_extension_maxima:
                raise PicoNoConfirmado(
                    f"Se agotaron los {pasos_extension_maxima} pasos de "
                    f"extensión disponibles sin poder confirmar el máximo "
                    f"(le faltan "
                    f"{max(vecinos_para_confirmar - vecinos_izquierda, 0)} "
                    f"vecinos a la izquierda y "
                    f"{max(vecinos_para_confirmar - vecinos_derecha, 0)} "
                    f"a la derecha)."
                )

            # Se extiende de a un paso, priorizando el lado que todavía
            # tenga menos vecinos de los necesarios (si hacen falta los
            # dos lados, se van alternando solos: en cuanto uno queda
            # satisfecho, el próximo paso de extensión va para el otro).

            if vecinos_izquierda < vecinos_para_confirmar:

                z_nuevo = posiciones_z[0] - paso_z

                if z_min_seguridad is not None and z_nuevo < z_min_seguridad:
                    raise PicoNoConfirmado(
                        f"No se puede extender el barrido hacia abajo sin "
                        f"cruzar z_min_seguridad = {z_min_seguridad} mm."
                    )

                _, _, z_actual = impresora.pregunta_posicion_actual()

                impresora.desp_lin_rel(0, 0, z_nuevo - z_actual)
                time.sleep(espera_movimiento)

                print()

                image_data = camara.capturar()
                imagen = bytes_a_imagen(image_data)

                valores_metrica.insert(0, metrica(imagen))
                posiciones_z.insert(0, impresora.posicion_z)

            else:

                z_nuevo = posiciones_z[-1] + paso_z

                if z_max_seguro is not None and z_nuevo > z_max_seguro:
                    raise PicoNoConfirmado(
                        f"No se puede extender el barrido hacia arriba sin "
                        f"cruzar z_max_seguro = {z_max_seguro} mm."
                    )

                _, _, z_actual = impresora.pregunta_posicion_actual()

                impresora.desp_lin_rel(0, 0, z_nuevo - z_actual)
                time.sleep(espera_movimiento)

                image_data = camara.capturar()
                imagen = bytes_a_imagen(image_data)

                valores_metrica.append(metrica(imagen))
                posiciones_z.append(impresora.posicion_z)

            pasos_extendidos += 1

        # ==============
        # ENCONTRAR MEJOR POSICION
        # ==============

        indice_mejor = np.argmax(valores_metrica)

        mejor_z = posiciones_z[indice_mejor]
        mejor_valor = valores_metrica[indice_mejor]

        print("\n" + "=" * 50)
        print("RESULTADO DEL AUTOFOCUS")
        print("=" * 50)

        print(f"Mejor Z: {mejor_z:.6f} mm")
        print(f"Métrica: {mejor_valor:.6f}")

        # ===========
        # GRAFICAR LA METRICA
        # ===========

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



#%% Lo uso para moverme en la muestra



grua.desp_lin_rel(0,0,0.01)
cam.config_autofoco_local()
_ = autofocus(
    cam,
    grua,
    laplacian_variance,
    1,
    1,
    espera_movimiento=0.5,
    pasos_extension_maxima=10,
    graficamos=True,
    guardamos=False,
    carpeta_guardado=rf"C:\BackUp\Facultad\Labo 7\Fotos\26.10.02 autofocus guard 3",
    z_min_seguridad=27.5
)
cam.config_de_una()
cam.mostrar(cam.capturar())



#%%



# ==================
# EJEMPLO DE USO
# ==================

resultado = barrido_espiral_muestra(
    camara=cam,
    impresora=grua,
    x_ini=150.0,
    y_ini=153.0,
    z_arriba_seguro=27.79,   # MAYOR Z = más lejos de la muestra (seguro)
    z_min_seguridad=27.5,   # límite inferior (con margen del objetivo)
    carpeta_salida=r"C:\BackUp\Facultad\Labo 7\Fotos\26.10.02 mosaico parasitos fijos pasada 4",
    paso_x_mm = 0.1,
    paso_y_mm = 0.15,
    paso_grueso=0.005,
    guardar_fotos_autofoco_local=True,
    metrica_foco_final = laplacian_variance,
    pasos_radiales_maximo=15,   # primera prueba chica: 9 tiles
)

print(f"{len(resultado['tiles_espiral'])} fotos guardadas en "
      f"{resultado['carpeta_salida']}")



# %%
