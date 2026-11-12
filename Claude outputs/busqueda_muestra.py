#%%



import numbers
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# Módulos base del proyecto
from camara_openmv import bytes_a_imagen, validar_dict_params_camara
import impresora_cosmos
from metricas_imagen import (
    CRITERIOS_POR_DEFECTO,
    FUNCIONES_METRICA,
    detectar_aumento,
    evaluar_criterios,
    intensidad_fondo,
    linea_de_base,
    validar_aumentos,
)


# ============================================================
# EXCEPCIONES PROPIAS
# ============================================================

class MuestraNoEncontrada(Exception):
    """buscar_foco_en_xy() bajó hasta z_min_seguridad sin confirmar el foco.
    Trae lo medido en el atributo 'resultado_parcial' (línea de base, umbrales,
    candidatos descartados e historial), para poder ver qué pasó."""
    pass


class CampoVisualSinMuestra(MuestraNoEncontrada):
    """SUELTA desde el 2026-11-09: ya no la lanza ninguna función de la base
    (la lanzaba la verificación de presencia de buscar_muestra_en_z(), que
    desapareció). La importan aplicaciones (26.10.08, 26.10.13); se deja
    hasta que se haga limpieza.

    La verificación temprana (intensidad de fondo) indica que
    este (x, y) no tiene glóbulos rojos en ninguna altura razonable."""
    pass


class RiesgoDeColision(Exception):
    pass


class PicoNoConfirmado(Exception):
    """autofocus() no pudo confirmar el máximo que encontró: ni el
    barrido inicial ni la extensión (dentro de 'pasos_extension_maxima'
    y de los límites de seguridad) le dieron suficientes vecinos de
    cada lado. No implica que no haya muestra: puede ser una señal
    falsa, o simplemente que el máximo real está fuera del rango que
    se barrió."""
    pass


# ============================================================
# AUTOFOCUS
# ============================================================

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

    # Falla ANTES de mover nada (si no, el ValueError saltaría recién
    # en la primera foto, con la platina ya desplazada).
    if guardamos and carpeta_guardado is None:
        raise ValueError(
            "Si 'guardamos' es True hay que pasar 'carpeta_guardado'"
        )

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

    # --------------------------------------------------------
    # UNA sola forma de sacar una foto en este autofocus: captura,
    # guarda (si 'guardamos') y mide. Antes el guardado vivía solo en
    # el barrido inicial, así que las fotos de la extensión del
    # barrido se medían pero nunca se guardaban.
    # --------------------------------------------------------

    def _capturar_y_medir():

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

        imagen = bytes_a_imagen(image_data)

        return metrica(imagen)

    aviso_sin_piso_emitido = False

    try:

        if pasos_abajo > 0:

            impresora.desp_lin_cam(
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
            # Tomar fotografía (se guarda si corresponde) y medir
            # --------------------------------------------

            valor = _capturar_y_medir()

            posiciones_z.append(impresora.posicion_z)
            valores_metrica.append(valor)

            # --------------------------------------------
            # Mover al siguiente Z
            # --------------------------------------------

            if paso < cantidad_fotos - 1:

                impresora.desp_lin_cam(
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

                if z_min_seguridad is None and not aviso_sin_piso_emitido:
                    print(
                        "[AVISO DE SEGURIDAD] autofocus() está extendiendo el "
                        "barrido HACIA LA MUESTRA (Z decreciente) sin "
                        "'z_min_seguridad': no hay ningún piso que lo frene. "
                        "Pasale z_min_seguridad desde quien lo llama."
                    )
                    aviso_sin_piso_emitido = True

                _, _, z_actual = impresora.pregunta_posicion_actual()

                impresora.desp_lin_cam(0, 0, z_nuevo - z_actual)
                time.sleep(espera_movimiento)

                valor = _capturar_y_medir()

                valores_metrica.insert(0, valor)
                posiciones_z.insert(0, impresora.posicion_z)

            else:

                z_nuevo = posiciones_z[-1] + paso_z

                if z_max_seguro is not None and z_nuevo > z_max_seguro:
                    raise PicoNoConfirmado(
                        f"No se puede extender el barrido hacia arriba sin "
                        f"cruzar z_max_seguro = {z_max_seguro} mm."
                    )

                _, _, z_actual = impresora.pregunta_posicion_actual()

                impresora.desp_lin_cam(0, 0, z_nuevo - z_actual)
                time.sleep(espera_movimiento)

                valor = _capturar_y_medir()

                valores_metrica.append(valor)
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

                impresora.desp_lin_cam(
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

                impresora.desp_lin_cam(
                    0,
                    0,
                    desplazamiento
                )

                time.sleep(espera_movimiento)

    # ========================================================
    # DEVOLVER RESULTADOS
    # ========================================================

    return mejor_z, posiciones_z, valores_metrica


# ============================================================
# FASE 0: VERIFICACIÓN TEMPRANA DE PRESENCIA DE MUESTRA
# ============================================================

def verificar_presencia_muestra(
    camara,
    impresora,
    z_arriba_seguro,
    z_min_seguridad,
    alturas_a_revisar=3,
    umbral_intensidad_fondo=152.5,
    resolucion="QVGA",
    espera_movimiento=0.3,
    devolveme_la_respuesta=False,
):
    """
    SUELTA desde el 2026-11-09 (no la llama ninguna función de la base; se decide en la limpieza).

    Chequeo rápido ANTES del barrido grueso: revisa unas pocas alturas
    de Z (con pixformat RGB565, que es como se calibró el umbral) y
    decide si a este (x, y) le vale la pena buscarle foco.

    La intensidad de fondo (255 - media de gris) se mantiene
    prácticamente constante en Z para un campo visual CON muestra, y
    se separa muy bien de un campo visual vacío — no hace falta estar
    cerca del plano ideal para que esta verificación funcione, así
    que un puñado de alturas alcanza.

    Revisa de la altura más segura a la más riesgosa, y corta apenas
    una confirma presencia (no hace falta seguir arriesgando).
    """

    if z_min_seguridad >= z_arriba_seguro:
        raise ValueError(
            "z_min_seguridad tiene que ser menor que z_arriba_seguro"
        )

    camara.set_pixformat("RGB565")
    camara.set_framesize(resolucion)

    alturas = np.linspace(z_min_seguridad, z_arriba_seguro, alturas_a_revisar)
    alturas = alturas[::-1]  # de más segura (arriba) a más riesgosa (abajo)

    for altura in alturas:

        _, _, z_actual = impresora.pregunta_posicion_actual()

        desplazamiento = altura - z_actual

        if desplazamiento != 0:
            impresora.desp_lin_cam(
                0, 0, desplazamiento,
                devolveme_la_respuesta=devolveme_la_respuesta,
            )
            time.sleep(espera_movimiento)

        image_data = camara.capturar()
        imagen = bytes_a_imagen(image_data)

        valor = intensidad_fondo(imagen)

        print(
            f"[Verificación] Z = {impresora.posicion_z:.4f} mm | "
            f"intensidad_fondo = {valor:.2f}"
        )

        if valor > umbral_intensidad_fondo:
            return True

    return False


# ============================================================
# BÚSQUEDA DEL FOCO EN UN (x, y): ir, línea de base, detección relativa
# y confirmación con autofocus()
# ============================================================

def _entero(valor, nombre, minimo):
    if isinstance(valor, bool) or not isinstance(valor, numbers.Integral):
        raise TypeError(f"'{nombre}' tiene que ser un entero, no {valor!r}")

    if valor < minimo:
        raise ValueError(f"'{nombre}' tiene que ser >= {minimo}, no {valor}")


def buscar_foco_en_xy(
    camara,
    impresora,
    x,
    y,
    z_arriba_seguro,
    z_min_seguridad,
    dict_params_global,
    aumento_deteccion,
    z_seguro_xy=40.0,
    pasos_linea_base=5,
    fotos_seguidas_para_detectar=2,
    paso_grueso=0.005,
    paso_fino=0.0025,
    margen_confirmacion=0.15,
    pasos_confirmacion=6,
    vecinos_para_confirmar=3,
    pasos_extension_maxima=3,
    feedrate_xy=3000,
    espera_movimiento=0.5,
    devolveme_la_respuesta=False,
):
    """
    Lleva la platina a (x, y), baja en Z hasta que las métricas de enfoque suben respecto de una línea de base y deja la platina en el máximo confirmado.

    Antiguo: buscar_muestra_en_z() y posicionar_en_muestra() (2026-11-09: se unieron en
    esta; ya no verifica la presencia de la muestra ni hace foco final: si hace falta,
    se lo pide a autofocus() o a foco_final_multimetrica() desde afuera, con la platina
    donde esta función la deja).

    Pasos:
    1) moverse_a_xy() a (x, y) con 'z_seguro_xy' como altura de viaje (los viajes largos
       suben hasta ahí; los cortos mantienen el Z) y después va a 'z_arriba_seguro'.
       Si la platina no llegó a (x, y) (destino fuera de rango) se detiene con
       RuntimeError, sin bajar.
    2) Configura la cámara con 'dict_params_global' y baja de a 'paso_grueso' sacando
       fotos. Las primeras 'pasos_linea_base' fotos son la línea de base: el promedio
       de cada métrica. 'z_arriba_seguro' tiene que estar lejos del foco (si no, la
       línea de base ya incluye la subida y no se detecta nada).
    3) Detección: una métrica detecta cuando supera su línea de base por más de su
       aumento (promedio * (1 + aumento)) en 'fotos_seguidas_para_detectar' fotos
       seguidas. Si lo logran varias, se elige la que más pasó su umbral. La regla
       está en metricas_imagen.detectar_aumento().
    4) Confirmación: autofocus() con esa métrica, alrededor de donde se detectó (paso
       'paso_fino', 'pasos_confirmacion' hacia arriba y hacia abajo). Se confirma si
       autofocus() confirma estructuralmente el máximo (ver su docstring) y el máximo
       supera el umbral de detección * (1 + 'margen_confirmacion'). Si no, es una señal
       falsa: se salta un 'paso_grueso' más abajo del punto de la detección y se sigue
       bajando (las fotos seguidas vuelven a contarse desde cero; la línea de base es
       la misma). Si se llega a 'z_min_seguridad' sin confirmar: MuestraNoEncontrada.

    Parámetros:
      'dict_params_global': parámetros de la cámara para todo el barrido
        (pixformat, resolucion, exposure_us, gain_db, whitebal; ver
        OpenMVCamera.config_de_una()). Obligatorio; se valida antes de mover nada.
      'aumento_deteccion': {métrica: aumento}, uno por métrica que se quiere usar
        ("laplacian", "tenengrad", "varianza"), como fracción: 0.3 = +30 % sobre el
        promedio de la línea de base. Obligatorio y sin valores por defecto: se miden.
        Se mide siempre las tres métricas (queda en el historial), pero solo detectan
        las que figuren acá.
      'z_seguro_xy': altura (mm) a la que se viaja en XY; tiene que ser >= z_arriba_seguro.
      'vecinos_para_confirmar', 'pasos_extension_maxima': los de autofocus().

    Supuestos de seguridad (¡no tocar sin revisar el hardware!):
    - Z creciente = alejarse de la muestra (más seguro); Z decreciente = acercarse.
    - 'z_min_seguridad' ya tiene que venir con el margen respecto de la distancia de
      trabajo del objetivo (300 µm). Esta función NUNCA manda una posición menor a ese
      valor, y se lo pasa también a autofocus().
    - Todo se revisa ANTES de mover: diccionario de cámara, aumentos, enteros, que
      z_min_seguridad < z_arriba_seguro <= z_seguro_xy y que la línea de base entre
      entre z_arriba_seguro y z_min_seguridad con lugar para detectar.

    Devuelve un diccionario:
        {
            "x", "y": donde quedó la platina (en la grilla de la impresora),
            "z_confirmado", "metrica_confirmada", "valor_confirmado",
            "linea_base": {métrica: promedio} (las tres),
            "umbrales_deteccion": {métrica: umbral} (las de 'aumento_deteccion'),
            "candidatos": [una entrada por cada detección, con su resultado:
                "confirmado" | "no_alcanza_umbral" | "pico_no_confirmado"],
            "historial_barrido_grueso": [{"z", "fase" ("linea_base" | "barrido"),
                "laplacian", "tenengrad", "varianza"}, ...],
        }
    Con MuestraNoEncontrada, el mismo diccionario sin los "confirmado" va en
    error.resultado_parcial.
    """

    # --------------------------------------------------------
    # Todo se revisa ANTES de mover nada
    # --------------------------------------------------------

    validar_dict_params_camara(dict_params_global, "dict_params_global")
    validar_aumentos(aumento_deteccion, "aumento_deteccion")

    _entero(pasos_linea_base, "pasos_linea_base", 1)
    _entero(fotos_seguidas_para_detectar, "fotos_seguidas_para_detectar", 1)
    _entero(vecinos_para_confirmar, "vecinos_para_confirmar", 1)
    _entero(pasos_confirmacion, "pasos_confirmacion", 0)
    _entero(pasos_extension_maxima, "pasos_extension_maxima", 0)

    if z_min_seguridad >= z_arriba_seguro:
        raise ValueError(
            "z_min_seguridad tiene que ser menor que z_arriba_seguro"
        )

    if z_seguro_xy < z_arriba_seguro:
        raise ValueError(
            "z_seguro_xy (la altura de los viajes en XY) no puede ser menor "
            "que z_arriba_seguro"
        )

    if z_seguro_xy > impresora.max_posicion_z:
        raise ValueError(
            f"z_seguro_xy = {z_seguro_xy} mm pasa el máximo de la impresora "
            f"({impresora.max_posicion_z} mm)"
        )

    if paso_grueso <= 0 or paso_fino <= 0:
        raise ValueError("paso_grueso y paso_fino tienen que ser positivos")

    if margen_confirmacion < 0:
        raise ValueError("margen_confirmacion no puede ser negativo")

    if z_arriba_seguro - pasos_linea_base * paso_grueso < z_min_seguridad - 1e-9:
        raise ValueError(
            f"La línea de base ({pasos_linea_base} fotos, de a {paso_grueso} mm) "
            f"no entra entre z_arriba_seguro ({z_arriba_seguro} mm) y "
            f"z_min_seguridad ({z_min_seguridad} mm) con lugar para detectar."
        )

    _, _, z_actual = impresora.pregunta_posicion_actual()

    if z_actual is None:
        raise RuntimeError(
            "La posición de la impresora no está inicializada. "
            "Llamá primero a impresora.home()"
        )

    if z_actual < z_min_seguridad:
        raise RiesgoDeColision(
            f"La posición actual en Z ({z_actual} mm) ya está por debajo "
            f"del límite de seguridad ({z_min_seguridad} mm). "
            f"Se aborta antes de mover nada."
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

    # desp_lin_000() cancela SIN mover si el destino sale del rango: no seguir
    # buscando en otro lado como si nada.
    if (
        abs(impresora.posicion_x - x) > impresora.resolucion_x + 1e-9
        or abs(impresora.posicion_y - y) > impresora.resolucion_y + 1e-9
    ):
        raise RuntimeError(
            f"La platina no llegó a (x, y) = ({x}, {y}) mm: está en "
            f"({impresora.posicion_x}, {impresora.posicion_y}) mm. "
            f"No se baja en Z."
        )

    _, _, z_actual = impresora.pregunta_posicion_actual()

    desplazamiento_a_arriba = z_arriba_seguro - z_actual

    if abs(desplazamiento_a_arriba) > 1e-9:
        impresora.desp_lin_cam(
            0, 0, desplazamiento_a_arriba,
            devolveme_la_respuesta=devolveme_la_respuesta,
        )
        time.sleep(espera_movimiento)

    # --------------------------------------------------------
    # 2) a 4) Línea de base, detección relativa y confirmación
    # --------------------------------------------------------

    camara.config_de_una(dict_params_global)

    historial = []
    candidatos = []
    resultado_confirmado = None
    indice_desde = 0

    z = impresora.posicion_z

    while True:

        image_data = camara.capturar()
        imagen = bytes_a_imagen(image_data)

        # Se miden las tres métricas siempre (sirve para analizar después);
        # detectan solo las de 'aumento_deteccion'.
        valores = {nombre: funcion(imagen) for nombre, funcion in FUNCIONES_METRICA.items()}

        fase = "linea_base" if len(historial) < pasos_linea_base else "barrido"

        historial.append({"z": impresora.posicion_z, "fase": fase, **valores})

        print(
            f"Z = {impresora.posicion_z:.4f} mm | "
            + " | ".join(f"{n} = {v:.2f}" for n, v in valores.items())
            + (" | (línea de base)" if fase == "linea_base" else "")
        )

        deteccion = detectar_aumento(
            historial,
            aumento_deteccion,
            pasos_linea_base=pasos_linea_base,
            fotos_seguidas=fotos_seguidas_para_detectar,
            desde=indice_desde,
        )

        if deteccion is not None:

            nombre_disparo = deteccion["metrica"]
            z_disparo = impresora.posicion_z
            umbral_deteccion = deteccion["umbrales"][nombre_disparo]
            umbral_confirmacion = umbral_deteccion * (1 + margen_confirmacion)

            candidato = {
                "z_disparo": z_disparo,
                "metrica": nombre_disparo,
                "valor": deteccion["valor"],
                "umbral_deteccion": umbral_deteccion,
                "umbral_confirmacion": umbral_confirmacion,
                "resultado": None,
                "z_pico": None,
                "valor_pico": None,
            }
            candidatos.append(candidato)

            print(
                f"  -> Candidato: '{nombre_disparo}' en Z = {z_disparo:.4f} mm "
                f"(valor = {deteccion['valor']:.2f}, umbral de detección = "
                f"{umbral_deteccion:.2f})"
            )

            margen_abajo_disponible = z_disparo - z_min_seguridad
            pasos_abajo_confirmacion = max(
                min(pasos_confirmacion, int(margen_abajo_disponible // paso_fino)),
                0,
            )

            metrica_funcion = FUNCIONES_METRICA[nombre_disparo]

            # ------------------------------------------------
            # Confirmación del máximo: delegada enteramente en
            # autofocus() (ver su docstring). PicoNoConfirmado se
            # trata igual que una señal falsa: se descarta y se
            # sigue bajando con el barrido grueso.
            # ------------------------------------------------

            try:

                z_pico, posiciones_pico, valores_pico = autofocus(
                    camara=camara,
                    impresora=impresora,
                    metrica=metrica_funcion,
                    pasos_arriba=pasos_confirmacion,
                    pasos_abajo=pasos_abajo_confirmacion,
                    paso_z=paso_fino,
                    espera_movimiento=espera_movimiento,
                    graficamos=False,
                    vecinos_para_confirmar=vecinos_para_confirmar,
                    pasos_extension_maxima=pasos_extension_maxima,
                    z_min_seguridad=z_min_seguridad,
                    z_max_seguro=z_arriba_seguro,
                )

                valor_pico = max(valores_pico)

                candidato["z_pico"] = z_pico
                candidato["valor_pico"] = valor_pico

                if valor_pico >= umbral_confirmacion:

                    # autofocus() ya deja posicionado al equipo en
                    # z_pico (su propio máximo), no hace falta volver
                    # a moverse.

                    print(
                        f"  -> CONFIRMADO: {nombre_disparo} = {valor_pico:.2f} "
                        f">= {umbral_confirmacion:.2f}"
                    )

                    candidato["resultado"] = "confirmado"

                    resultado_confirmado = {
                        "z_confirmado": z_pico,
                        "metrica_confirmada": nombre_disparo,
                        "valor_confirmado": valor_pico,
                    }
                    break

                else:

                    candidato["resultado"] = "no_alcanza_umbral"

                    print(
                        f"  -> Señal falsa descartada: {nombre_disparo} = "
                        f"{valor_pico:.2f} (el máximo se confirmó "
                        f"estructuralmente, pero no alcanza el umbral de "
                        f"{umbral_confirmacion:.2f}). Se sigue bajando."
                    )

            except PicoNoConfirmado as motivo:

                candidato["resultado"] = "pico_no_confirmado"

                print(
                    f"  -> Señal falsa descartada: '{nombre_disparo}' no "
                    f"tiene un máximo confirmable ({motivo}). Se sigue "
                    f"bajando."
                )

            # Señal falsa (por cualquiera de los dos motivos de
            # arriba): saltar un paso_grueso más abajo del punto de la
            # detección, para no volver a toparse con la misma señal, y
            # retomar el barrido. Las fotos seguidas vuelven a contarse
            # desde cero (indice_desde); la línea de base es la misma.

            z_siguiente = z_disparo - paso_grueso

            if z_siguiente < z_min_seguridad:
                break

            _, _, z_actual = impresora.pregunta_posicion_actual()

            impresora.desp_lin_cam(
                0, 0, z_siguiente - z_actual,
                devolveme_la_respuesta=devolveme_la_respuesta,
            )
            time.sleep(espera_movimiento)

            z = impresora.posicion_z
            indice_desde = len(historial)
            continue

        if z - z_min_seguridad <= 1e-9:
            break

        paso = -min(paso_grueso, z - z_min_seguridad)

        impresora.desp_lin_cam(
            0, 0, paso, devolveme_la_respuesta=devolveme_la_respuesta
        )
        time.sleep(espera_movimiento)

        z = impresora.posicion_z

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------

    base = linea_de_base(historial, FUNCIONES_METRICA, pasos_linea_base)

    medido = {
        "x": impresora.posicion_x,
        "y": impresora.posicion_y,
        "linea_base": base,
        "umbrales_deteccion": (
            None if base is None
            else {n: base[n] * (1 + a) for n, a in aumento_deteccion.items()}
        ),
        "candidatos": candidatos,
        "historial_barrido_grueso": historial,
    }

    if resultado_confirmado is None:
        error = MuestraNoEncontrada(
            f"No se confirmó el foco entre Z = {z_arriba_seguro} mm y "
            f"Z = {z_min_seguridad} mm en (x, y) = ({x}, {y})."
        )
        error.resultado_parcial = medido
        raise error

    return {**resultado_confirmado, **medido}


# ============================================================
# FOCO FINAL MULTIMÉTRICA: UN máximo por cada una de las 3 métricas,
# fotografiados en alta calidad (color, máxima resolución)
# ============================================================

def _agrupar_maximos_por_z(resultados_por_metrica, tolerancia_z):
    """
    Agrupa métricas cuyo Z de máximo cae a 'tolerancia_z' (o menos)
    una de la otra, para sacar una sola foto por grupo en vez de
    fotos repetidas del mismo punto -- y para poder reportar cuándo
    dos o tres métricas apuntan al mismo plano (señal más fuerte de
    que ese es el foco correcto).

    Devuelve una lista de grupos, cada uno
    {"z": <z representativo>, "metricas": [nombre, ...]}, ordenada
    por Z creciente.
    """

    pendientes = sorted(
        resultados_por_metrica.items(),
        key=lambda item: item[1]["z"]
    )

    grupos = []

    for nombre, datos in pendientes:

        if grupos and abs(datos["z"] - grupos[-1]["z"]) <= tolerancia_z:

            grupos[-1]["metricas"].append(nombre)

            # Promedio simple, para no quedarnos solo con el primer
            # Z del grupo si hay más de dos coincidiendo.
            n = len(grupos[-1]["metricas"])
            grupos[-1]["z"] = (
                grupos[-1]["z"] * (n - 1) + datos["z"]
            ) / n

        else:
            grupos.append({"z": datos["z"], "metricas": [nombre]})

    return grupos


def foco_final_multimetrica(
    camara,
    impresora,
    dict_params_grayscale,
    dict_params_color,
    z_min_seguridad,
    z_max_seguro=None,
    paso_z=0.0025,
    pasos_arriba=10,
    pasos_abajo=10,
    vecinos_para_confirmar=3,
    pasos_extension_maxima=5,
    espera_movimiento=0.5,
    guardamos=False,
    carpeta_guardado=None,
    graficamos=True,
):
    """
    Pensada para usarse justo después de que buscar_muestra_en_z()
    confirma una muestra (es decir, arrancando ya cerca del plano de
    foco). Barre en Z UNA sola vez, con la cámara configurada con
    'dict_params_grayscale' (pensada para GRAYSCALE), evaluando las tres
    métricas disponibles (laplacian_variance, tenengrad,
    varianza_intensidad) sobre cada foto -- no hace tres barridos
    separados, reusa las mismas fotos para las tres.

    'dict_params_grayscale' y 'dict_params_color' son diccionarios con
    los parámetros de la cámara (pixformat, resolucion, exposure_us,
    gain_db, whitebal; ver OpenMVCamera.config_de_una()). Son
    obligatorios y se validan antes de mover nada. Por ejemplo:
        dict_params_grayscale = {"pixformat": "GRAYSCALE", "resolucion": "QVGA",
                                 "exposure_us": 50000, "gain_db": 25,
                                 "whitebal": (66.22236, 61.43028, 60.206)}
        dict_params_color = {"pixformat": "RGB565", "resolucion": "WQXGA2",
                             "exposure_us": 50000, "gain_db": 25,
                             "whitebal": (66.22236, 61.43028, 60.206)}

    El máximo de cada métrica se confirma con el mismo criterio
    posicional que autofocus() (al menos 'vecinos_para_confirmar'
    valores a cada lado, ver su docstring). Si hace falta, extiende
    el barrido de a un paso -- compartido entre las tres métricas, en
    la dirección que haga falta -- respetando
    'z_min_seguridad'/'z_max_seguro', hasta que las tres queden
    confirmadas o se agote 'pasos_extension_maxima'. Si se agota sin
    confirmar alguna, se usa igual el mejor valor que se haya
    encontrado para esa métrica, marcada como no confirmada en el
    resultado (no se levanta ninguna excepción: a esta altura ya se
    sabe que hay muestra, así que no tiene sentido abortar).

    Después va a la altura del máximo de cada métrica, configura la
    cámara con 'dict_params_color' (por ejemplo RGB565 a "WQXGA2", o
    sea "4K") y saca ahí la foto de mayor calidad. Si dos
    o tres métricas comparten el mismo Z (a 'paso_z' de diferencia o
    menos), se saca una sola foto para ese grupo y se la etiqueta con
    todas las métricas que coinciden, en vez de repetir la foto.

    Si 'guardamos' es True, además de esa foto del máximo:

      - Se guardan, también con 'dict_params_color',
        las fotos de las alturas vecinas: las 'vecinos_para_confirmar'
        alturas (de a 'paso_z') de cada lado del máximo de cada métrica,
        o sea las mismas que miró el criterio de confirmación. Con
        vecinos_para_confirmar = 3 son como mínimo 7 fotos (el máximo y 3
        a cada lado); son más si los máximos de las métricas no
        coinciden. Cada altura se fotografía una sola vez aunque las
        ventanas de dos métricas se pisen, de abajo hacia arriba. Las
        alturas que cruzarían 'z_min_seguridad' o 'z_max_seguro' se
        omiten (con un aviso).
      - Se guarda el gráfico de las tres métricas (el mismo que se
        muestra con 'graficamos') como PNG, en la misma carpeta.

    Nombres de archivo (la x, y, z son las de la platina en la foto):
        foto del máximo: "<métricas> (x, y, z)"   (igual que siempre)
        foto vecina:     "<métricas> vecino +k (x, y, z)"
        gráfico:         "plot metricas (x, y, z)", con la z de donde
                         arrancó la función.
    k es la cantidad de pasos de 'paso_z' respecto de la foto del máximo
    (positivo = más arriba, o sea más lejos de la muestra). Si una altura
    vecina de un grupo coincide con la de otro grupo, queda con el nombre
    del grupo al que le queda más cerca (el máximo, si es uno).

    Al terminar, la platina queda donde la dejaba siempre esta función:
    en el máximo del último grupo (el de Z más alto).

    Devuelve:
        {
            "laplacian": {
                "z": ..., "valor": ..., "confirmado": bool,
                "indice_en_barrido": int,   # posición de su máximo en posiciones_z
                "archivo": Path | None,     # la foto del máximo (del grupo)
                "archivos_ventana": [Path, ...],  # fotos de sus vecinos y del máximo, de abajo hacia arriba
            },
            "tenengrad": {...},
            "varianza":  {...},
            "posiciones_z": [...],          # el barrido GRAYSCALE completo
            "valores_por_metrica": {...},   # idem, las 3 listas
            "archivo_plot": Path | None,    # el gráfico guardado
        }
    """

    # Los dos diccionarios se revisan ahora: el de color recién se usa
    # después de barrer, y un error ahí dejaría la platina a mitad de camino.
    validar_dict_params_camara(dict_params_grayscale, "dict_params_grayscale")
    validar_dict_params_camara(dict_params_color, "dict_params_color")

    if guardamos and carpeta_guardado is None:
        raise ValueError(
            "Si 'guardamos' es True hay que pasar 'carpeta_guardado'"
        )

    _, _, z_inicial = impresora.pregunta_posicion_actual()

    print(f"\n[foco_final_multimetrica] Z inicial: {z_inicial} mm")

    # --------------------------------------------------------
    # No bajar/subir más de lo seguro en el barrido inicial
    # --------------------------------------------------------

    if z_min_seguridad is not None and pasos_abajo > 0:
        pasos_abajo_max = int((z_inicial - z_min_seguridad) / paso_z)
        pasos_abajo = max(min(pasos_abajo, pasos_abajo_max), 0)

    if z_max_seguro is not None and pasos_arriba > 0:
        pasos_arriba_max = int((z_max_seguro - z_inicial) / paso_z)
        pasos_arriba = max(min(pasos_arriba, pasos_arriba_max), 0)

    # --------------------------------------------------------
    # Barrido inicial (dict_params_grayscale), evaluando las tres métricas
    # --------------------------------------------------------

    camara.config_de_una(dict_params_grayscale)

    posiciones_z = []
    valores_por_metrica = {nombre: [] for nombre in FUNCIONES_METRICA}

    if pasos_abajo > 0:
        impresora.desp_lin_cam(0, 0, -pasos_abajo * paso_z)
        time.sleep(1)

    cantidad_fotos = pasos_arriba + pasos_abajo + 1

    for paso in range(cantidad_fotos):

        image_data = camara.capturar()
        imagen = bytes_a_imagen(image_data)

        posiciones_z.append(impresora.posicion_z)

        for nombre, funcion in FUNCIONES_METRICA.items():
            valores_por_metrica[nombre].append(funcion(imagen))

        if paso < cantidad_fotos - 1:
            impresora.desp_lin_cam(0, 0, paso_z)
            time.sleep(espera_movimiento)

    # --------------------------------------------------------
    # Confirmar el máximo de cada métrica (extensión compartida: una
    # foto nueva sirve para las tres a la vez)
    # --------------------------------------------------------

    pasos_extendidos = 0

    while True:

        estado = {}
        todas_confirmadas = True
        necesita_izquierda = False
        necesita_derecha = False

        for nombre, valores in valores_por_metrica.items():

            indx_max = int(np.argmax(valores))
            vecinos_izquierda = indx_max
            vecinos_derecha = len(valores) - 1 - indx_max
            confirmada = (
                vecinos_izquierda >= vecinos_para_confirmar
                and vecinos_derecha >= vecinos_para_confirmar
            )

            estado[nombre] = {"indx_max": indx_max, "confirmada": confirmada}

            if not confirmada:
                todas_confirmadas = False
                necesita_izquierda = necesita_izquierda or (
                    vecinos_izquierda < vecinos_para_confirmar
                )
                necesita_derecha = necesita_derecha or (
                    vecinos_derecha < vecinos_para_confirmar
                )

        if todas_confirmadas:
            break

        if pasos_extendidos >= pasos_extension_maxima:
            print(
                f"Aviso: se agotaron los {pasos_extension_maxima} pasos de "
                f"extensión sin confirmar todas las métricas. Se usa el "
                f"mejor valor disponible para: "
                f"{[n for n, e in estado.items() if not e['confirmada']]}"
            )
            break

        if necesita_izquierda:

            z_nuevo = posiciones_z[0] - paso_z

            if z_min_seguridad is not None and z_nuevo < z_min_seguridad:
                print(
                    "Aviso: no se puede extender más hacia abajo sin "
                    "cruzar z_min_seguridad. Se usa el mejor valor "
                    "disponible para las métricas que no llegaron a "
                    "confirmarse."
                )
                break

            _, _, z_actual = impresora.pregunta_posicion_actual()
            impresora.desp_lin_cam(0, 0, z_nuevo - z_actual)
            time.sleep(espera_movimiento)

            image_data = camara.capturar()
            imagen = bytes_a_imagen(image_data)

            posiciones_z.insert(0, impresora.posicion_z)
            for nombre, funcion in FUNCIONES_METRICA.items():
                valores_por_metrica[nombre].insert(0, funcion(imagen))

        else:  # necesita_derecha

            z_nuevo = posiciones_z[-1] + paso_z

            if z_max_seguro is not None and z_nuevo > z_max_seguro:
                print(
                    "Aviso: no se puede extender más hacia arriba sin "
                    "cruzar z_max_seguro. Se usa el mejor valor "
                    "disponible para las métricas que no llegaron a "
                    "confirmarse."
                )
                break

            _, _, z_actual = impresora.pregunta_posicion_actual()
            impresora.desp_lin_cam(0, 0, z_nuevo - z_actual)
            time.sleep(espera_movimiento)

            image_data = camara.capturar()
            imagen = bytes_a_imagen(image_data)

            posiciones_z.append(impresora.posicion_z)
            for nombre, funcion in FUNCIONES_METRICA.items():
                valores_por_metrica[nombre].append(funcion(imagen))

        pasos_extendidos += 1

    # Estado final de cada métrica (por si el 'while' cortó por
    # agotar el presupuesto de extensión, en vez de confirmar todo)
    resultados_por_metrica = {}

    for nombre, valores in valores_por_metrica.items():

        indx_max = int(np.argmax(valores))
        vecinos_izquierda = indx_max
        vecinos_derecha = len(valores) - 1 - indx_max
        confirmada = (
            vecinos_izquierda >= vecinos_para_confirmar
            and vecinos_derecha >= vecinos_para_confirmar
        )

        resultados_por_metrica[nombre] = {
            "z": posiciones_z[indx_max],
            "valor": valores[indx_max],
            "confirmado": confirmada,
            "indice_en_barrido": indx_max,
            "archivo": None,
            "archivos_ventana": [],
        }

    # --------------------------------------------------------
    # Fotos en alta calidad (color, 'dict_params_color')
    #
    # Las alturas se arman sobre la grilla del barrido (múltiplos de
    # paso_z a partir de posiciones_z[0]; 'j' es el índice entero de la
    # altura) y se fotografían UNA sola vez cada una, de abajo hacia
    # arriba, aunque las ventanas de vecinos de dos grupos se pisen.
    # --------------------------------------------------------

    grupos = _agrupar_maximos_por_z(resultados_por_metrica, tolerancia_z=paso_z)

    z_ref = posiciones_z[0]

    def _z_de_indice(j):
        return float(np.round(z_ref + j * paso_z, 4))

    # plan[j] = {"grupo": índice en 'grupos', "k": pasos respecto del
    # máximo del grupo}; k == 0 es la foto del máximo.
    plan = {}

    for g, grupo in enumerate(grupos):

        # La altura de siempre: desp_lin_cam() redondea hacia abajo al
        # múltiplo de la resolución del eje (el Z promedio del grupo puede
        # caer entre dos alturas del barrido).
        grupo["j"] = int(np.floor((grupo["z"] - z_ref) / paso_z + 1e-6))

        plan[grupo["j"]] = {"grupo": g, "k": 0}

    alturas_omitidas = []

    if guardamos:

        for g, grupo in enumerate(grupos):

            indices = [
                resultados_por_metrica[nombre_metrica]["indice_en_barrido"]
                for nombre_metrica in grupo["metricas"]
            ]

            for j in range(
                min(indices) - vecinos_para_confirmar,
                max(indices) + vecinos_para_confirmar + 1,
            ):

                z_j = _z_de_indice(j)

                if (
                    (z_min_seguridad is not None and z_j < z_min_seguridad - 1e-9)
                    or (z_max_seguro is not None and z_j > z_max_seguro + 1e-9)
                ):
                    alturas_omitidas.append(z_j)
                    continue

                k = j - grupo["j"]
                previo = plan.get(j)

                # Un máximo nunca se pisa; entre vecinos gana el más cercano.
                if previo is None or (previo["k"] != 0 and abs(k) < abs(previo["k"])):
                    plan[j] = {"grupo": g, "k": k}

    if alturas_omitidas:
        alturas_omitidas = sorted(set(alturas_omitidas))
        print(
            f"[foco_final_multimetrica] Aviso: se omiten {len(alturas_omitidas)} "
            f"alturas vecinas por salirse de [z_min_seguridad, z_max_seguro]: "
            f"{alturas_omitidas}"
        )

    camara.config_de_una(dict_params_color)

    archivos_por_indice = {}

    for j in sorted(plan):

        _, _, z_actual = impresora.pregunta_posicion_actual()

        desplazamiento = _z_de_indice(j) - z_actual

        if abs(desplazamiento) > 1e-9:
            impresora.desp_lin_cam(0, 0, desplazamiento)
            time.sleep(espera_movimiento)

        image_data = camara.capturar()

        rol = plan[j]
        etiqueta_metricas = "+".join(grupos[rol["grupo"]]["metricas"])
        k = rol["k"]

        posicion = (
            f"({np.round(impresora.posicion_x, 4)}, "
            f"{np.round(impresora.posicion_y, 4)}, "
            f"{np.round(impresora.posicion_z, 4)})"
        )

        if k == 0:
            nombre = f"{etiqueta_metricas} {posicion}"
            descripcion = f"para: {etiqueta_metricas}"
        else:
            nombre = f"{etiqueta_metricas} vecino {k:+d} {posicion}"
            descripcion = f"vecino {k:+d} de: {etiqueta_metricas}"

        print(
            f"[foco_final_multimetrica] Foto en Z = "
            f"{impresora.posicion_z:.4f} mm {descripcion}"
        )

        archivo = None

        if guardamos:

            archivo = camara.guardar_imagen(
                image_data=image_data,
                carpeta=carpeta_guardado,
                nombre=nombre,
                mostrar=False,
            )

        archivos_por_indice[j] = archivo

    # Con las fotos de los vecinos la platina termina más arriba que el
    # máximo del último grupo: se la devuelve ahí, que es donde esta
    # función la dejaba antes de guardar los vecinos.
    _, _, z_actual = impresora.pregunta_posicion_actual()

    desplazamiento = _z_de_indice(grupos[-1]["j"]) - z_actual

    if abs(desplazamiento) > 1e-9:
        impresora.desp_lin_cam(0, 0, desplazamiento)
        time.sleep(espera_movimiento)

    for grupo in grupos:

        for nombre_metrica in grupo["metricas"]:

            datos = resultados_por_metrica[nombre_metrica]
            indice = datos["indice_en_barrido"]

            datos["archivo"] = archivos_por_indice.get(grupo["j"])

            datos["archivos_ventana"] = [
                archivos_por_indice[j]
                for j in range(
                    indice - vecinos_para_confirmar,
                    indice + vecinos_para_confirmar + 1,
                )
                if archivos_por_indice.get(j) is not None
            ]

    # --------------------------------------------------------
    # Gráfico: las tres métricas comparten el mismo eje Z, porque
    # salieron del mismo barrido. Se muestra ('graficamos') y/o se
    # guarda como PNG ('guardamos'), con el mismo contenido.
    # --------------------------------------------------------

    archivo_plot = None

    if graficamos or guardamos:

        fig, ax = plt.subplots()

        for nombre, valores in valores_por_metrica.items():

            ax.plot(posiciones_z, valores/np.max(valores), marker="o", label=nombre)

            datos = resultados_por_metrica[nombre]

            ax.scatter(
                datos["z"],
                1,#datos["valor"],
                s=100,
                zorder=5,
                marker="*" if datos["confirmado"] else "x",
            )

        titulo = "Foco final multimétrica"

        coincidencias = [g for g in grupos if len(g["metricas"]) > 1]

        if coincidencias:
            etiquetas = [" = ".join(g["metricas"]) for g in coincidencias]
            titulo += " (coinciden: " + "; ".join(etiquetas) + ")"

        ax.set_xlabel("Z [mm]")
        ax.set_ylabel("Métrica")
        ax.set_title(titulo)
        ax.grid()
        ax.legend()

        if guardamos:

            # Antes de plt.show(): en algunos backends show() destruye la figura.
            carpeta_plot = Path(carpeta_guardado)
            carpeta_plot.mkdir(parents=True, exist_ok=True)

            archivo_plot = carpeta_plot / (
                f"plot metricas ({np.round(impresora.posicion_x, 4)}, "
                f"{np.round(impresora.posicion_y, 4)}, "
                f"{np.round(z_inicial, 4)}).png"
            )

            fig.savefig(archivo_plot, dpi=150, bbox_inches="tight")

            print(f"[foco_final_multimetrica] Gráfico guardado en:\n{archivo_plot}")

        if graficamos:
            plt.show()
        else:
            plt.close(fig)

    resultados_por_metrica["posiciones_z"] = posiciones_z
    resultados_por_metrica["valores_por_metrica"] = valores_por_metrica
    resultados_por_metrica["archivo_plot"] = archivo_plot

    return resultados_por_metrica


# ============================================================
# EJEMPLO DE USO
# ============================================================

# dict_params_grayscale = {
#     "pixformat": "GRAYSCALE",
#     "resolucion": "QVGA",
#     "exposure_us": 50000,
#     "gain_db": 25,
#     "whitebal": (66.22236, 61.43028, 60.206),
# }
#
# # Los aumentos se miden (no hay valores por defecto): 0.3 = +30 % sobre la línea de base.
# aumento_deteccion = {"laplacian": ..., "tenengrad": ..., "varianza": ...}
#
# resultado = buscar_foco_en_xy(
#     camara=cam,
#     impresora=grua,
#     x=150.0,
#     y=100.0,
#     z_arriba_seguro=27.9,   # MAYOR Z = más lejos de la muestra (seguro); lejos del foco
#     z_min_seguridad=27.5,   # límite inferior, con margen del objetivo
#     dict_params_global=dict_params_grayscale,
#     aumento_deteccion=aumento_deteccion,
# )
#
# print(resultado["z_confirmado"], resultado["metrica_confirmada"])
# # El autofoco final, si hace falta, se hace desde afuera: autofocus() o
# # foco_final_multimetrica(), con la platina donde quedó.


#%%
