#%%



import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt

import io
from PIL import Image


# ============================================================
# EXCEPCIONES PROPIAS
# ============================================================

class MuestraNoEncontrada(Exception):
    pass


class CampoVisualSinMuestra(MuestraNoEncontrada):
    """La verificación temprana (intensidad de fondo) indica que
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
# ESCALA DE GRISES (soporta GRAYSCALE y color, sin necesidad de
# saber de antemano qué pixformat se usó)
# ============================================================

def _a_escala_de_grises(imagen):
    """
    Devuelve la imagen en escala de grises, sea cual sea el formato
    con el que se haya capturado:
      - 2 dimensiones (H, W): ya viene en escala de grises
        (pixformat GRAYSCALE en una captura en vivo vía
        bytes_a_imagen()), se devuelve tal cual.
      - 3 dimensiones (H, W, 3): viene a color (RGB565, JPEG a color,
        o cualquier imagen cargada desde disco con cargar_imagen(),
        que cv2.imread() siempre fuerza a 3 canales), se convierte
        con cv2.cvtColor().

    Así las 4 métricas de abajo funcionan con cualquier pixformat sin
    necesidad de que quien las llama le pase el formato a mano — el
    array ya "sabe" si es color o no por su propia forma.
    """

    if imagen.ndim == 2:
        return imagen

    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)


# ============================================================
# MÉTRICAS
# ============================================================

def intensidad_fondo(imagen):

    gray = _a_escala_de_grises(imagen)

    return 255 - gray.mean()


def laplacian_variance(imagen):

    gray = _a_escala_de_grises(imagen)

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    return laplacian.var()


def tenengrad(imagen):

    gray = _a_escala_de_grises(imagen)

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

    gray = _a_escala_de_grises(imagen)

    return gray.var()


def bytes_a_imagen(image_data):

    imagen = Image.open(
        io.BytesIO(image_data)
    )

    imagen = np.array(imagen)

    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(
            imagen,
            cv2.COLOR_RGB2BGR
        )

    return imagen


# ============================================================
# MÉTRICAS DISPONIBLES PARA EL BARRIDO GRUESO / CONFIRMACIÓN
# ============================================================

FUNCIONES_METRICA = {
    "laplacian": laplacian_variance,
    "tenengrad": tenengrad,
    "varianza": varianza_intensidad,
}

CRITERIOS_POR_DEFECTO = {
    "laplacian": 12,
    "tenengrad": 200,
    "varianza": 30,
}


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
            # Tomar fotografía (se guarda si corresponde) y medir
            # --------------------------------------------

            valor = _capturar_y_medir()

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

                impresora.desp_lin_rel(0, 0, z_nuevo - z_actual)
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

                impresora.desp_lin_rel(0, 0, z_nuevo - z_actual)
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


# ============================================================
# EVALUAR CRITERIOS OR DE UNA IMAGEN
# ============================================================

def _evaluar_criterios(imagen, criterios):
    """
    Devuelve (valores, disparados):
      valores: {nombre_metrica: valor_calculado}
      disparados: lista de nombres de métrica que superaron su umbral
    """

    valores = {
        nombre: FUNCIONES_METRICA[nombre](imagen)
        for nombre in criterios
    }

    disparados = [
        nombre
        for nombre, umbral in criterios.items()
        if valores[nombre] >= umbral
    ]

    return valores, disparados


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
            impresora.desp_lin_rel(
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
# FASES 1 y 2: BARRIDO GRUESO + CONFIRMACIÓN ANTI-SEÑAL-FALSA
# ============================================================

def buscar_muestra_en_z(
    camara,
    impresora,
    z_arriba_seguro,
    z_min_seguridad,
    resolucion_global="QVGA",
    pixformat_global="GRAYSCALE",
    paso_grueso=0.005,
    paso_fino=0.0025,
    criterios_deteccion=None,
    margen_confirmacion=0.15,
    pasos_confirmacion=10,
    pasos_extension_maxima=3,
    verificar_presencia=False,
    alturas_verificacion=3,
    umbral_intensidad_fondo=152.5,
    espera_movimiento=0.5,
    devolveme_la_respuesta=False,
):
    """
    Devuelve un diccionario:
        {
            "z_confirmado": ...,
            "metrica_confirmada": "laplacian" | "tenengrad" | "varianza",
            "valor_confirmado": ...,
            "historial_barrido_grueso": [...],
        }

    Supuestos de seguridad (¡no tocar sin revisar el hardware!):
    - Z creciente = alejarse de la muestra (más seguro).
    - Z decreciente = acercarse a la muestra (más riesgo de choque).
    - 'z_min_seguridad' ya tiene que venir con el margen de seguridad
      incorporado respecto a la distancia de trabajo del objetivo
      (300 µm). Esta función NUNCA manda una posición menor a ese
      valor, y se lo pasa también a autofocus() para que tampoco lo
      cruce mientras confirma un pico.

    Fase 0 (opcional): verificar_presencia_muestra(). Si no hay
    indicios de muestra en este (x, y), corta acá con
    CampoVisualSinMuestra.

    Fase 1: barrido grueso con 'paso_grueso', evaluando Laplaciano,
    Tenengrad y Varianza de intensidad contra 'criterios_deteccion'
    como un OR (alcanza con que una sola supere su umbral).

    Fase 2: al primer disparo, corre autofocus() con la métrica que
    disparó, en el mismo resolucion_global/pixformat_global pero con
    'paso_fino', para encontrar y confirmar estructuralmente el máximo
    local (ver la docstring de autofocus() para el criterio de
    confirmación). Si autofocus() no logra confirmarlo
    (PicoNoConfirmado) o el máximo confirmado no supera umbral × (1 +
    margen_confirmacion), se descarta como señal falsa, se salta un
    paso_grueso más abajo del punto del disparo (para no volver a
    toparse con la misma señal) y se retoma el barrido. Si lo supera,
    queda confirmado (y ya posicionado ahí, porque autofocus() deja al
    equipo en su propio máximo).
    """

    if criterios_deteccion is None:
        criterios_deteccion = dict(CRITERIOS_POR_DEFECTO)

    if z_min_seguridad >= z_arriba_seguro:
        raise ValueError(
            "z_min_seguridad tiene que ser menor que z_arriba_seguro"
        )

    if paso_grueso <= 0 or paso_fino <= 0:
        raise ValueError("paso_grueso y paso_fino tienen que ser positivos")

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
    # Subir primero a la altura segura
    # --------------------------------------------------------

    desplazamiento_a_seguro = z_arriba_seguro - z_actual

    if desplazamiento_a_seguro != 0:
        impresora.desp_lin_rel(
            0, 0, desplazamiento_a_seguro,
            devolveme_la_respuesta=devolveme_la_respuesta,
        )
        time.sleep(espera_movimiento)

    # --------------------------------------------------------
    # Fase 0: verificación temprana
    # --------------------------------------------------------

    if verificar_presencia:

        hay_muestra = verificar_presencia_muestra(
            camara=camara,
            impresora=impresora,
            z_arriba_seguro=z_arriba_seguro,
            z_min_seguridad=z_min_seguridad,
            alturas_a_revisar=alturas_verificacion,
            umbral_intensidad_fondo=umbral_intensidad_fondo,
            resolucion=resolucion_global,
            espera_movimiento=espera_movimiento,
            devolveme_la_respuesta=devolveme_la_respuesta,
        )

        if not hay_muestra:
            raise CampoVisualSinMuestra(
                f"La verificación temprana (intensidad de fondo <= "
                f"{umbral_intensidad_fondo}) indica que este campo visual "
                f"no tiene glóbulos rojos."
            )

        # Volver a la altura segura antes de arrancar el barrido grueso
        _, _, z_actual = impresora.pregunta_posicion_actual()
        desplazamiento = z_arriba_seguro - z_actual

        if desplazamiento != 0:
            impresora.desp_lin_rel(
                0, 0, desplazamiento,
                devolveme_la_respuesta=devolveme_la_respuesta,
            )
            time.sleep(espera_movimiento)

    # --------------------------------------------------------
    # Fase 1 y 2: barrido grueso + confirmación
    # --------------------------------------------------------

    camara.config_de_una(
        pixformat=pixformat_global,
        resolucion=resolucion_global,
    )

    historial = []
    resultado_confirmado = None

    z = z_arriba_seguro

    while True:

        image_data = camara.capturar()
        imagen = bytes_a_imagen(image_data)

        valores, disparados = _evaluar_criterios(imagen, criterios_deteccion)

        historial.append({"z": impresora.posicion_z, **valores})

        print(
            f"Z = {impresora.posicion_z:.4f} mm | "
            + " | ".join(f"{n} = {v:.2f}" for n, v in valores.items())
        )

        if disparados:

            # Si dispararon varias, priorizo la que superó su umbral
            # por mayor margen relativo (más probable que sea señal real).
            nombre_disparo = max(
                disparados, key=lambda n: valores[n] / criterios_deteccion[n]
            )

            z_disparo = impresora.posicion_z

            print(
                f"  -> Candidato: '{nombre_disparo}' en Z = {z_disparo:.4f} mm "
                f"(valor = {valores[nombre_disparo]:.2f})"
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
                    vecinos_para_confirmar=3,
                    pasos_extension_maxima=pasos_extension_maxima,
                    z_min_seguridad=z_min_seguridad,
                    z_max_seguro=z_arriba_seguro,
                )

                valor_pico = max(valores_pico)

                umbral_confirmacion = criterios_deteccion[nombre_disparo] * (
                    1 + margen_confirmacion
                )

                if valor_pico >= umbral_confirmacion:

                    # autofocus() ya deja posicionado al equipo en
                    # z_pico (su propio máximo), no hace falta volver
                    # a moverse.

                    print(
                        f"  -> CONFIRMADO: {nombre_disparo} = {valor_pico:.2f} "
                        f">= {umbral_confirmacion:.2f}"
                    )

                    resultado_confirmado = {
                        "z_confirmado": z_pico,
                        "metrica_confirmada": nombre_disparo,
                        "valor_confirmado": valor_pico,
                    }
                    break

                else:
                    print(
                        f"  -> Señal falsa descartada: {nombre_disparo} = "
                        f"{valor_pico:.2f} (el máximo se confirmó "
                        f"estructuralmente, pero no alcanza el umbral de "
                        f"{umbral_confirmacion:.2f}). Se sigue bajando."
                    )

            except PicoNoConfirmado as motivo:

                print(
                    f"  -> Señal falsa descartada: '{nombre_disparo}' no "
                    f"tiene un máximo confirmable ({motivo}). Se sigue "
                    f"bajando."
                )

            # Señal falsa (por cualquiera de los dos motivos de
            # arriba): saltar un paso_grueso más abajo del punto del
            # disparo, para no volver a toparse con la misma señal, y
            # retomar el barrido grueso.

            z_siguiente = z_disparo - paso_grueso

            if z_siguiente < z_min_seguridad:
                break

            _, _, z_actual = impresora.pregunta_posicion_actual()

            impresora.desp_lin_rel(
                0, 0, z_siguiente - z_actual,
                devolveme_la_respuesta=devolveme_la_respuesta,
            )
            time.sleep(espera_movimiento)

            z = z_siguiente
            continue

        if z <= z_min_seguridad:
            break

        paso = -min(paso_grueso, z - z_min_seguridad)

        impresora.desp_lin_rel(
            0, 0, paso, devolveme_la_respuesta=devolveme_la_respuesta
        )
        time.sleep(espera_movimiento)

        z = impresora.posicion_z

    if resultado_confirmado is None:
        raise MuestraNoEncontrada(
            f"No se confirmó la muestra entre Z = {z_arriba_seguro} mm y "
            f"Z = {z_min_seguridad} mm en esta posición (x, y)."
        )

    resultado_confirmado["historial_barrido_grueso"] = historial

    return resultado_confirmado


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
    z_min_seguridad,
    z_max_seguro=None,
    paso_z=0.0025,
    pasos_arriba=10,
    pasos_abajo=10,
    vecinos_para_confirmar=3,
    pasos_extension_maxima=5,
    espera_movimiento=0.3,
    resolucion_grayscale="QVGA",
    pixformat_color="RGB565",
    resolucion_color="WQXGA2",  # alias: "4K"
    guardamos=False,
    carpeta_guardado=None,
    graficamos=True,
):
    """
    Pensada para usarse justo después de que buscar_muestra_en_z()
    confirma una muestra (es decir, arrancando ya cerca del plano de
    foco). Barre en Z UNA sola vez, en GRAYSCALE, evaluando las tres
    métricas disponibles (laplacian_variance, tenengrad,
    varianza_intensidad) sobre cada foto -- no hace tres barridos
    separados, reusa las mismas fotos para las tres.

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

    Después va a la altura del máximo de cada métrica, cambia a
    'pixformat_color'/'resolucion_color' (por defecto RGB565 a
    "WQXGA2", o sea "4K") y saca ahí la foto de mayor calidad. Si dos
    o tres métricas comparten el mismo Z (a 'paso_z' de diferencia o
    menos), se saca una sola foto para ese grupo y se la etiqueta con
    todas las métricas que coinciden, en vez de repetir la foto.

    Devuelve:
        {
            "laplacian": {"z": ..., "valor": ..., "confirmado": bool, "archivo": Path | None},
            "tenengrad": {...},
            "varianza":  {...},
            "posiciones_z": [...],          # el barrido GRAYSCALE completo
            "valores_por_metrica": {...},   # idem, las 3 listas
        }
    """

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
    # Barrido inicial en GRAYSCALE, evaluando las tres métricas
    # --------------------------------------------------------

    camara.config_de_una(pixformat="GRAYSCALE", resolucion=resolucion_grayscale)

    posiciones_z = []
    valores_por_metrica = {nombre: [] for nombre in FUNCIONES_METRICA}

    if pasos_abajo > 0:
        impresora.desp_lin_rel(0, 0, -pasos_abajo * paso_z)
        time.sleep(1)

    cantidad_fotos = pasos_arriba + pasos_abajo + 1

    for paso in range(cantidad_fotos):

        image_data = camara.capturar()
        imagen = bytes_a_imagen(image_data)

        posiciones_z.append(impresora.posicion_z)

        for nombre, funcion in FUNCIONES_METRICA.items():
            valores_por_metrica[nombre].append(funcion(imagen))

        if paso < cantidad_fotos - 1:
            impresora.desp_lin_rel(0, 0, paso_z)
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
            impresora.desp_lin_rel(0, 0, z_nuevo - z_actual)
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
            impresora.desp_lin_rel(0, 0, z_nuevo - z_actual)
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
            "archivo": None,
        }

    # --------------------------------------------------------
    # Fotos en alta calidad en cada máximo (agrupando coincidencias)
    # --------------------------------------------------------

    grupos = _agrupar_maximos_por_z(resultados_por_metrica, tolerancia_z=paso_z)

    for grupo in grupos:

        _, _, z_actual = impresora.pregunta_posicion_actual()
        impresora.desp_lin_rel(0, 0, grupo["z"] - z_actual)
        time.sleep(espera_movimiento)

        camara.config_de_una(pixformat=pixformat_color, resolucion=resolucion_color)

        image_data = camara.capturar()

        etiqueta_metricas = "+".join(grupo["metricas"])

        print(
            f"[foco_final_multimetrica] Foto en Z = "
            f"{impresora.posicion_z:.4f} mm para: {etiqueta_metricas}"
        )

        archivo = None

        if guardamos:

            nombre = (
                f"{etiqueta_metricas} "
                f"({np.round(impresora.posicion_x, 4)}, "
                f"{np.round(impresora.posicion_y, 4)}, "
                f"{np.round(impresora.posicion_z, 4)})"
            )

            archivo = camara.guardar_imagen(
                image_data=image_data,
                carpeta=carpeta_guardado,
                nombre=nombre,
                mostrar=False,
            )

        for nombre_metrica in grupo["metricas"]:
            resultados_por_metrica[nombre_metrica]["archivo"] = archivo

    # --------------------------------------------------------
    # Gráfico: las tres métricas comparten el mismo eje Z, porque
    # salieron del mismo barrido
    # --------------------------------------------------------

    if graficamos:

        fig, ax = plt.subplots()

        for nombre, valores in valores_por_metrica.items():

            ax.plot(posiciones_z, valores, marker="o", label=nombre)

            datos = resultados_por_metrica[nombre]

            ax.scatter(
                datos["z"],
                datos["valor"],
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

        plt.show()

    resultados_por_metrica["posiciones_z"] = posiciones_z
    resultados_por_metrica["valores_por_metrica"] = valores_por_metrica

    return resultados_por_metrica


# ============================================================
# MOVIMIENTO EN XY SEGURO: regla de las distancias cortas
# ============================================================
#
# Regla de seguridad (validada en el laboratorio): parado en el
# z_enfoque de un campo visual, un desplazamiento XY corto (hasta
# DISTANCIA_XY_SIN_SUBIR_MM, hoy 0.4 mm)
# no choca con la muestra, aunque esté inclinada (se supone que no
# demasiado). Para distancias mayores ya no se puede asegurar eso: una
# pendiente que sube puede quedar en el camino del objetivo (distancia
# de trabajo de 300 µm). Entonces, para viajes largos HAY QUE subir
# antes a una altura de seguridad más alta, moverse en XY, y recién
# después enfocar de nuevo (posicionar_en_muestra(), o el plano de
# enfoque aproximado + autofocus() para confirmar y ajustar la altura).
#
# Todo código nuevo que se mueva en XY entre campos visuales tiene que
# pasar por moverse_a_xy() en vez de llamar a desp_lin_abs() a mano.
#
# Historia del umbral: la regla se validó en el laboratorio con desplazamientos
# de ~150 µm (0.15 mm). El 2026-10-15 se subió a 0.4 mm a pedido de Franco.

DISTANCIA_XY_SIN_SUBIR_MM = 0.4


def moverse_a_xy(
    impresora,
    x,
    y,
    z_seguro,
    distancia_sin_subir=DISTANCIA_XY_SIN_SUBIR_MM,
    feedrate_xy=3000,
    espera_movimiento=0.3,
    devolveme_la_respuesta=False,
):
    """
    Lleva la platina a (x, y) aplicando la regla de arriba.

    - Distancia XY <= 'distancia_sin_subir' (default 0.4 mm): se
      mueve en XY manteniendo el Z actual (que es el z_enfoque del
      campo visual donde está parado).
    - Distancia mayor: si el Z actual es menor que 'z_seguro', primero
      sube hasta 'z_seguro' (si ya está más arriba, no baja); después
      se mueve en XY a esa altura, y se queda ahí.

    Devuelve True si fue un viaje largo (en ese caso la platina queda
    en una altura de viaje, SIN enfocar: quien llama tiene que
    reenfocar), o False si fue un desplazamiento corto (Z se mantiene,
    y alcanza con un autofocus() local).
    """

    x_actual = impresora.posicion_x
    y_actual = impresora.posicion_y
    _, _, z_actual = impresora.pregunta_posicion_actual()

    if x_actual is None or y_actual is None or z_actual is None:
        raise RuntimeError("Llamá primero a impresora.home()")

    distancia = float(np.hypot(x - x_actual, y - y_actual))

    viaje_largo = distancia > distancia_sin_subir + 1e-9

    if viaje_largo and z_actual < z_seguro:

        impresora.desp_lin_rel(
            0, 0, z_seguro - z_actual,
            devolveme_la_respuesta=devolveme_la_respuesta,
        )
        time.sleep(espera_movimiento)

    _, _, z_viaje = impresora.pregunta_posicion_actual()

    impresora.desp_lin_abs(
        x=x, y=y, z=z_viaje,
        feedrate=feedrate_xy,
        devolveme_la_respuesta=devolveme_la_respuesta,
    )
    time.sleep(espera_movimiento)

    return viaje_largo


# ============================================================
# ORQUESTADOR: (x, y) -> PLANO DE ENFOQUE IDEAL
# ============================================================

def posicionar_en_muestra(
    camara,
    impresora,
    x,
    y,
    z_arriba_seguro,
    z_min_seguridad,
    resolucion_global="QVGA",
    resolucion_local=None,
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
    espera_movimiento=0.5,
    hacer_foco_final=True,
    metrica_foco_final=None,
    pasos_arriba_foco_final=5,
    pasos_abajo_foco_final=5,
    guardar_fotos_foco_final=False,
    carpeta_fotos_foco_final=None,
    foco_final_multimetrica_activado=False,
    pixformat_color_foco_final="RGB565",
    resolucion_color_foco_final="WQXGA2",
    devolveme_la_respuesta=False,
):
    """
    Automatiza el flujo completo para una posición (x, y):

    1) Sube a altura segura y se mueve en XY.
    2) buscar_muestra_en_z(): verificación temprana + barrido grueso
       + confirmación anti-señal-falsa (ver esa función).
    3) (Opcional) foco final, en 'resolucion_local' (por defecto, la
       misma que 'resolucion_global'), alrededor del Z confirmado.
       Dos modos, elegidos con 'foco_final_multimetrica_activado':
         - False (default): autofocus() de una sola métrica (la
           misma que confirmó la muestra, salvo que se pase
           'metrica_foco_final'). Comportamiento de siempre.
         - True: foco_final_multimetrica() -- confirma el máximo de
           las tres métricas (laplacian, tenengrad, varianza) con un
           solo barrido, y saca una foto en color de alta resolución
           ('pixformat_color_foco_final'/'resolucion_color_foco_final')
           en cada una (agrupando las que coincidan en el mismo Z).

    Si 'guardar_fotos_foco_final' es True, las fotos del foco final
    se guardan dentro de 'carpeta_fotos_foco_final' -- en el modo de
    una sola métrica, en una subcarpeta nombrada con la posición
    donde arranca ese foco final: "(x, y, z)"; en el modo
    multimétrica, directamente ahí, nombradas con la/s métrica/s que
    confirman cada foto.
    """

    if resolucion_local is None:
        resolucion_local = resolucion_global

    # --------------------------------------------------------
    # 1) Altura segura antes de moverse en XY
    # --------------------------------------------------------

    _, _, z_actual = impresora.pregunta_posicion_actual()

    if z_actual is None:
        raise RuntimeError("Llamá primero a impresora.home()")

    if z_actual < z_arriba_seguro:
        impresora.desp_lin_rel(
            0, 0, z_arriba_seguro - z_actual,
            devolveme_la_respuesta=devolveme_la_respuesta,
        )
        time.sleep(espera_movimiento)

    impresora.desp_lin_abs(
        x=x, y=y, z=z_arriba_seguro,
        feedrate=feedrate_xy,
        devolveme_la_respuesta=devolveme_la_respuesta,
    )
    time.sleep(espera_movimiento)

    # --------------------------------------------------------
    # 2) Verificación + barrido grueso + confirmación
    # --------------------------------------------------------

    resultado_busqueda = buscar_muestra_en_z(
        camara=camara,
        impresora=impresora,
        z_arriba_seguro=z_arriba_seguro,
        z_min_seguridad=z_min_seguridad,
        resolucion_global=resolucion_global,
        pixformat_global=pixformat_global,
        paso_grueso=paso_grueso,
        paso_fino=paso_fino,
        criterios_deteccion=criterios_deteccion,
        margen_confirmacion=margen_confirmacion,
        pasos_confirmacion=pasos_confirmacion,
        verificar_presencia=verificar_presencia,
        alturas_verificacion=alturas_verificacion,
        umbral_intensidad_fondo=umbral_intensidad_fondo,
        espera_movimiento=espera_movimiento,
        devolveme_la_respuesta=devolveme_la_respuesta,
    )

    resultado = {
        "x": x,
        "y": y,
        "z_confirmado": resultado_busqueda["z_confirmado"],
        "metrica_confirmada": resultado_busqueda["metrica_confirmada"],
        "valor_confirmado": resultado_busqueda["valor_confirmado"],
        "historial_barrido_grueso": resultado_busqueda["historial_barrido_grueso"],
        "z_foco_final": None,
        "posiciones_z_foco_final": None,
        "valores_metrica_foco_final": None,
    }

    # --------------------------------------------------------
    # 3) Foco final en resolucion_local
    # --------------------------------------------------------

    if hacer_foco_final and foco_final_multimetrica_activado:

        carpeta_guardado_final = None

        if guardar_fotos_foco_final:

            if carpeta_fotos_foco_final is None:
                raise ValueError(
                    "Si 'guardar_fotos_foco_final' es True hay que "
                    "pasar 'carpeta_fotos_foco_final'."
                )

            carpeta_guardado_final = Path(carpeta_fotos_foco_final)

        resultado["foco_final_multimetrica"] = foco_final_multimetrica(
            camara=camara,
            impresora=impresora,
            z_min_seguridad=z_min_seguridad,
            z_max_seguro=z_arriba_seguro,
            paso_z=paso_fino,
            pasos_arriba=pasos_arriba_foco_final,
            pasos_abajo=pasos_abajo_foco_final,
            espera_movimiento=espera_movimiento,
            resolucion_grayscale=resolucion_local,
            pixformat_color=pixformat_color_foco_final,
            resolucion_color=resolucion_color_foco_final,
            guardamos=guardar_fotos_foco_final,
            carpeta_guardado=carpeta_guardado_final,
            graficamos=False,
        )

    elif hacer_foco_final:

        metrica_final = (
            metrica_foco_final
            or FUNCIONES_METRICA[resultado_busqueda["metrica_confirmada"]]
        )

        z_confirmado = resultado_busqueda["z_confirmado"]

        margen_disponible_abajo = z_confirmado - z_min_seguridad
        pasos_abajo_reales = max(
            min(pasos_abajo_foco_final, int(margen_disponible_abajo // paso_fino)),
            0,
        )

        camara.set_framesize(resolucion_local)
        # pixformat sigue siendo pixformat_global, no hace falta cambiarlo

        carpeta_guardado_final = None

        if guardar_fotos_foco_final:

            if carpeta_fotos_foco_final is None:
                raise ValueError(
                    "Si 'guardar_fotos_foco_final' es True hay que "
                    "pasar 'carpeta_fotos_foco_final'."
                )

            nombre_carpeta = (
                f"({impresora.posicion_x}, "
                f"{impresora.posicion_y}, "
                f"{impresora.posicion_z})"
            )

            carpeta_guardado_final = Path(carpeta_fotos_foco_final) / nombre_carpeta

        mejor_z, posiciones_z_final, valores_final = autofocus(
            camara=camara,
            impresora=impresora,
            metrica=metrica_final,
            pasos_arriba=pasos_arriba_foco_final,
            pasos_abajo=pasos_abajo_reales,
            paso_z=paso_fino,
            espera_movimiento=espera_movimiento,
            graficamos=False,
            guardamos=guardar_fotos_foco_final,
            carpeta_guardado=carpeta_guardado_final,
            z_min_seguridad=z_min_seguridad,
        )

        resultado["z_foco_final"] = mejor_z
        resultado["posiciones_z_foco_final"] = posiciones_z_final
        resultado["valores_metrica_foco_final"] = valores_final

    return resultado


# ============================================================
# EJEMPLO DE USO
# ============================================================

# resultado = posicionar_en_muestra(
#     camara=cam,
#     impresora=grua,
#     x=150.0,
#     y=100.0,
#     z_arriba_seguro=27.9,   # MAYOR Z = más lejos de la muestra (seguro)
#     z_min_seguridad=27.5,   # límite inferior, con margen del objetivo
#     resolucion_global="QVGA",
#     resolucion_local="QVGA",   # ajustar cuando la definas
#     hacer_foco_final=True,
# )
#
# print(resultado["z_confirmado"], resultado["metrica_confirmada"])
# print(resultado["z_foco_final"])


#%%
