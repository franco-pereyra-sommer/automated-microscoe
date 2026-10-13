#%%



# Agrego la raiz del proyecto
import sys
import os

os.environ["PYTHONPATH"] = r"C:\BackUp\Facultad\Labo 7\env_labo7"

sys.path.insert(0, os.environ["PYTHONPATH"])


import time
import numpy as np

from Main import (
    OpenMVCamera,
    TrimakerCOSMOS
)

import cv2
import re
import time
import matplotlib.pyplot as plt

from metricas_enfoque import (
    intensidad_fondo,
    laplacian_variance,
    tenengrad,
    varianza_intensidad,
    bytes_a_imagen,
    autofocus
)



#%%



grua = TrimakerCOSMOS("COM11")

grua.escuchar(2)
time.sleep(1)

grua.home(devolveme_timepos=True)

grua.escuchar(2)
time.sleep(1)



#%%



cam = OpenMVCamera("COM7")

cam.conectar()

cam.set_framesize("HD") 

cam.set_pixformat("RGB565") 

cam.set_pixformat("JPEG") 

cam.set_exposure_auto()

cam.set_gain_auto()



#%%



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


# ============================================================
# METRICAS DISPONIBLES PARA EL BARRIDO GRUESO / CONFIRMACIÓN
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
# FRACCION DE PIXELES ROJOS (UTILIDAD SECUNDARIA / DIAGNÓSTICO)
# ============================================================
# No forma parte del criterio de detección actual (ver
# intensidad_fondo() y el barrido grueso más abajo), se deja
# disponible por si hace falta más adelante.

def fraccion_pixeles_rojos(imagen_bgr, s_min=50, v_min=50, v_max=255):

    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)

    rango_bajo_1 = np.array([0, s_min, v_min])
    rango_alto_1 = np.array([10, 255, v_max])

    rango_bajo_2 = np.array([160, s_min, v_min])
    rango_alto_2 = np.array([180, 255, v_max])

    mascara_1 = cv2.inRange(hsv, rango_bajo_1, rango_alto_1)
    mascara_2 = cv2.inRange(hsv, rango_bajo_2, rango_alto_2)

    mascara = cv2.bitwise_or(mascara_1, mascara_2)

    return mascara.mean() / 255.0


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
    paso_grueso=0.01,
    paso_fino=0.0025,
    criterios_deteccion=None,
    margen_confirmacion=0.15,
    pasos_confirmacion=6,
    verificar_presencia=True,
    alturas_verificacion=3,
    umbral_intensidad_fondo=152.5,
    espera_movimiento=0.3,
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
      (300 µm). Esta función NUNCA manda una posición menor a ese valor,
      ni tampoco lo hace autofocus() internamente (se le acota
      'pasos_abajo' a mano en cada llamada para garantizar esto).

    Fase 0 (opcional): verificar_presencia_muestra(). Si no hay
    indicios de muestra en este (x, y), corta acá con
    CampoVisualSinMuestra.

    Fase 1: barrido grueso con 'paso_grueso', evaluando Laplaciano,
    Tenengrad y Varianza de intensidad contra 'criterios_deteccion'
    como un OR (alcanza con que una sola supere su umbral).

    Fase 2: al primer disparo, corre autofocus() con la métrica que
    disparó, en el mismo resolucion_global/pixformat_global pero con
    'paso_fino', para encontrar el máximo local. Si ese máximo no
    supera umbral × (1 + margen_confirmacion), se descarta como señal
    falsa, se salta un paso_grueso más abajo del punto del disparo
    (para no volver a toparse con la misma señal) y se retoma el
    barrido. Si lo supera, queda confirmado (y ya posicionado ahí,
    porque autofocus() deja al equipo en su propio máximo).
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

    camara.set_pixformat(pixformat_global)
    camara.set_framesize(resolucion_global)

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

            z_pico, _, valores_pico = autofocus(
                camara=camara,
                impresora=impresora,
                metrica=metrica_funcion,
                pasos_arriba=pasos_confirmacion,
                pasos_abajo=pasos_abajo_confirmacion,
                paso_z=paso_fino,
                espera_movimiento=espera_movimiento,
                graficamos=False,
            )

            valor_pico = max(valores_pico)

            umbral_confirmacion = criterios_deteccion[nombre_disparo] * (
                1 + margen_confirmacion
            )

            if valor_pico >= umbral_confirmacion:

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
                    f"{valor_pico:.2f} < {umbral_confirmacion:.2f}. "
                    f"Se sigue bajando."
                )

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
    paso_grueso=0.01,
    paso_fino=0.0025,
    criterios_deteccion=None,
    margen_confirmacion=0.15,
    pasos_confirmacion=10,
    verificar_presencia=True,
    alturas_verificacion=3,
    umbral_intensidad_fondo=152.5,
    feedrate_xy=3000,
    espera_movimiento=0.3,
    hacer_foco_final=True,
    metrica_foco_final=None,
    pasos_arriba_foco_final=10,
    pasos_abajo_foco_final=10,
    devolveme_la_respuesta=False,
):
    """
    Automatiza el flujo completo para una posición (x, y):

    1) Sube a altura segura y se mueve en XY.
    2) buscar_muestra_en_z(): verificación temprana + barrido grueso
       + confirmación anti-señal-falsa (ver esa función).
    3) (Opcional) foco final: autofocus() en 'resolucion_local'
       (por defecto, la misma que 'resolucion_global' — todavía no
       definida), alrededor del Z confirmado, reusando por defecto
       la misma métrica que confirmó la muestra.
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

    if hacer_foco_final:

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

        mejor_z, posiciones_z_final, valores_final = autofocus(
            camara=camara,
            impresora=impresora,
            metrica=metrica_final,
            pasos_arriba=pasos_arriba_foco_final,
            pasos_abajo=pasos_abajo_reales,
            paso_z=paso_fino,
            espera_movimiento=espera_movimiento,
            graficamos=False,
        )

        resultado["z_foco_final"] = mejor_z
        resultado["posiciones_z_foco_final"] = posiciones_z_final
        resultado["valores_metrica_foco_final"] = valores_final

    return resultado



#%%



# ============================================================
# EJEMPLO DE USO
# ============================================================

resultado = posicionar_en_muestra(
    camara=cam,
    impresora=grua,
    x=150.0,
    y=150.0,
    z_arriba_seguro=28.0,
    z_min_seguridad=27.7,
    resolucion_global="QVGA",
    resolucion_local="WQXGA2",   # ajustar cuando la definas
    metrica_foco_final=tenengrad,
    pixformat_global="RGB565",
    verificar_presencia=False,
    hacer_foco_final=True,
)

cam.mostrar(cam.capturar)

print(resultado["z_confirmado"], resultado["metrica_confirmada"])
print(resultado["z_foco_final"])




