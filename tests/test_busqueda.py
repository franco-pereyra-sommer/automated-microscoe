# %%

"""
Pruebas de busqueda_muestra.py (el módulo que combina cámara + impresora)
con la impresora real sobre serie simulado y una cámara falsa.

Falta simular (pendiente): buscar_muestra_en_z, foco_final_multimetrica y
posicionar_en_muestra de punta a punta. Mientras tanto, test_estatico.py
atrapa los nombres sin definir en esas funciones.

Creado el 2026-10-27.
"""

from simuladores import CamaraFalsa, crear_impresora_falsa, sin_esperas

import busqueda_muestra
import camara_openmv
import impresora_cosmos
import metricas_imagen

PASO_Z = 0.0025

# Dónde vive la ÚNICA definición de lo que se movió de busqueda_muestra.py.
# Si busqueda_muestra (u otro módulo) tiene un objeto con ese nombre, tiene que
# ser ese mismo objeto: nada de copias ni versiones paralelas.
DUENOS = {
    "bytes_a_imagen": "camara_openmv",
    "moverse_a_xy": "impresora_cosmos",
    "DISTANCIA_XY_SIN_SUBIR_MM": None,  # es un número: se compara el valor
    "intensidad_fondo": "metricas_imagen",
    "laplacian_variance": "metricas_imagen",
    "tenengrad": "metricas_imagen",
    "varianza_intensidad": "metricas_imagen",
    "evaluar_criterios": "metricas_imagen",
    "FUNCIONES_METRICA": None,
    "CRITERIOS_POR_DEFECTO": None,
}

MODULOS = {
    "camara_openmv": camara_openmv,
    "impresora_cosmos": impresora_cosmos,
    "metricas_imagen": metricas_imagen,
}


def test_lo_que_se_movio_tiene_una_sola_definicion():
    for nombre, dueno in DUENOS.items():
        if not hasattr(busqueda_muestra, nombre):
            continue

        objeto = getattr(busqueda_muestra, nombre)

        if dueno is not None:
            assert objeto is getattr(MODULOS[dueno], nombre), f"{nombre} está duplicada en busqueda_muestra"
        elif nombre == "FUNCIONES_METRICA":
            assert objeto is metricas_imagen.FUNCIONES_METRICA
        elif nombre == "CRITERIOS_POR_DEFECTO":
            assert objeto is metricas_imagen.CRITERIOS_POR_DEFECTO
        else:
            assert objeto == getattr(impresora_cosmos, nombre)


def test_autofocus_encuentra_el_plano_de_foco_con_cada_metrica():
    z_inicial = 27.5
    z_foco = z_inicial + 2 * PASO_Z

    for nombre, metrica in busqueda_muestra.FUNCIONES_METRICA.items():
        impresora = crear_impresora_falsa()
        impresora.desp_lin_000(100, 100, z_inicial)
        camara = CamaraFalsa(impresora, z_foco=z_foco)

        with sin_esperas():
            mejor_z, posiciones_z, valores = busqueda_muestra.autofocus(
                camara=camara,
                impresora=impresora,
                metrica=metrica,
                pasos_arriba=6,
                pasos_abajo=6,
                paso_z=PASO_Z,
                graficamos=False,
                z_min_seguridad=z_inicial - 0.05,
                z_max_seguro=z_inicial + 0.05,
            )

        assert abs(mejor_z - z_foco) <= PASO_Z + 1e-9, f"{nombre}: foco en {mejor_z}, esperado {z_foco}"
        assert len(posiciones_z) == len(valores) > 5


def test_autofocus_no_pasa_de_los_limites_de_seguridad():
    z_inicial = 27.5
    impresora = crear_impresora_falsa()
    impresora.desp_lin_000(100, 100, z_inicial)
    camara = CamaraFalsa(impresora, z_foco=z_inicial)
    z_min = z_inicial - 2 * PASO_Z

    alturas = []  # Z absoluta después de cada desplazamiento relativo
    desp_lin_cam_original = impresora.desp_lin_cam

    def espiar_desp_lin_cam(*args, **kwargs):
        resultado = desp_lin_cam_original(*args, **kwargs)
        alturas.append(impresora.posicion_z)
        return resultado

    impresora.desp_lin_cam = espiar_desp_lin_cam

    # Con tan poco margen hacia abajo el máximo no se puede confirmar: autofocus
    # tiene que negarse (PicoNoConfirmado) y dejar la platina donde estaba,
    # sin cruzar nunca z_min_seguridad.
    try:
        with sin_esperas():
            busqueda_muestra.autofocus(
                camara=camara,
                impresora=impresora,
                metrica=busqueda_muestra.FUNCIONES_METRICA["varianza"],
                pasos_arriba=3,
                pasos_abajo=10,  # pide bajar mucho más de lo permitido
                paso_z=PASO_Z,
                graficamos=False,
                z_min_seguridad=z_min,
                z_max_seguro=z_inicial + 0.05,
            )
    except busqueda_muestra.PicoNoConfirmado:
        pass

    assert alturas and min(alturas) >= z_min - 1e-9, f"bajó hasta {min(alturas)}, el límite era {z_min}"
    assert abs(impresora.posicion_z - z_inicial) < 1e-9, "tras interrumpirse tiene que restaurar la Z inicial"
