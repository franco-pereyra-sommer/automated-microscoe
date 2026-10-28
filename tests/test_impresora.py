# %%

"""
Pruebas de impresora_cosmos.py con la clase REAL sobre un puerto serie
simulado (ver SerialFalso): límites, grilla de movimiento, G-code con
offsets y la regla de seguridad moverse_a_xy().

Creado el 2026-10-27.
"""

from simuladores import crear_impresora_falsa, movimientos_g1, sin_esperas

import impresora_cosmos


def _aprox(a, b, tolerancia=1e-6):
    return abs(a - b) <= tolerancia


def _impresora_en(x, y, z):
    impresora = crear_impresora_falsa()
    impresora.desp_lin_000(x, y, z)
    impresora.ser.escrituras.clear()
    return impresora


def test_home_deja_la_camara_en_0_0_40():
    impresora = crear_impresora_falsa()
    assert impresora.pregunta_posicion_actual() == (0, 0, 40)


_AVISO_LINEAS_SIN_LEER = (
    "Quedaron líneas de posición sin leer en el buffer: la próxima espera las tomaría "
    "como la respuesta de SU orden y volvería antes de que termine el movimiento. "
    "Si cambiaste cuántas líneas lee home() para G28, ajustá `lineas_por_g28` en "
    "SerialFalso (tests/simuladores.py) a lo que hayas medido en el equipo."
)


def test_home_no_deja_lineas_de_posicion_sin_leer():
    impresora = crear_impresora_falsa()

    assert impresora.ser.in_waiting == 0, _AVISO_LINEAS_SIN_LEER


def test_cada_movimiento_consume_solo_la_respuesta_de_su_propia_orden():
    impresora = crear_impresora_falsa()

    impresora.desp_lin_000(100, 100, 27.5)
    assert impresora.ser.in_waiting == 0, _AVISO_LINEAS_SIN_LEER

    impresora.desp_lin_cam(0, 0, 0.4)
    assert impresora.ser.in_waiting == 0, _AVISO_LINEAS_SIN_LEER

    impresora.desp_lin_cam(0.025, -0.0125, 0)
    assert impresora.ser.in_waiting == 0, _AVISO_LINEAS_SIN_LEER


def test_home_deja_el_buffer_alineado_aunque_g28_conteste_una_sola_linea():
    # Con 1 sola línea por G28 el home() vigente (que lee dos) espera su segundo
    # esperar_fin_movimiento() hasta el timeout. Se acorta el timeout para no
    # frenar la prueba; lo que importa es que no quede nada sin leer ni falten líneas.
    impresora = crear_impresora_falsa(hacer_home=False, lineas_por_g28=1)
    espera_original = impresora.esperar_fin_movimiento
    impresora.esperar_fin_movimiento = lambda timeout=0.05: espera_original(timeout=0.05)

    with sin_esperas():
        impresora.home()

    assert impresora.ser.in_waiting == 0, _AVISO_LINEAS_SIN_LEER
    assert impresora.pregunta_posicion_actual() == (0, 0, 40)


def test_grilla_de_movimiento_redondea_hacia_abajo():
    impresora = crear_impresora_falsa()

    assert _aprox(impresora.ajustar_movimiento(0.004, "z"), 0.0025)
    assert _aprox(impresora.ajustar_movimiento(0.0124, "x"), 0.0)
    assert _aprox(impresora.ajustar_movimiento(0.0125, "x"), 0.0125)  # múltiplo exacto: no baja un paso
    assert _aprox(impresora.ajustar_movimiento(10.5, "z"), 10.5)


def test_absoluto_manda_g90_y_suma_los_offsets():
    impresora = _impresora_en(100, 100, 27.5)

    impresora.desp_lin_000(110, 120, 28.0, feedrate=2000)

    assert impresora.ser.escrituras[0] == "G90"
    xy, z = movimientos_g1(impresora)
    assert _aprox(xy["X"], 110 + impresora.offset_x) and _aprox(xy["Y"], 120 + impresora.offset_y)
    assert _aprox(z["Z"], 28.0 + impresora.offset_z) and xy["F"] == z["F"] == 2000
    assert impresora.pregunta_posicion_actual() == (110, 120, 28.0)


def test_relativo_manda_g91_y_actualiza_la_posicion():
    impresora = _impresora_en(100, 100, 27.5)

    impresora.desp_lin_cam(0, 0, 0.4)

    assert impresora.ser.escrituras[0] == "G91"
    assert [m.get("Z") for m in movimientos_g1(impresora)] == [None, 0.4]
    assert impresora.pregunta_posicion_actual() == (100, 100, 27.9)


def test_un_movimiento_fuera_de_rango_se_cancela_sin_mandar_nada():
    impresora = _impresora_en(100, 100, 27.5)

    assert impresora.desp_lin_000(impresora.max_posicion_x + 1, 0, 30) is None

    assert impresora.ser.escrituras == []
    assert impresora.pregunta_posicion_actual() == (100, 100, 27.5)


def test_un_movimiento_relativo_que_termina_fuera_de_rango_tambien_se_cancela():
    impresora = _impresora_en(100, 100, 27.5)

    assert impresora.desp_lin_cam(0, 0, impresora.max_posicion_z) is None   # terminaría en z = 177.5
    assert impresora.desp_lin_cam(0, 0, -28) is None                        # terminaría en z = -0.5

    assert impresora.ser.escrituras == []
    assert impresora.pregunta_posicion_actual() == (100, 100, 27.5)


def test_moverse_a_xy_corto_mantiene_z_y_no_sube():
    impresora = _impresora_en(100, 100, 27.5)

    with sin_esperas():
        fue_viaje_largo = impresora_cosmos.moverse_a_xy(impresora, 100.3, 100, z_seguro=27.9)

    assert fue_viaje_largo is False
    assert "G91" not in impresora.ser.escrituras
    assert impresora.pregunta_posicion_actual() == (100.3, 100, 27.5)


def test_moverse_a_xy_en_el_umbral_exacto_sigue_siendo_corto():
    impresora = _impresora_en(100, 100, 27.5)

    with sin_esperas():
        fue_viaje_largo = impresora_cosmos.moverse_a_xy(
            impresora, 100 + impresora_cosmos.DISTANCIA_XY_SIN_SUBIR_MM, 100, z_seguro=27.9
        )

    assert fue_viaje_largo is False


def test_moverse_a_xy_largo_sube_primero_a_z_seguro_y_no_enfoca():
    impresora = _impresora_en(100, 100, 27.5)

    with sin_esperas():
        fue_viaje_largo = impresora_cosmos.moverse_a_xy(impresora, 105, 100, z_seguro=27.9)

    assert fue_viaje_largo is True
    modos = [c for c in impresora.ser.escrituras if c in ("G90", "G91")]
    assert modos == ["G91", "G90"]  # primero sube (relativo), después viaja (absoluto)
    assert impresora.pregunta_posicion_actual() == (105, 100, 27.9)


def test_moverse_a_xy_largo_no_baja_si_ya_esta_mas_arriba_que_z_seguro():
    impresora = _impresora_en(100, 100, 28.5)

    with sin_esperas():
        fue_viaje_largo = impresora_cosmos.moverse_a_xy(impresora, 105, 100, z_seguro=27.9)

    assert fue_viaje_largo is True
    assert "G91" not in impresora.ser.escrituras
    assert impresora.pregunta_posicion_actual() == (105, 100, 28.5)


def test_moverse_a_xy_exige_haber_hecho_home():
    impresora = crear_impresora_falsa(hacer_home=False)
    impresora.posicion_x = impresora.posicion_y = impresora.posicion_z = None

    try:
        impresora_cosmos.moverse_a_xy(impresora, 1, 1, z_seguro=27.9)
    except RuntimeError:
        pass
    else:
        raise AssertionError("sin home() debía tirar RuntimeError")
