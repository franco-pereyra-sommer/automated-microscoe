# %%

"""
Pruebas de camara_openmv.py: decodificación de la captura, mostrar() y el
protocolo serie (contra un firmware simulado, ver SerialCamaraFalso).

Creado el 2026-10-27. 2026-11-09: pruebas de config_de_una() con el diccionario de
parámetros y de validar_dict_params_camara().
"""

from unittest import mock

import matplotlib.pyplot as plt

from simuladores import (
    SerialCamaraFalso, a_jpeg, np, sin_esperas,
)

import camara_openmv


def _imagen_de_prueba():
    generador = np.random.default_rng(0)
    color = generador.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    color[:, :32] = (255, 0, 0)  # mitad izquierda ROJA (en RGB)
    gris = generador.integers(0, 256, (48, 64), dtype=np.uint8)
    return a_jpeg(color, "RGB"), a_jpeg(gris, "L")


def test_bytes_a_imagen_color_es_bgr_y_gris_es_2d():
    jpeg_color, jpeg_gris = _imagen_de_prueba()

    color = camara_openmv.bytes_a_imagen(jpeg_color)
    gris = camara_openmv.bytes_a_imagen(jpeg_gris)

    assert color.shape == (48, 64, 3) and color.dtype == np.uint8
    assert gris.shape == (48, 64) and gris.dtype == np.uint8

    # El rojo (RGB 255,0,0) tiene que quedar con el canal 2 alto: eso es BGR.
    assert color[10, 5, 2] > 200 and color[10, 5, 0] < 60


def test_mostrar_entrega_rgb_a_matplotlib_y_gris_con_cmap_gray():
    jpeg_color, jpeg_gris = _imagen_de_prueba()
    camara = camara_openmv.OpenMVCamera()
    visto = {}

    def espiar(*args, **kwargs):
        imagen = plt.gca().images[0]
        visto["arreglo"] = np.asarray(imagen.get_array())
        visto["cmap"] = imagen.get_cmap().name

    with mock.patch.object(plt, "show", espiar):
        camara.mostrar(jpeg_color)
        assert visto["arreglo"].shape == (48, 64, 3)
        assert visto["arreglo"][10, 5, 0] > 200 and visto["arreglo"][10, 5, 2] < 60  # RGB: rojo en canal 0

        camara.mostrar(jpeg_gris)
        assert visto["arreglo"].ndim == 2 and visto["cmap"] == "gray"

    plt.close("all")


def _camara_conectada(serial_falso):
    camara = camara_openmv.OpenMVCamera()
    with sin_esperas(), mock.patch.object(
        camara_openmv.serial, "Serial", lambda *args, **kwargs: serial_falso
    ):
        camara.conectar()
    return camara


def test_los_comandos_viajan_con_el_formato_que_entiende_el_firmware():
    serial_falso = SerialCamaraFalso()
    camara = _camara_conectada(serial_falso)

    camara.set_pixformat("grayscale")
    camara.set_framesize("4K")  # alias de WQXGA2

    assert serial_falso.escrituras == ["set_pixformat GRAYSCALE", "set_framesize WQXGA2"]


def test_un_typo_falla_en_la_pc_antes_de_mandar_nada():
    serial_falso = SerialCamaraFalso()
    camara = _camara_conectada(serial_falso)

    try:
        camara.set_pixformat("JPG")
    except ValueError:
        pass
    else:
        raise AssertionError("set_pixformat('JPG') debía tirar ValueError")

    assert serial_falso.escrituras == []


def test_capturar_devuelve_los_bytes_del_firmware_e_info_los_parsea():
    _, jpeg_gris = _imagen_de_prueba()
    serial_falso = SerialCamaraFalso(jpeg=jpeg_gris)
    camara = _camara_conectada(serial_falso)

    assert camara.capturar() == jpeg_gris

    info = camara.info()
    assert info["exposure_us"] == 50000 and info["gain_db"] == 25.0
    assert info["rgb_gain_db"] == (1.0, 1.0, 1.0)


# ----------------------------------------------------------------------
# config_de_una(dict_params) y validar_dict_params_camara()  (2026-11-09)
# ----------------------------------------------------------------------

DICT_PARAMS = {
    "pixformat": "GRAYSCALE",
    "resolucion": "QVGA",
    "exposure_us": 50000,
    "gain_db": 25,
    "whitebal": (66.22236, 61.43028, 60.206),
}


def _con(**cambios):
    return {**DICT_PARAMS, **cambios}


def _sin(clave):
    return {k: v for k, v in DICT_PARAMS.items() if k != clave}


def test_config_de_una_manda_todos_los_parametros_del_diccionario_en_orden():
    serial_falso = SerialCamaraFalso()
    camara = _camara_conectada(serial_falso)

    camara.config_de_una(DICT_PARAMS, skip=0)

    assert serial_falso.escrituras == [
        "set_pixformat GRAYSCALE",
        "set_framesize QVGA",
        "set_quality 100",
        "set_exposure_auto 0 50000",
        "set_gain_auto 0 25",
        "set_whitebal_auto 0 66.22236 61.43028 60.206",
    ]


def test_config_de_una_descarta_skip_cuadros_y_usa_la_calidad_pedida():
    serial_falso = SerialCamaraFalso()
    camara = _camara_conectada(serial_falso)

    camara.config_de_una(_con(pixformat="RGB565", resolucion="4K"), calidad=80, skip=3)

    assert serial_falso.escrituras[:3] == ["set_pixformat RGB565", "set_framesize WQXGA2", "set_quality 80"]
    assert serial_falso.escrituras.count("capture") == 3

    # skip por defecto: 2 cuadros
    serial_falso.escrituras.clear()
    camara.config_de_una(DICT_PARAMS)
    assert serial_falso.escrituras.count("capture") == 2


def test_none_pide_automatico_y_hay_que_escribirlo_a_proposito():
    serial_falso = SerialCamaraFalso()
    camara = _camara_conectada(serial_falso)

    camara.config_de_una(_con(exposure_us=None, gain_db=None, whitebal=None), skip=0)

    assert serial_falso.escrituras[3:] == ["set_exposure_auto 1", "set_gain_auto 1", "set_whitebal_auto 1"]


def test_config_de_una_acepta_enums_y_no_modifica_el_diccionario():
    serial_falso = SerialCamaraFalso()
    camara = _camara_conectada(serial_falso)
    parametros = _con(pixformat=camara_openmv.PixFormat.RGB565, resolucion=camara_openmv.Framesize.QVGA)
    copia = dict(parametros)

    camara.config_de_una(parametros, skip=0)

    assert parametros == copia
    assert serial_falso.escrituras[:2] == ["set_pixformat RGB565", "set_framesize QVGA"]


def _tiene_que_fallar(llamada, excepcion, contiene=None):
    try:
        llamada()
    except excepcion as error:
        if contiene is not None:
            assert contiene in str(error), str(error)
    else:
        raise AssertionError(f"tenía que tirar {excepcion.__name__}")


def test_un_diccionario_malo_falla_en_la_pc_antes_de_mandar_nada():
    serial_falso = SerialCamaraFalso()
    camara = _camara_conectada(serial_falso)

    casos = [
        # (diccionario, excepción, texto que tiene que aparecer en el mensaje)
        (_sin("exposure_us"), ValueError, "exposure_us"),                       # falta una clave
        (_sin("whitebal"), ValueError, "whitebal"),
        ({}, ValueError, "faltan"),
        (_con(exposure="50000"), ValueError, "exposure"),                       # una clave que no existe (typo)
        (_con(pixformat="JPG"), ValueError, "pixformat"),
        (_con(resolucion="VGAA"), ValueError, "resolucion"),
        (_con(pixformat=3), TypeError, "pixformat"),
        (_con(exposure_us=0), ValueError, "exposure_us"),
        (_con(exposure_us=-5), ValueError, "exposure_us"),
        (_con(exposure_us="50000"), ValueError, "exposure_us"),
        (_con(exposure_us=True), ValueError, "exposure_us"),
        (_con(exposure_us=float("nan")), ValueError, "exposure_us"),
        (_con(gain_db="25"), ValueError, "gain_db"),
        (_con(whitebal=(66.2, 61.4)), ValueError, "whitebal"),                  # faltan canales
        (_con(whitebal=(66.2, 61.4, 60.2, 1.0)), ValueError, "whitebal"),
        (_con(whitebal="rgb"), ValueError, "whitebal"),
        (_con(whitebal=66.2), ValueError, "whitebal"),
    ]

    for parametros, excepcion, texto in casos:
        _tiene_que_fallar(lambda: camara.config_de_una(parametros), excepcion, texto)

    assert serial_falso.escrituras == [], serial_falso.escrituras


def test_config_de_una_ya_no_acepta_el_formato_viejo_ni_valores_por_defecto():
    serial_falso = SerialCamaraFalso()
    camara = _camara_conectada(serial_falso)

    _tiene_que_fallar(lambda: camara.config_de_una(), TypeError)                                  # sin nada
    _tiene_que_fallar(lambda: camara.config_de_una("RGB565", "QVGA"), TypeError, "diccionario")   # posicionales
    _tiene_que_fallar(lambda: camara.config_de_una(pixformat="RGB565", resolucion="QVGA"), TypeError)
    _tiene_que_fallar(lambda: camara.config_de_una(DICT_PARAMS, whitebal=None), TypeError)

    assert serial_falso.escrituras == []


def test_validar_dict_params_camara_usa_el_nombre_que_se_le_da_en_el_mensaje():
    _tiene_que_fallar(
        lambda: camara_openmv.validar_dict_params_camara(_sin("gain_db"), "dict_params_color"),
        ValueError,
        "dict_params_color",
    )

    assert camara_openmv.validar_dict_params_camara(DICT_PARAMS) is None
    assert set(DICT_PARAMS) == set(camara_openmv.CLAVES_DICT_PARAMS_CAMARA)
