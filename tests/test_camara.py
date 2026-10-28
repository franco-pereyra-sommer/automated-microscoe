# %%

"""
Pruebas de camara_openmv.py: decodificación de la captura, mostrar() y el
protocolo serie (contra un firmware simulado, ver SerialCamaraFalso).

Creado el 2026-10-27.
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
