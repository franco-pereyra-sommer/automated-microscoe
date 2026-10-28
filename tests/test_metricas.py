# %%

"""
Pruebas de metricas_imagen.py (procesamiento de imagen sin hardware).

Creado el 2026-10-27.
"""

from simuladores import cv2, np

import metricas_imagen as m


def _textura(semilla=0):
    return np.random.default_rng(semilla).integers(0, 256, (60, 80), dtype=np.uint8)


def test_imagen_lisa_no_tiene_variacion():
    lisa = np.full((60, 80), 100, np.uint8)

    assert m.laplacian_variance(lisa) == 0
    assert m.tenengrad(lisa) == 0
    assert m.varianza_intensidad(lisa) == 0


def test_intensidad_fondo_es_255_menos_el_promedio():
    assert m.intensidad_fondo(np.full((10, 10), 255, np.uint8)) == 0
    assert m.intensidad_fondo(np.zeros((10, 10), np.uint8)) == 255
    assert m.intensidad_fondo(np.full((10, 10), 100, np.uint8)) == 155


def test_una_imagen_borrosa_puntua_menos_que_la_nitida_en_las_tres_metricas():
    nitida = _textura()
    borrosa = cv2.GaussianBlur(nitida, (0, 0), 2.0)

    for nombre, funcion in m.FUNCIONES_METRICA.items():
        assert funcion(nitida) > funcion(borrosa), nombre


def test_escala_de_grises_respeta_2d_y_trata_el_color_como_bgr():
    gris = _textura()
    assert m._a_escala_de_grises(gris) is gris

    azul_bgr = np.zeros((4, 4, 3), np.uint8)
    azul_bgr[..., 0] = 255  # canal 0 = azul en BGR
    resultado = m._a_escala_de_grises(azul_bgr)

    assert resultado.ndim == 2
    assert abs(int(resultado[0, 0]) - 29) <= 1  # 0.114 * 255; si lo tratara como RGB daría ~76


def test_las_metricas_funcionan_igual_con_la_misma_imagen_en_gris_o_en_color():
    gris = _textura()
    color = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)

    for nombre, funcion in m.FUNCIONES_METRICA.items():
        assert abs(funcion(gris) - funcion(color)) < 1e-6 * max(1.0, abs(funcion(gris))), nombre


def test_evaluar_criterios_devuelve_valores_y_los_que_superan_el_umbral():
    ruido = _textura()
    lisa = np.full((60, 80), 100, np.uint8)
    criterios = {"varianza": 30, "laplacian": 12}

    valores, disparados = m.evaluar_criterios(ruido, criterios)
    assert set(valores) == {"varianza", "laplacian"} and sorted(disparados) == ["laplacian", "varianza"]

    valores, disparados = m.evaluar_criterios(lisa, criterios)
    assert disparados == []


def test_los_criterios_por_defecto_cubren_exactamente_las_metricas_disponibles():
    assert set(m.CRITERIOS_POR_DEFECTO) == set(m.FUNCIONES_METRICA)
