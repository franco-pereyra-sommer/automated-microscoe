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


# ----------------------------------------------------------------------
# Detección relativa a una línea de base (2026-11-09)
# ----------------------------------------------------------------------

def _fotos(**series):
    """_fotos(varianza=[1, 1, 5], laplacian=[2, 2, 2]) -> [{"varianza": 1, "laplacian": 2}, ...]"""
    cantidad = len(next(iter(series.values())))
    return [{nombre: valores[i] for nombre, valores in series.items()} for i in range(cantidad)]


def test_validar_aumentos_acepta_un_subconjunto_de_metricas_y_rechaza_lo_demas():
    m.validar_aumentos({"varianza": 0.3})
    m.validar_aumentos({"laplacian": 1, "tenengrad": 0.25, "varianza": 2.5})

    casos = [
        (None, TypeError), ("varianza", TypeError), ([("varianza", 0.3)], TypeError),
        ({}, ValueError), ({"foco": 0.3}, ValueError), ({"varianza": 0}, ValueError),
        ({"varianza": -1}, ValueError), ({"varianza": "0.3"}, ValueError), ({"varianza": None}, ValueError),
        ({"varianza": True}, ValueError), ({"varianza": float("inf")}, ValueError),
        ({"varianza": float("nan")}, ValueError), ({"varianza": 0.3, "foco": 0.3}, ValueError),
    ]

    for malo, excepcion in casos:
        try:
            m.validar_aumentos(malo)
        except excepcion:
            pass
        else:
            raise AssertionError(f"{malo!r} tenía que tirar {excepcion.__name__}")


def test_linea_de_base_es_el_promedio_de_las_primeras_fotos():
    fotos = _fotos(varianza=[2, 4, 6, 100], laplacian=[1, 1, 1, 100])

    assert m.linea_de_base(fotos, ["varianza", "laplacian"], pasos_linea_base=3) == {"varianza": 4, "laplacian": 1}
    assert m.linea_de_base(fotos, {"varianza": 0.3}, pasos_linea_base=2) == {"varianza": 3}     # acepta un dict
    assert m.linea_de_base(fotos[:2], ["varianza"], pasos_linea_base=3) is None                  # todavía no hay tantas
    assert m.linea_de_base(fotos, ["varianza"], pasos_linea_base=1) == {"varianza": 2}

    for malo in (0, -1, 2.5, True, None):
        try:
            m.linea_de_base(fotos, ["varianza"], pasos_linea_base=malo)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(malo)


def test_detectar_aumento_no_evalua_las_fotos_de_la_linea_de_base_ni_detecta_sin_ella():
    # las 3 primeras son la línea de base (promedio 10): aunque la 3ª sea alta, no cuenta como detección
    fotos = _fotos(varianza=[10, 10, 10, 10, 10])
    assert m.detectar_aumento(fotos, {"varianza": 0.5}, pasos_linea_base=3, fotos_seguidas=1) is None
    assert m.detectar_aumento(fotos[:2], {"varianza": 0.5}, pasos_linea_base=3, fotos_seguidas=1) is None

    fotos = _fotos(varianza=[10, 10, 10, 40, 40])
    deteccion = m.detectar_aumento(fotos, {"varianza": 0.5}, pasos_linea_base=3, fotos_seguidas=1)
    assert deteccion["indice"] == 3 and deteccion["linea_base"] == {"varianza": 10}
    assert deteccion["umbrales"] == {"varianza": 15}

    # si el foco ya está dentro de la línea de base, el promedio sube y el umbral con él: no detecta
    # (por eso z_arriba_seguro tiene que estar lejos del foco)
    contaminada = _fotos(varianza=[10, 40, 40, 40, 40])      # promedio 30 -> umbral 45
    assert m.detectar_aumento(contaminada, {"varianza": 0.5}, pasos_linea_base=3, fotos_seguidas=1) is None


def test_detectar_aumento_el_umbral_es_estricto_y_pide_fotos_seguidas():
    base = [10, 10, 10]
    umbral = 15.0          # 10 * (1 + 0.5)

    # justo en el umbral no alcanza; apenas encima, sí
    assert m.detectar_aumento(_fotos(varianza=base + [umbral]), {"varianza": 0.5}, 3, 1) is None
    assert m.detectar_aumento(_fotos(varianza=base + [umbral + 0.01]), {"varianza": 0.5}, 3, 1)["indice"] == 3

    # un pico aislado no detecta con 2 seguidas; dos juntas, sí, y se detecta en la SEGUNDA
    pico = _fotos(varianza=base + [20, 10, 20, 10, 20])
    assert m.detectar_aumento(pico, {"varianza": 0.5}, 3, 2) is None
    assert m.detectar_aumento(pico, {"varianza": 0.5}, 3, 1)["indice"] == 3

    seguidas = _fotos(varianza=base + [20, 10, 20, 20, 10])
    assert m.detectar_aumento(seguidas, {"varianza": 0.5}, 3, 2)["indice"] == 6
    assert m.detectar_aumento(_fotos(varianza=base + [20, 20, 20]), {"varianza": 0.5}, 3, 3)["indice"] == 5


def test_detectar_aumento_con_varias_metricas_elige_la_que_mas_paso_su_umbral():
    fotos = _fotos(varianza=[10, 10, 10, 30, 30], laplacian=[1, 1, 1, 8, 8], tenengrad=[5, 5, 5, 5, 5])
    aumentos = {"varianza": 0.5, "laplacian": 0.5, "tenengrad": 0.5}

    deteccion = m.detectar_aumento(fotos, aumentos, pasos_linea_base=3, fotos_seguidas=2)

    # varianza: 30 / 15 = 2; laplacian: 8 / 1.5 = 5.33; tenengrad no pasó
    assert deteccion["indice"] == 4 and deteccion["metrica"] == "laplacian"
    assert sorted(deteccion["disparadas"]) == ["laplacian", "varianza"]
    assert deteccion["valor"] == 8 and deteccion["umbrales"]["tenengrad"] == 7.5

    # solo mira las métricas de 'aumentos' (y tolera claves de más, como "z" o "fase")
    fotos_con_extras = [{**f, "z": 27.9 - 0.005 * i, "fase": "x"} for i, f in enumerate(fotos)]
    assert m.detectar_aumento(fotos_con_extras, {"tenengrad": 0.5}, 3, 1) is None
    assert m.detectar_aumento(fotos_con_extras, {"varianza": 0.5}, 3, 2)["metrica"] == "varianza"


def test_detectar_aumento_retoma_desde_un_indice_y_cuenta_las_seguidas_de_cero():
    fotos = _fotos(varianza=[10, 10, 10, 20, 20, 20, 10, 20, 20])
    aumentos = {"varianza": 0.5}

    assert m.detectar_aumento(fotos, aumentos, 3, 2)["indice"] == 4
    # retoma después de descartar la señal de las fotos 3-5: las anteriores no cuentan, y la línea de base sigue siendo la misma
    retomada = m.detectar_aumento(fotos, aumentos, 3, 2, desde=5)
    assert retomada["indice"] == 8 and retomada["linea_base"] == {"varianza": 10}
    # 'desde' dentro de la línea de base no la mueve: no se evalúan esas fotos
    assert m.detectar_aumento(fotos, aumentos, 3, 2, desde=0)["indice"] == 4


def test_detectar_aumento_con_linea_de_base_en_cero_cualquier_valor_positivo_detecta():
    assert m.detectar_aumento(_fotos(laplacian=[0, 0, 0, 0, 0]), {"laplacian": 0.5}, 3, 1) is None
    assert m.detectar_aumento(_fotos(laplacian=[0, 0, 0, 0, 0.001]), {"laplacian": 0.5}, 3, 1)["indice"] == 4


def test_detectar_aumento_valida_sus_parametros():
    fotos = _fotos(varianza=[10, 10, 10, 30])

    for kwargs in (dict(fotos_seguidas=0), dict(fotos_seguidas=1.5), dict(desde=-1), dict(pasos_linea_base=0)):
        try:
            m.detectar_aumento(fotos, {"varianza": 0.5}, **kwargs)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(kwargs)

    try:
        m.detectar_aumento(fotos, {"foco": 0.5})
    except ValueError:
        pass
    else:
        raise AssertionError("una métrica que no existe tenía que fallar")
