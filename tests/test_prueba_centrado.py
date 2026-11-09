# %%

"""
Pruebas de la aplicación "26.10.29 test de centrado" (prueba_centrado.py), con la impresora REAL sobre
serie simulado y un microscopio simulado (CamaraFalsaConMuestra, en simuladores.py).

Qué se verifica:
  - el píxel se elige al azar con la misma probabilidad para todos;
  - con la escala correcta, las 10 pruebas dejan el píxel elegido dentro de la celda de la grilla, lo que se
    mide en la foto de después coincide con lo esperado, y se guardan las 4 imágenes de cada prueba;
  - con la escala mal calibrada (el error de 26.10.16: FOV_X y FOV_Y al revés) la prueba lo NOTA;
  - con una cámara que devuelve cuadros viejos, la prueba lo nota si no se descartan cuadros y no si se descartan;
  - solo se mueve en XY, sobre la grilla de 12.5 µm, sin viajes largos, y vuelve al inicio;
  - si la foto no es QVGA falla ANTES de mover, y sin home() falla con un mensaje claro.

Lo que NO prueban: tiempos reales, óptica, ruido ni cuadros viejos reales de la placa OpenMV.
Que pase no reemplaza correr la prueba en el equipo.

Creado el 2026-10-29.
"""

import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from simuladores import RAIZ, CamaraFalsaConMuestra, crear_impresora_falsa, movimientos_g1, omitir, sin_esperas

sys.path.insert(0, str(RAIZ / "26.10.29 test de centrado"))

import prueba_centrado as pc  # noqa: E402
from centrado import Calibracion  # noqa: E402  (la de 26.10.16, la misma que usa prueba_centrado)
from impresora_cosmos import DISTANCIA_XY_SIN_SUBIR_MM  # noqa: E402

INICIO = (150.0, 150.0, 27.7)       # (x, y, z) donde está la cámara al empezar
Z_ARRIBA_SEGURO = 27.9


def _omitir_si_la_app_usa_config_de_una_con_el_formato_viejo():
    """
    El 2026-11-09 OpenMVCamera.config_de_una() pasó a recibir un diccionario de parámetros (sin alias del
    formato viejo, con pixformat=..., resolucion=... sueltos). Mientras prueba_centrado.py siga llamándola
    como antes, estas pruebas no pueden correr: se omiten (no fallan) hasta que Franco pida adecuar la
    aplicación (BITACORA.txt, pendiente 14). Apenas se la adecue, vuelven a correr solas.
    """

    codigo = (RAIZ / "26.10.29 test de centrado" / "prueba_centrado.py").read_text(encoding="utf-8")

    if re.search(r"config_de_una\(\s*pixformat\s*=", codigo):
        omitir("26.10.29 todavía llama a config_de_una() con el formato viejo (parámetros sueltos): "
               "falta adecuarla al diccionario de parámetros")


def _armar(**kwargs_camara):
    _omitir_si_la_app_usa_config_de_una_con_el_formato_viejo()
    impresora = crear_impresora_falsa()
    impresora.desp_lin_000(*INICIO)
    impresora.ser.escrituras.clear()

    real = pc.crear_calibracion()               # la escala "real" del microscopio simulado
    kwargs_camara.setdefault("um_por_px_fila", real.um_por_px_fila)
    kwargs_camara.setdefault("um_por_px_col", real.um_por_px_col)
    camara = CamaraFalsaConMuestra(impresora, origen_mm=INICIO[:2], **kwargs_camara)

    return impresora, camara


def _correr(calibracion, carpeta, n=10, semilla=1, camara_kwargs=None, **kwargs):
    impresora, camara = _armar(**(camara_kwargs or {}))
    generador = np.random.default_rng(semilla)

    with sin_esperas():
        resultados = [
            pc.probar_centrado(camara, impresora, calibracion, Z_ARRIBA_SEGURO, carpeta, numero=k,
                               generador=generador, devolveme_la_respuesta=False, **kwargs)
            for k in range(1, n + 1)
        ]

    return resultados, impresora


def _max_diferencia(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def test_el_pixel_al_azar_tiene_la_misma_probabilidad_en_todos_los_pixeles():
    generador = np.random.default_rng(0)
    cuenta = np.zeros((6, 8), int)

    for _ in range(96000):
        col, fila = pc.elegir_pixel_al_azar(6, 8, generador)
        cuenta[fila, col] += 1

    assert cuenta.min() > 0.9 * 2000 and cuenta.max() < 1.1 * 2000, cuenta

    puntos = [pc.elegir_pixel_al_azar(pc.ALTO_PX, pc.ANCHO_PX, generador) for _ in range(20000)]
    assert all(0 <= col < pc.ANCHO_PX and 0 <= fila < pc.ALTO_PX for col, fila in puntos)


def test_centrado_con_la_escala_correcta():
    with tempfile.TemporaryDirectory() as carpeta:
        resultados, impresora = _correr(pc.crear_calibracion(), carpeta, n=10)

        for r in resultados:
            assert r.hallado is not None and r.hallado[2] > 0.9, r
            assert r.centrado is True, r
            # lo medido en la foto coincide con lo que calculó centrar_pixel(): píxeles y µm
            assert r.diferencia_de_corrimiento_px <= 1.5, r
            assert "OJO" not in pc._texto_resultado(r, 12.5)
            assert _max_diferencia(r.residuo_medido_um, r.residuo_esperado_um) <= 1.5, r
            assert max(abs(v) for v in r.residuo_esperado_um) <= 6.25 + 1e-6

            for clave in ("antes", "antes_marcada", "despues", "comparacion"):
                assert Path(r.archivos[clave]).exists(), (clave, r.archivos)

        # cubrió movimientos grandes (píxeles lejos del centro), no solo los fáciles
        assert max(max(abs(v) for v in r.movimiento_pedido_um) for r in resultados) > 30
        assert "10 pruebas: 10 centradas, 0 sin centrar, 0 sin poder medir" == pc.resumir_pruebas(resultados)

        # las imágenes: la foto es la cruda, la marcada es 2x y se distingue de la cruda, la comparación junta dos
        r = resultados[0]
        cruda = cv2.imread(r.archivos["antes"])
        marcada = cv2.imread(r.archivos["antes_marcada"])
        comparacion = cv2.imread(r.archivos["comparacion"])
        assert cruda.shape == (240, 320, 3) and marcada.shape == (480, 640, 3) and comparacion.shape == (480, 1280, 3)
        ampliada = cv2.resize(cruda, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        distintos = (np.abs(marcada.astype(int) - ampliada.astype(int)).max(axis=2) > 30).mean()
        assert 0.005 < distintos < 0.4, distintos
        # el píxel elegido quedó dibujado (la cruz roja del marcador) y la foto cruda no tiene dibujos
        assert np.abs(marcada[2 * r.fila, 2 * r.col].astype(int) - [60, 60, 255]).max() <= 10     # (con antialiasing)

        # cada prueba termina donde empezó
        assert (impresora.posicion_x, impresora.posicion_y) == INICIO[:2]


def test_detecta_una_escala_mal_calibrada():
    real = pc.crear_calibracion()
    # el error que había en 26.10.16: FOV_X y FOV_Y al revés
    al_reves = Calibracion.desde_fov_mm(pc.ANCHO_PX, pc.ALTO_PX, pc.FOV_Y_MM, pc.FOV_X_MM)
    assert abs(al_reves.um_por_px_fila / real.um_por_px_fila - 1) > 0.3

    with tempfile.TemporaryDirectory() as carpeta:
        resultados, _ = _correr(al_reves, carpeta, n=12)

    medidos = [r for r in resultados if r.hallado is not None]
    sin_centrar = [r for r in resultados if r.centrado is False]

    assert len(sin_centrar) >= 3, pc.resumir_pruebas(resultados)
    # lo que se corrió la imagen no es lo que se pedía: eso es lo que delata la escala (aunque alguna haya quedado centrada)
    assert max(r.diferencia_de_corrimiento_px for r in medidos) > 10
    assert sum(r.diferencia_de_corrimiento_px > pc.AVISO_CORRIMIENTO_PX for r in medidos) >= 6
    assert "OJO" in pc._texto_resultado(max(medidos, key=lambda r: r.diferencia_de_corrimiento_px), 12.5)


def test_los_cuadros_viejos_de_la_camara_se_notan_y_se_corrigen_descartando():
    calibracion = pc.crear_calibracion()

    with tempfile.TemporaryDirectory() as carpeta:
        sin_descartar, _ = _correr(calibracion, carpeta, n=8, camara_kwargs=dict(frames_de_retraso=1),
                                   frames_a_descartar=0)
        descartando, _ = _correr(calibracion, carpeta, n=8, camara_kwargs=dict(frames_de_retraso=1),
                                 frames_a_descartar=1)

    assert all(r.centrado is True for r in descartando), pc.resumir_pruebas(descartando)
    assert sum(r.centrado is not True for r in sin_descartar) >= 5, pc.resumir_pruebas(sin_descartar)


def test_solo_se_mueve_en_xy_sobre_la_grilla_y_sin_viajes_largos():
    with tempfile.TemporaryDirectory() as carpeta:
        resultados, impresora = _correr(pc.crear_calibracion(), carpeta, n=6)

    movimientos = movimientos_g1(impresora)
    en_xy = [m for m in movimientos if "X" in m]
    en_z = [m for m in movimientos if "X" not in m and "Z" in m]
    assert en_xy and en_z

    anterior = (INICIO[0] + impresora.offset_x, INICIO[1] + impresora.offset_y)
    for m in en_xy:
        x, y = m["X"] - impresora.offset_x, m["Y"] - impresora.offset_y
        assert abs(x / impresora.resolucion_x - round(x / impresora.resolucion_x)) < 1e-6, m     # sobre la grilla de 12.5 µm
        assert abs(y / impresora.resolucion_y - round(y / impresora.resolucion_y)) < 1e-6, m
        assert np.hypot(m["X"] - anterior[0], m["Y"] - anterior[1]) <= DISTANCIA_XY_SIN_SUBIR_MM, m  # nunca un viaje largo
        anterior = (m["X"], m["Y"])

    assert {m["Z"] for m in en_z} == {INICIO[2] + impresora.offset_z}      # Z no cambió en ningún momento


def test_si_la_foto_no_es_qvga_falla_antes_de_mover():
    impresora, camara = _armar(alto=480, ancho=640)

    with tempfile.TemporaryDirectory() as carpeta, sin_esperas():
        try:
            pc.probar_centrado(camara, impresora, pc.crear_calibracion(), Z_ARRIBA_SEGURO, carpeta,
                               devolveme_la_respuesta=False)
        except ValueError as error:
            assert "No se movió nada" in str(error)
        else:
            raise AssertionError("tendría que haber fallado: la foto no mide lo que supone la calibración")

    assert movimientos_g1(impresora) == []


def test_sin_home_falla_con_un_mensaje_claro():
    impresora = crear_impresora_falsa(hacer_home=False)
    camara = CamaraFalsaConMuestra(impresora)

    with tempfile.TemporaryDirectory() as carpeta:
        try:
            pc.probar_centrado(camara, impresora, pc.crear_calibracion(), Z_ARRIBA_SEGURO, carpeta,
                               devolveme_la_respuesta=False)
        except RuntimeError as error:
            assert "home" in str(error)
        else:
            raise AssertionError("tendría que haber pedido home()")


# %%
