# %%

"""
Pruebas de la aplicación "26.10.29 test de centrado" (prueba_centrado.py), con la impresora REAL sobre
serie simulado y un microscopio simulado (CamaraFalsaConMuestra, en simuladores.py).

Qué se verifica:
  - el píxel se elige al azar con la misma probabilidad para todos los permitidos, y por defecto solo entre los que
    están a más de 6.25 µm del centro en algún eje (los que obligan a mover la impresora);
  - con la escala correcta, las 10 pruebas dejan el píxel elegido dentro de la celda de la grilla en el primer paso, lo que
    se mide en la foto de después coincide con lo esperado, y se guardan las 4 imágenes del paso;
  - la grilla punteada va en las imágenes marcadas y NO en las crudas, y sus líneas caen justo en los bordes de las zonas
    a las que centrar_pixel() manda la platina;
  - si no quedó centrado se hace otro paso (hasta max_correcciones) con las mismas fotos y dibujos, y el lugar a centrar es
    donde se midió lo elegido; si no se pudo medir, no se corrige;
  - con la escala mal calibrada (el error de 26.10.16: FOV_X y FOV_Y al revés) la prueba lo NOTA;
  - con una cámara que devuelve cuadros viejos, la prueba lo nota si no se descartan cuadros y no si se descartan;
  - solo se mueve en XY, sobre la grilla de 12.5 µm, sin viajes largos (también en las correcciones), y vuelve al inicio;
  - si la foto no es QVGA falla ANTES de mover, y sin home() o sin diccionario de cámara falla con un mensaje claro.

Lo que NO prueban: tiempos reales, óptica, ruido ni cuadros viejos reales de la placa OpenMV.
Que pase no reemplaza correr la prueba en el equipo.

Creado el 2026-10-29. Actualizado el 2026-11-12: diccionario de parámetros de cámara, grilla punteada, píxel fuera de
la celda central y paso de corrección.
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from simuladores import RAIZ, CamaraFalsaConMuestra, crear_impresora_falsa, movimientos_g1, sin_esperas

sys.path.insert(0, str(RAIZ / "26.10.29 test de centrado"))

import prueba_centrado as pc  # noqa: E402
from centrado import Calibracion, centrar_pixel, esta_centrado  # noqa: E402  (las de 26.10.16, las mismas que usa prueba_centrado)
from impresora_cosmos import DISTANCIA_XY_SIN_SUBIR_MM  # noqa: E402

INICIO = (150.0, 150.0, 27.7)       # (x, y, z) donde está la cámara al empezar
Z_ARRIBA_SEGURO = 27.9
DICT_PARAMS_CAMARA = {"pixformat": "RGB565", "resolucion": "QVGA", "exposure_us": None, "gain_db": None, "whitebal": None}
NARANJA_BGR = (0, 100, 230)         # el color de la grilla punteada


def _armar(**kwargs_camara):
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
    kwargs.setdefault("dic_param_cam", DICT_PARAMS_CAMARA)

    with sin_esperas():
        resultados = [
            pc.probar_centrado(camara, impresora, calibracion, Z_ARRIBA_SEGURO, carpeta, numero=k,
                               generador=generador, devolveme_la_respuesta=False, **kwargs)
            for k in range(1, n + 1)
        ]

    return resultados, impresora


def _max_diferencia(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def _escala_al_reves():
    """El error que había en 26.10.16: FOV_X y FOV_Y al revés."""
    return Calibracion.desde_fov_mm(pc.ANCHO_PX, pc.ALTO_PX, pc.FOV_Y_MM, pc.FOV_X_MM)


def _escala_con_error(factor):
    """La calibración de QVGA con todos los µm/px multiplicados por 'factor'. Con factor < 1 la prueba mueve de menos
    (el primer paso a veces no alcanza y la corrección lo arregla); con factor > 1 mueve de más (la corrección puede no
    converger: mide en µm equivocados, así que se ve 'sin centrar' algo que está dentro de la celda)."""
    real = pc.crear_calibracion()
    return Calibracion(pc.ANCHO_PX, pc.ALTO_PX, um_por_px_fila=real.um_por_px_fila * factor,
                       um_por_px_col=real.um_por_px_col * factor)


def _pixeles_naranja(imagen):
    """Cuántos píxeles de la imagen tienen el color de la grilla (con un poco de tolerancia)."""
    return int((np.abs(imagen.astype(int) - NARANJA_BGR).max(axis=2) <= 25).sum())


# ---------------------------------------------------------------------- elegir el píxel
def test_el_pixel_al_azar_tiene_la_misma_probabilidad_en_todos_los_pixeles():
    generador = np.random.default_rng(0)
    cuenta = np.zeros((6, 8), int)

    for _ in range(96000):
        col, fila = pc.elegir_pixel_al_azar(6, 8, generador)
        cuenta[fila, col] += 1

    assert cuenta.min() > 0.9 * 2000 and cuenta.max() < 1.1 * 2000, cuenta

    puntos = [pc.elegir_pixel_al_azar(pc.ALTO_PX, pc.ANCHO_PX, generador) for _ in range(20000)]
    assert all(0 <= col < pc.ANCHO_PX and 0 <= fila < pc.ALTO_PX for col, fila in puntos)


def test_el_pixel_al_azar_con_mascara_es_uniforme_entre_los_permitidos():
    generador = np.random.default_rng(0)
    permitidos = np.ones((6, 8), bool)
    permitidos[2:4, 3:5] = False                # un hueco de 2 x 2 en el medio
    cuenta = np.zeros((6, 8), int)

    for _ in range(88000):
        col, fila = pc.elegir_pixel_al_azar(6, 8, generador, permitidos)
        cuenta[fila, col] += 1

    assert (cuenta[~permitidos] == 0).all(), cuenta
    assert cuenta[permitidos].min() > 0.9 * 2000 and cuenta[permitidos].max() < 1.1 * 2000, cuenta

    try:
        pc.elegir_pixel_al_azar(6, 8, generador, np.zeros((6, 8), bool))
    except ValueError:
        pass
    else:
        raise AssertionError("sin píxeles permitidos tendría que fallar")


def test_los_pixeles_que_obligan_a_mover_son_los_que_estan_a_mas_de_6_25_um_en_algun_eje():
    um_por_px = 12.5 / 117.469      # la escala medida con la óptica nueva
    escalas = {
        "lente de octubre": pc.crear_calibracion(),
        "óptica nueva": Calibracion.desde_fov_mm(320, 240, 240 * um_por_px / 1000, 320 * um_por_px / 1000),
    }

    for nombre, calibracion in escalas.items():
        mascara = pc.pixeles_que_obligan_a_mover(calibracion)
        assert mascara.shape == (240, 320), nombre

        # coincide, píxel por píxel, con la definición de centrado.esta_centrado() (muestra grande + las cuatro esquinas)
        generador = np.random.default_rng(3)
        muestra = [(int(c), int(f)) for c, f in zip(generador.integers(0, 320, 3000), generador.integers(0, 240, 3000))]
        muestra += [(0, 0), (319, 0), (0, 239), (319, 239), (160, 120), (159, 119)]
        for col, fila in muestra:
            assert bool(mascara[fila, col]) == (not esta_centrado(calibracion, col, fila)), (nombre, col, fila)

        # a más de 6.25 µm del centro en algún eje (y solo así), en µm
        dx = (np.arange(240) - calibracion.centro_fila) * calibracion.um_por_px_fila
        dy = (np.arange(320) - calibracion.centro_col) * calibracion.um_por_px_col
        lejos = (np.abs(dx)[:, None] > 6.25 + 1e-6) | (np.abs(dy)[None, :] > 6.25 + 1e-6)
        assert (mascara == lejos).all(), nombre
        assert not mascara[119, 159] and not mascara[120, 160], nombre         # el centro está excluido
        assert mascara[0, 0] and mascara[239, 319], nombre

    # con la óptica nueva la celda central es grande (~18 % de la imagen); con la de octubre, ~1.4 %
    assert 0.15 < 1 - pc.pixeles_que_obligan_a_mover(escalas["óptica nueva"]).mean() < 0.22
    assert 0.005 < 1 - pc.pixeles_que_obligan_a_mover(escalas["lente de octubre"]).mean() < 0.03


def test_probar_centrado_elige_el_pixel_fuera_de_la_celda_central_y_se_puede_desactivar():
    calibracion = pc.crear_calibracion()
    pedidos = []
    original = pc.elegir_pixel_al_azar

    def espia(alto, ancho, generador, permitidos=None):
        pedidos.append(permitidos)
        return original(alto, ancho, generador, permitidos)

    pc.elegir_pixel_al_azar = espia
    try:
        with tempfile.TemporaryDirectory() as carpeta:
            resultados, _ = _correr(calibracion, carpeta, n=20)
            assert len(pedidos) == 20 and all(p is not None for p in pedidos)
            assert all(not esta_centrado(calibracion, r.col, r.fila) for r in resultados)
            assert all(max(abs(v) for v in r.pasos[0].movimiento_pedido_um) > 6.25 for r in resultados)

            pedidos.clear()
            _correr(calibracion, carpeta, n=2, evitar_celda_central=False)
            assert pedidos == [None, None]
    finally:
        pc.elegir_pixel_al_azar = original


def test_si_ningun_pixel_obliga_a_mover_avisa_y_elige_entre_todos():
    minuscula = Calibracion(pc.ANCHO_PX, pc.ALTO_PX, um_por_px_fila=0.01, um_por_px_col=0.01)   # toda la imagen cae en ±2 µm
    assert not pc.pixeles_que_obligan_a_mover(minuscula).any()

    impresora, camara = _armar(um_por_px_fila=0.01, um_por_px_col=0.01)
    salida = io.StringIO()

    with tempfile.TemporaryDirectory() as carpeta, sin_esperas(), contextlib.redirect_stdout(salida):
        r = pc.probar_centrado(camara, impresora, minuscula, Z_ARRIBA_SEGURO, carpeta, dic_param_cam=DICT_PARAMS_CAMARA,
                               devolveme_la_respuesta=False)

    assert "AVISO" in salida.getvalue() and "celda central" in salida.getvalue()
    assert 0 <= r.col < pc.ANCHO_PX and 0 <= r.fila < pc.ALTO_PX


# ---------------------------------------------------------------------- un paso, con la escala correcta
def test_centrado_con_la_escala_correcta():
    calibracion = pc.crear_calibracion()

    with tempfile.TemporaryDirectory() as carpeta:
        resultados, impresora = _correr(calibracion, carpeta, n=10)

        for r in resultados:
            assert len(r.pasos) == 1 and r.n_correcciones == 0, r      # no hizo falta corregir
            p = r.pasos[0]
            assert p.numero == 1 and (p.col, p.fila) == (r.col, r.fila)
            assert p.hallado is not None and p.hallado[2] > 0.9, p
            assert p.centrado is True and r.centrado is True and r.centrado_al_primer_paso is True, p
            # lo medido en la foto coincide con lo que calculó centrar_pixel(): píxeles y µm
            assert p.diferencia_de_corrimiento_px <= 1.5, p
            assert "OJO" not in pc._texto_resultado(r, 12.5)
            assert "CENTRADO en el primer paso" in pc._texto_resultado(r, 12.5)
            assert _max_diferencia(p.residuo_medido_um, p.residuo_esperado_um) <= 1.5, p
            assert max(abs(v) for v in p.residuo_esperado_um) <= 6.25 + 1e-6

            for clave in ("antes", "antes_marcada", "despues", "comparacion"):
                assert Path(p.archivos[clave]).exists(), (clave, p.archivos)

        # cubrió movimientos grandes (píxeles lejos del centro), no solo los fáciles
        assert max(max(abs(v) for v in r.pasos[0].movimiento_pedido_um) for r in resultados) > 30
        assert pc.resumir_pruebas(resultados) == ("10 pruebas: 10 centradas en el primer paso, 0 centradas tras corregir, "
                                                  "0 sin centrar, 0 sin poder medir")
        assert not [a for a in Path(carpeta).iterdir() if "paso" in a.name]     # sin correcciones no hay archivos de 'paso2'
        assert len(list(Path(carpeta).iterdir())) == 4 * 10

        # las imágenes: la foto es la cruda, la marcada es 2x y se distingue de la cruda, la comparación junta dos
        r = resultados[0]
        p = r.pasos[0]
        cruda = cv2.imread(p.archivos["antes"])
        marcada = cv2.imread(p.archivos["antes_marcada"])
        comparacion = cv2.imread(p.archivos["comparacion"])
        assert cruda.shape == (240, 320, 3) and marcada.shape == (480, 640, 3) and comparacion.shape == (480, 1280, 3)
        ampliada = cv2.resize(cruda, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        distintos = (np.abs(marcada.astype(int) - ampliada.astype(int)).max(axis=2) > 30).mean()
        assert 0.005 < distintos < 0.4, distintos
        # el píxel elegido quedó dibujado (la cruz roja del marcador) y la foto cruda no tiene dibujos
        assert np.abs(marcada[2 * r.fila, 2 * r.col].astype(int) - [60, 60, 255]).max() <= 10     # (con antialiasing)

        # cada prueba termina donde empezó
        assert (impresora.posicion_x, impresora.posicion_y) == INICIO[:2]


def test_centra_tambien_con_la_escala_de_la_optica_nueva():
    um_por_px = 12.5 / 117.469      # 12.5 µm = 117.469 px: la celda de la grilla mide ~117 px, más que la mitad del alto
    calibracion = Calibracion.desde_fov_mm(pc.ANCHO_PX, pc.ALTO_PX, pc.ALTO_PX * um_por_px / 1000, pc.ANCHO_PX * um_por_px / 1000)

    with tempfile.TemporaryDirectory() as carpeta:
        resultados, impresora = _correr(calibracion, carpeta, n=8,
                                        camara_kwargs=dict(um_por_px_fila=um_por_px, um_por_px_col=um_por_px, grano=3.0))

    assert all(r.centrado is True for r in resultados), pc.resumir_pruebas(resultados)
    assert all(r.n_correcciones == 0 for r in resultados), pc.resumir_pruebas(resultados)
    assert all(not esta_centrado(calibracion, r.col, r.fila) for r in resultados)
    assert (impresora.posicion_x, impresora.posicion_y) == INICIO[:2]


# ---------------------------------------------------------------------- la grilla punteada
def test_la_grilla_punteada_va_en_las_imagenes_marcadas_y_no_en_las_crudas():
    with tempfile.TemporaryDirectory() as carpeta:
        resultados, _ = _correr(pc.crear_calibracion(), carpeta, n=3)

        for r in resultados:
            p = r.pasos[0]
            # crudas: sin un solo píxel de la grilla (el mundo simulado no tiene ese color)
            assert _pixeles_naranja(cv2.imread(p.archivos["antes"])) == 0
            assert _pixeles_naranja(cv2.imread(p.archivos["despues"])) == 0
            # marcadas: tienen la grilla, y las dos mitades de la comparación también
            marcada = cv2.imread(p.archivos["antes_marcada"])
            comparacion = cv2.imread(p.archivos["comparacion"])
            assert _pixeles_naranja(marcada) > 1000
            assert _pixeles_naranja(comparacion[:, :640]) > 1000 and _pixeles_naranja(comparacion[:, 640:]) > 1000

        # es PUNTEADA (no una línea continua) y cae donde se dijo: miramos la columna de cada línea vertical, debajo del texto
        calibracion = pc.crear_calibracion()
        columnas = pc._posiciones_de_las_lineas(calibracion.centro_col, calibracion.um_por_px_col, calibracion.paso_um,
                                                calibracion.ancho_px, pc.AUMENTO)
        marcada = cv2.imread(resultados[0].pasos[0].archivos["antes_marcada"])
        for x in columnas:
            tramo = marcada[130:, x - 1:x + 2]
            fraccion = (np.abs(tramo.astype(int) - NARANJA_BGR).max(axis=2) <= 25).any(axis=1).mean()
            assert 0.35 < fraccion < 0.65, (x, fraccion)      # rayas de 4 px cada 8 (0.5); una línea continua daría 1


def test_las_lineas_de_la_grilla_son_los_bordes_de_las_zonas_de_centrar_pixel():
    for nombre, calibracion in {"lente de octubre": pc.crear_calibracion(),
                                "óptica nueva": Calibracion.desde_fov_mm(320, 240, 0.0255, 0.034)}.items():
        pos = (150.0, 150.0)

        # dónde cambia el destino que calcula centrar_pixel() al recorrer la imagen por columnas (eje y) y por filas (eje x)
        bordes_col = [c + 0.5 for c in range(pc.ANCHO_PX - 1)
                      if centrar_pixel(calibracion, pos, c, calibracion.centro_fila).pasos_y
                      != centrar_pixel(calibracion, pos, c + 1, calibracion.centro_fila).pasos_y]
        bordes_fila = [f + 0.5 for f in range(pc.ALTO_PX - 1)
                       if centrar_pixel(calibracion, pos, calibracion.centro_col, f).pasos_x
                       != centrar_pixel(calibracion, pos, calibracion.centro_col, f + 1).pasos_x]

        columnas = pc._posiciones_de_las_lineas(calibracion.centro_col, calibracion.um_por_px_col, calibracion.paso_um,
                                                calibracion.ancho_px, pc.AUMENTO)
        filas = pc._posiciones_de_las_lineas(calibracion.centro_fila, calibracion.um_por_px_fila, calibracion.paso_um,
                                             calibracion.alto_px, pc.AUMENTO)

        for lineas, bordes in ((columnas, bordes_col), (filas, bordes_fila)):
            esperadas = [pc._coordenada_ampliada(b, pc.AUMENTO) for b in bordes]
            assert len(lineas) == len(esperadas) and len(esperadas) >= 2, (nombre, lineas, esperadas)
            assert max(abs(a - b) for a, b in zip(lineas, esperadas)) <= 1, (nombre, lineas, esperadas)

        # simétricas respecto de la cruz del centro: la zona central mide un paso de ancho
        centro = pc._coordenada_ampliada(calibracion.centro_col, pc.AUMENTO)
        central = [x for x in columnas if abs(x - centro) < 0.6 * calibracion.paso_um / calibracion.um_por_px_col * pc.AUMENTO]
        assert len(central) == 2 and abs((central[0] + central[1]) / 2 - centro) <= 1, (nombre, columnas)


# ---------------------------------------------------------------------- el paso de corrección
def test_si_no_queda_centrado_hace_otro_paso_con_las_mismas_fotos_y_el_mismo_estilo():
    with tempfile.TemporaryDirectory() as carpeta:
        # una calibración que mueve 12 % de menos: el primer paso no siempre alcanza y una corrección lo arregla
        resultados, impresora = _correr(_escala_con_error(0.88), carpeta, n=30, semilla=5, max_correcciones=1)
        archivos = {a.name for a in Path(carpeta).iterdir()}

        corregidas = [r for r in resultados if r.n_correcciones > 0]
        assert len(corregidas) >= 5, pc.resumir_pruebas(resultados)

        for r in resultados:
            primero = r.pasos[0]
            assert len(r.pasos) <= 2
            assert r.centrado_al_primer_paso is primero.centrado

            if primero.centrado is True:
                assert len(r.pasos) == 1                                # centrado: no se corrige
                assert f"prueba_{r.numero:02d}_paso2_1_antes.jpg" not in archivos
                continue

            assert len(r.pasos) == 2 and r.n_correcciones == 1
            segundo = r.pasos[1]
            assert primero.centrado is False and segundo.numero == 2
            # el lugar a centrar es donde se midió lo elegido, y se parte de donde quedó la impresora
            assert (segundo.col, segundo.fila) == tuple(primero.hallado[:2])
            assert segundo.pos_antes_mm == primero.pos_despues_mm
            # mismas cuatro fotos que el primer paso, con 'paso2' en el nombre; la de antes es la de después del paso 1
            assert set(segundo.archivos) == set(primero.archivos) == {"antes", "antes_marcada", "despues", "comparacion"}
            for clave, ruta in segundo.archivos.items():
                assert Path(ruta).exists() and f"prueba_{r.numero:02d}_paso2_" in Path(ruta).name, (clave, ruta)
            assert Path(segundo.archivos["antes"]).read_bytes() == Path(primero.archivos["despues"]).read_bytes()
            # mismo estilo: marcada 2x con grilla punteada, comparación con las dos mitades; las crudas, sin dibujos
            marcada = cv2.imread(segundo.archivos["antes_marcada"])
            comparacion = cv2.imread(segundo.archivos["comparacion"])
            assert marcada.shape == (480, 640, 3) and comparacion.shape == (480, 1280, 3)
            assert _pixeles_naranja(marcada) > 1000 and _pixeles_naranja(comparacion[:, 640:]) > 1000
            assert _pixeles_naranja(cv2.imread(segundo.archivos["antes"])) == 0
            assert _pixeles_naranja(cv2.imread(segundo.archivos["despues"])) == 0
            # y la corrección sirve: con un paso más queda centrado
            assert segundo.centrado is True and r.centrado is True, pc._texto_resultado(r, 12.5)
            assert "CENTRADO después de 1 corrección" in pc._texto_resultado(r, 12.5)
            assert "Paso 2 (corrección" in pc._texto_resultado(r, 12.5)

        resumen = pc.resumir_pruebas(resultados)
        assert f"{len(corregidas)} centradas tras corregir" in resumen and "0 sin centrar" in resumen, resumen
        assert (impresora.posicion_x, impresora.posicion_y) == INICIO[:2]


def test_max_correcciones_limita_los_pasos_y_con_cero_no_corrige():
    with tempfile.TemporaryDirectory() as carpeta:
        sin_corregir, _ = _correr(_escala_con_error(0.88), carpeta, n=30, semilla=5, max_correcciones=0)
        assert all(len(r.pasos) == 1 for r in sin_corregir)
        assert sum(r.centrado is False for r in sin_corregir) >= 5, pc.resumir_pruebas(sin_corregir)
        assert not [a for a in Path(carpeta).iterdir() if "paso" in a.name]

    with tempfile.TemporaryDirectory() as carpeta:
        # se corrige solo mientras no esté centrado: con esta escala alcanza con una corrección, aunque se permitan tres
        varias, _ = _correr(_escala_con_error(0.88), carpeta, n=30, semilla=5, max_correcciones=3)
        for r in varias:
            assert 1 <= len(r.pasos) <= 2
            assert all(p.centrado is False for p in r.pasos[:-1])
            assert r.pasos[-1].centrado is True, pc._texto_resultado(r, 12.5)
        assert any(len(r.pasos) == 2 for r in varias)

    with tempfile.TemporaryDirectory() as carpeta:
        # una escala que mueve 30 % de más: las correcciones no convergen y se corta en max_correcciones
        resultados, _ = _correr(_escala_con_error(1.3), carpeta, n=30, semilla=5, max_correcciones=2)
        sin_centrar = [r for r in resultados if r.centrado is False]
        assert sin_centrar, pc.resumir_pruebas(resultados)
        assert all(len(r.pasos) == 3 for r in sin_centrar)
        assert all(len(r.pasos) <= 3 for r in resultados)
        assert "SIN CENTRAR (después de 2 corrección(es))" in pc._texto_resultado(sin_centrar[0], 12.5)

    try:
        _correr(pc.crear_calibracion(), tempfile.gettempdir(), n=1, max_correcciones=-1)
    except ValueError as error:
        assert "max_correcciones" in str(error)
    else:
        raise AssertionError("max_correcciones negativo tendría que fallar")


def test_si_no_se_pudo_medir_no_corrige():
    original = pc.buscar_parche
    pc.buscar_parche = lambda *argumentos, **kwargs: None       # nunca ubica el parche
    try:
        with tempfile.TemporaryDirectory() as carpeta:
            resultados, impresora = _correr(pc.crear_calibracion(), carpeta, n=3, max_correcciones=2)
    finally:
        pc.buscar_parche = original

    for r in resultados:
        assert len(r.pasos) == 1 and r.centrado is None and r.centrado_al_primer_paso is None
        assert "NO SE PUDO MEDIR" in pc._texto_resultado(r, 12.5)

    assert pc.resumir_pruebas(resultados).endswith("0 sin centrar, 3 sin poder medir")
    assert (impresora.posicion_x, impresora.posicion_y) == INICIO[:2]


# ---------------------------------------------------------------------- la escala mal calibrada y los cuadros viejos
def test_detecta_una_escala_mal_calibrada():
    real = pc.crear_calibracion()
    al_reves = _escala_al_reves()
    assert abs(al_reves.um_por_px_fila / real.um_por_px_fila - 1) > 0.3

    with tempfile.TemporaryDirectory() as carpeta:
        resultados, _ = _correr(al_reves, carpeta, n=12, max_correcciones=0)

    pasos = [r.pasos[0] for r in resultados]
    medidos = [p for p in pasos if p.hallado is not None]
    sin_centrar = [p for p in pasos if p.centrado is False]

    assert len(sin_centrar) >= 3, pc.resumir_pruebas(resultados)
    # lo que se corrió la imagen no es lo que se pedía: eso es lo que delata la escala (aunque alguna haya quedado centrada)
    assert max(p.diferencia_de_corrimiento_px for p in medidos) > 10
    assert sum(p.diferencia_de_corrimiento_px > pc.AVISO_CORRIMIENTO_PX for p in medidos) >= 6
    peor = max(resultados, key=lambda r: r.pasos[0].diferencia_de_corrimiento_px)
    assert "OJO" in pc._texto_resultado(peor, 12.5)

    # y más pasos no la arreglan: con la escala mal puesta se mide en µm equivocados, así que algunas pruebas siguen
    # 'sin centrar' después de corregir, y el aviso OJO sigue ahí
    with tempfile.TemporaryDirectory() as carpeta:
        corregidas, _ = _correr(al_reves, carpeta, n=12, max_correcciones=2)

    persisten = [r for r in corregidas if r.centrado is False]
    assert persisten, pc.resumir_pruebas(corregidas)
    assert any("OJO" in pc._texto_resultado(r, 12.5) for r in persisten)


def test_los_cuadros_viejos_de_la_camara_se_notan_y_se_corrigen_descartando():
    calibracion = pc.crear_calibracion()

    with tempfile.TemporaryDirectory() as carpeta:
        sin_descartar, _ = _correr(calibracion, carpeta, n=8, camara_kwargs=dict(frames_de_retraso=1),
                                   frames_a_descartar=0, max_correcciones=0)
        descartando, _ = _correr(calibracion, carpeta, n=8, camara_kwargs=dict(frames_de_retraso=1),
                                 frames_a_descartar=1, max_correcciones=0)

    assert all(r.centrado is True for r in descartando), pc.resumir_pruebas(descartando)
    assert sum(r.centrado is not True for r in sin_descartar) >= 5, pc.resumir_pruebas(sin_descartar)


# ---------------------------------------------------------------------- seguridad
def test_solo_se_mueve_en_xy_sobre_la_grilla_y_sin_viajes_largos():
    # tanto con la escala bien puesta (un paso) como con una mal puesta que obliga a corregir (varios pasos)
    for calibracion in (pc.crear_calibracion(), _escala_con_error(1.3)):
        with tempfile.TemporaryDirectory() as carpeta:
            resultados, impresora = _correr(calibracion, carpeta, n=6, max_correcciones=2)

        assert sum(len(r.pasos) for r in resultados) >= 6
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
        assert (impresora.posicion_x, impresora.posicion_y) == INICIO[:2]


def test_si_la_foto_no_es_qvga_falla_antes_de_mover():
    impresora, camara = _armar(alto=480, ancho=640)

    with tempfile.TemporaryDirectory() as carpeta, sin_esperas():
        try:
            pc.probar_centrado(camara, impresora, pc.crear_calibracion(), Z_ARRIBA_SEGURO, carpeta,
                               dic_param_cam=DICT_PARAMS_CAMARA, devolveme_la_respuesta=False)
        except ValueError as error:
            assert "No se movió nada" in str(error)
        else:
            raise AssertionError("tendría que haber fallado: la foto no mide lo que supone la calibración")

    assert movimientos_g1(impresora) == []


def test_sin_diccionario_de_camara_falla_antes_de_mover():
    impresora, camara = _armar()

    with tempfile.TemporaryDirectory() as carpeta, sin_esperas():
        try:
            pc.probar_centrado(camara, impresora, pc.crear_calibracion(), Z_ARRIBA_SEGURO, carpeta,
                               devolveme_la_respuesta=False)
        except TypeError as error:
            assert "diccionario" in str(error)
        else:
            raise AssertionError("tendría que haber pedido el diccionario de parámetros de la cámara")

    assert movimientos_g1(impresora) == []


def test_sin_home_falla_con_un_mensaje_claro():
    impresora = crear_impresora_falsa(hacer_home=False)
    camara = CamaraFalsaConMuestra(impresora)

    with tempfile.TemporaryDirectory() as carpeta:
        try:
            pc.probar_centrado(camara, impresora, pc.crear_calibracion(), Z_ARRIBA_SEGURO, carpeta,
                               dic_param_cam=DICT_PARAMS_CAMARA, devolveme_la_respuesta=False)
        except RuntimeError as error:
            assert "home" in str(error)
        else:
            raise AssertionError("tendría que haber pedido home()")


# %%
