# %%

"""
Pruebas de busqueda_muestra.py (el módulo que combina cámara + impresora)
con la impresora real sobre serie simulado y una cámara falsa.

Desde el 2026-11-09 también hay pruebas de buscar_muestra_en_z,
foco_final_multimetrica y posicionar_en_muestra con la cámara falsa: se
centran en qué parámetros de cámara (los diccionarios) recibe cada etapa, en que
un diccionario malo se rechaza ANTES de mover la platina, y en las fotos y el
gráfico que guarda foco_final_multimetrica. NO prueban que los números de las
métricas sean los del equipo (la imagen es sintética).

Creado el 2026-10-27.
"""

import contextlib
import io
import re
import tempfile
from pathlib import Path

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


# ----------------------------------------------------------------------
# Parámetros de la cámara como diccionarios (2026-11-09)
# ----------------------------------------------------------------------

DICT_GRAYSCALE = {
    "pixformat": "GRAYSCALE",
    "resolucion": "QVGA",
    "exposure_us": 50000,
    "gain_db": 25,
    "whitebal": (66.22236, 61.43028, 60.206),
}
DICT_COLOR = {**DICT_GRAYSCALE, "pixformat": "RGB565", "resolucion": "WQXGA2"}
DICT_LOCAL = {**DICT_GRAYSCALE, "resolucion": "VGA"}

X, Y = 100.0, 100.0
Z_FOCO = 27.7
Z_ARRIBA_SEGURO = 27.9
Z_MIN_SEGURIDAD = 27.5

SEPARADOR = r"\(-?[\d.]+, -?[\d.]+, -?[\d.]+\)"
METRICAS = r"(?:laplacian|tenengrad|varianza)(?:\+(?:laplacian|tenengrad|varianza))*"
NOMBRE_MAXIMO = re.compile(rf"^{METRICAS} {SEPARADOR}\.jpg$")
NOMBRE_VECINO = re.compile(rf"^{METRICAS} vecino ([+-]\d+) {SEPARADOR}\.jpg$")
NOMBRE_PLOT = re.compile(rf"^plot metricas {SEPARADOR}\.png$")


def _preparar(z_inicial=Z_FOCO, z_foco=Z_FOCO):
    impresora = crear_impresora_falsa()
    impresora.desp_lin_000(X, Y, z_inicial)
    impresora.ser.escrituras.clear()
    return impresora, CamaraFalsa(impresora, z_foco=z_foco)


def _en_silencio(funcion, *args, **kwargs):
    with sin_esperas(), contextlib.redirect_stdout(io.StringIO()):
        return funcion(*args, **kwargs)


def _tiene_que_fallar(llamada, excepcion, contiene=None):
    try:
        llamada()
    except excepcion as error:
        if contiene is not None:
            assert contiene in str(error), str(error)
    else:
        raise AssertionError(f"tenía que tirar {excepcion.__name__}")


def _foco_final(camara, impresora, **kwargs):
    kwargs.setdefault("dict_params_grayscale", DICT_GRAYSCALE)
    kwargs.setdefault("dict_params_color", DICT_COLOR)
    kwargs.setdefault("z_min_seguridad", Z_MIN_SEGURIDAD)
    kwargs.setdefault("z_max_seguro", Z_ARRIBA_SEGURO)
    kwargs.setdefault("graficamos", False)
    return _en_silencio(busqueda_muestra.foco_final_multimetrica, camara, impresora, **kwargs)


def test_foco_final_multimetrica_barre_con_un_diccionario_y_saca_las_fotos_con_el_otro():
    impresora, camara = _preparar()

    _foco_final(camara, impresora)

    assert camara.llamadas_config_de_una == [DICT_GRAYSCALE, DICT_COLOR]
    assert (camara.pixformat, camara.framesize) == ("RGB565", "WQXGA2")


def test_foco_final_multimetrica_guarda_el_maximo_sus_vecinos_y_el_grafico():
    for vecinos in (1, 3):
        impresora, camara = _preparar()

        with tempfile.TemporaryDirectory() as carpeta:
            resultado = _foco_final(camara, impresora, vecinos_para_confirmar=vecinos,
                                    guardamos=True, carpeta_guardado=carpeta)

            nombres = sorted(archivo.name for archivo in Path(carpeta).iterdir())
            jpg = [n for n in nombres if n.endswith(".jpg")]
            png = [n for n in nombres if n.endswith(".png")]
            maximos = [n for n in jpg if NOMBRE_MAXIMO.match(n)]
            vecinos_guardados = [n for n in jpg if NOMBRE_VECINO.match(n)]

            # las tres métricas coinciden en este plano de foco: un máximo y `vecinos` fotos a cada lado
            assert len(maximos) == 1 and len(jpg) == len(maximos) + len(vecinos_guardados), nombres
            assert len(jpg) == 2 * vecinos + 1, nombres
            assert sorted(int(NOMBRE_VECINO.match(n).group(1)) for n in vecinos_guardados) == \
                [k for k in range(-vecinos, vecinos + 1) if k != 0]

            # una sola foto por altura
            alturas = [n.rsplit(", ", 1)[1].rstrip(").jpg") for n in jpg]
            assert len(alturas) == len(set(alturas)), alturas

            # el gráfico
            assert len(png) == 1 and NOMBRE_PLOT.match(png[0]), nombres
            assert (Path(carpeta) / png[0]).stat().st_size > 1000
            assert resultado["archivo_plot"] == Path(carpeta) / png[0]

            # el máximo de cada métrica quedó en su foto, y su ventana tiene las 2*vecinos+1 fotos
            for nombre in busqueda_muestra.FUNCIONES_METRICA:
                assert resultado[nombre]["archivo"].name == maximos[0]
                assert len(resultado[nombre]["archivos_ventana"]) == 2 * vecinos + 1

            # la platina termina en el máximo
            assert abs(float(impresora.posicion_z) - Z_FOCO) < 1e-9


def test_foco_final_multimetrica_sin_guardar_saca_solo_la_foto_del_maximo():
    impresora, camara = _preparar()

    with tempfile.TemporaryDirectory() as carpeta:
        resultado = _foco_final(camara, impresora, guardamos=False, carpeta_guardado=None)
        assert list(Path(carpeta).iterdir()) == []

    assert camara.archivos_guardados == []
    assert resultado["archivo_plot"] is None
    assert all(resultado[n]["archivo"] is None and resultado[n]["archivos_ventana"] == []
               for n in busqueda_muestra.FUNCIONES_METRICA)


def test_foco_final_multimetrica_no_fotografia_alturas_fuera_de_los_limites():
    z_min = Z_FOCO - 0.005     # justo 2 pasos por debajo del foco: los vecinos -3 quedarían fuera
    impresora, camara = _preparar()

    alturas = []  # Z absoluta después de cada desplazamiento
    desp_lin_cam_original = impresora.desp_lin_cam

    def espiar_desp_lin_cam(*args, **kwargs):
        resultado = desp_lin_cam_original(*args, **kwargs)
        alturas.append(float(impresora.posicion_z))
        return resultado

    impresora.desp_lin_cam = espiar_desp_lin_cam

    with tempfile.TemporaryDirectory() as carpeta:
        _foco_final(camara, impresora, z_min_seguridad=z_min, guardamos=True, carpeta_guardado=carpeta)
        vecinos = [int(NOMBRE_VECINO.match(a.name).group(1)) for a in Path(carpeta).glob("*.jpg")
                   if NOMBRE_VECINO.match(a.name)]

    assert -3 not in vecinos and {-2, -1, 1, 2, 3} == set(vecinos), sorted(vecinos)
    assert alturas and min(alturas) >= z_min - 1e-9, f"bajó hasta {min(alturas)}, el límite era {z_min}"


def test_foco_final_multimetrica_rechaza_un_diccionario_malo_antes_de_mover():
    casos = [
        dict(dict_params_grayscale={k: v for k, v in DICT_GRAYSCALE.items() if k != "gain_db"}),
        dict(dict_params_color={**DICT_COLOR, "exposicion": 50000}),          # clave que no existe
        dict(dict_params_color="RGB565"),
        dict(dict_params_grayscale=None),
    ]

    for kwargs in casos:
        impresora, camara = _preparar()

        _tiene_que_fallar(lambda: _foco_final(camara, impresora, **kwargs), (ValueError, TypeError))

        assert impresora.ser.escrituras == [] and camara.llamadas_config_de_una == [], kwargs


def test_foco_final_multimetrica_ya_no_acepta_los_parametros_sueltos_de_antes():
    impresora, camara = _preparar()

    for viejos in (dict(resolucion_grayscale="QVGA"), dict(pixformat_color="RGB565"), dict(resolucion_color="4K")):
        _tiene_que_fallar(lambda: _foco_final(camara, impresora, **viejos), TypeError)

    # y sin los diccionarios no corre
    _tiene_que_fallar(
        lambda: busqueda_muestra.foco_final_multimetrica(camara, impresora, z_min_seguridad=Z_MIN_SEGURIDAD),
        TypeError,
    )
    assert impresora.ser.escrituras == []


def _buscar(camara, impresora, **kwargs):
    kwargs.setdefault("dict_params_global", DICT_GRAYSCALE)
    return _en_silencio(busqueda_muestra.buscar_muestra_en_z, camara, impresora,
                        z_arriba_seguro=Z_ARRIBA_SEGURO, z_min_seguridad=Z_MIN_SEGURIDAD, **kwargs)


def test_buscar_muestra_en_z_configura_la_camara_con_dict_params_global():
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)

    resultado = _buscar(camara, impresora)

    assert camara.llamadas_config_de_una == [DICT_GRAYSCALE]
    assert abs(resultado["z_confirmado"] - Z_FOCO) <= PASO_Z + 1e-9


def test_la_verificacion_de_presencia_revisa_en_rgb565_con_la_resolucion_del_diccionario():
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)
    capturas = []
    capturar_original = camara.capturar
    camara.capturar = lambda: (capturas.append((camara.pixformat, camara.framesize)), capturar_original())[1]

    _buscar(camara, impresora, dict_params_global=DICT_LOCAL,        # VGA en el diccionario
            verificar_presencia=True, umbral_intensidad_fondo=100)

    assert capturas[0] == ("RGB565", "VGA"), capturas[:2]            # verificación: RGB565 a la resolución del diccionario
    assert capturas[-1] == ("GRAYSCALE", "VGA"), capturas[-2:]       # barrido y confirmación: el diccionario completo
    assert camara.llamadas_config_de_una == [DICT_LOCAL]


def test_buscar_muestra_en_z_rechaza_un_diccionario_malo_antes_de_mover():
    for malo in ({"pixformat": "GRAYSCALE", "resolucion": "QVGA"}, "QVGA", None):
        impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)

        _tiene_que_fallar(lambda: _buscar(camara, impresora, dict_params_global=malo), (ValueError, TypeError))

        assert impresora.ser.escrituras == [] and camara.llamadas_config_de_una == [], malo


def _posicionar(camara, impresora, **kwargs):
    return _en_silencio(busqueda_muestra.posicionar_en_muestra, camara, impresora, X, Y,
                        Z_ARRIBA_SEGURO, Z_MIN_SEGURIDAD, **kwargs)


def test_posicionar_en_muestra_una_metrica_usa_dict_params_local_solo_si_se_lo_pasan():
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)
    resultado = _posicionar(camara, impresora, dict_params_global=DICT_GRAYSCALE)

    assert camara.llamadas_config_de_una == [DICT_GRAYSCALE]       # el foco final sigue con la configuración global
    assert abs(resultado["z_foco_final"] - Z_FOCO) <= PASO_Z + 1e-9

    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)
    _posicionar(camara, impresora, dict_params_global=DICT_GRAYSCALE, dict_params_local=DICT_LOCAL)

    assert camara.llamadas_config_de_una == [DICT_GRAYSCALE, DICT_LOCAL]


def test_posicionar_en_muestra_sin_foco_final_configura_la_camara_una_sola_vez():
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)

    _posicionar(camara, impresora, dict_params_global=DICT_GRAYSCALE, hacer_foco_final=False)

    assert camara.llamadas_config_de_una == [DICT_GRAYSCALE]


def test_posicionar_en_muestra_multimetrica_pasa_los_diccionarios_a_foco_final_multimetrica():
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)

    with tempfile.TemporaryDirectory() as carpeta:
        resultado = _posicionar(camara, impresora, dict_params_global=DICT_GRAYSCALE,
                                foco_final_multimetrica_activado=True, dict_params_color_foco_final=DICT_COLOR,
                                guardar_fotos_foco_final=True, carpeta_fotos_foco_final=carpeta)

        # búsqueda (global) -> barrido del foco final (sin local: el global) -> fotos (color)
        assert camara.llamadas_config_de_una == [DICT_GRAYSCALE, DICT_GRAYSCALE, DICT_COLOR]
        assert len(list(Path(carpeta).glob("*.jpg"))) == 7        # el máximo y 3 vecinos a cada lado
        assert len(list(Path(carpeta).glob("*.png"))) == 1

    assert resultado["foco_final_multimetrica"]["laplacian"]["confirmado"]

    # con dict_params_local, ese es el del barrido del foco final
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)
    _posicionar(camara, impresora, dict_params_global=DICT_GRAYSCALE, dict_params_local=DICT_LOCAL,
                foco_final_multimetrica_activado=True, dict_params_color_foco_final=DICT_COLOR)

    assert camara.llamadas_config_de_una == [DICT_GRAYSCALE, DICT_LOCAL, DICT_COLOR]


def test_posicionar_en_muestra_rechaza_un_diccionario_malo_o_faltante_antes_de_mover():
    casos = [
        dict(dict_params_global={"pixformat": "GRAYSCALE"}),
        dict(dict_params_global=DICT_GRAYSCALE, dict_params_local="QVGA"),
        # multimétrica sin el diccionario de las fotos en color (o con uno incompleto)
        dict(dict_params_global=DICT_GRAYSCALE, foco_final_multimetrica_activado=True),
        dict(dict_params_global=DICT_GRAYSCALE, foco_final_multimetrica_activado=True,
             dict_params_color_foco_final={"pixformat": "RGB565", "resolucion": "WQXGA2"}),
    ]

    for kwargs in casos:
        impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)

        _tiene_que_fallar(lambda: _posicionar(camara, impresora, **kwargs), (ValueError, TypeError))

        assert impresora.ser.escrituras == [] and camara.llamadas_config_de_una == [], kwargs

    # sin foco final multimétrica, el diccionario de color no hace falta
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)
    _posicionar(camara, impresora, dict_params_global=DICT_GRAYSCALE, foco_final_multimetrica_activado=False)


def test_posicionar_en_muestra_ya_no_acepta_los_parametros_sueltos_de_antes():
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)

    for viejos in (dict(resolucion_global="QVGA"), dict(pixformat_global="GRAYSCALE"), dict(resolucion_local="QVGA"),
                   dict(pixformat_color_foco_final="RGB565"), dict(resolucion_color_foco_final="4K")):
        _tiene_que_fallar(lambda: _posicionar(camara, impresora, dict_params_global=DICT_GRAYSCALE, **viejos),
                          TypeError)

    assert impresora.ser.escrituras == []
