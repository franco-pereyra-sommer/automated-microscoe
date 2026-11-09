# %%

"""
Pruebas de busqueda_muestra.py (el módulo que combina cámara + impresora)
con la impresora real sobre serie simulado y una cámara falsa.

Desde el 2026-11-09 también hay pruebas de foco_final_multimetrica y de
buscar_foco_en_xy (la función que unió a buscar_muestra_en_z y posicionar_en_muestra)
con la cámara falsa: qué parámetros de cámara (los diccionarios) recibe cada etapa,
que un parámetro malo se rechaza ANTES de mover la platina, cómo va a (x, y)
(moverse_a_xy con z_seguro_xy = 40), la línea de base, la detección con fotos
seguidas, la confirmación y los límites de seguridad en Z; y las fotos y el
gráfico que guarda foco_final_multimetrica. NO prueban que los números de las
métricas sean los del equipo (la imagen es sintética, y su desenfoque dura mucho
más que el real).

Creado el 2026-10-27.
"""

import contextlib
import inspect
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


AUMENTO = {"varianza": 5.0}    # sobre el simulador: detecta cerca del foco (el real se mide)
TODOS_LOS_AUMENTOS = {"laplacian": 0.5, "tenengrad": 0.5, "varianza": 0.5}
PASO_GRUESO = 0.005


def _buscar(camara, impresora, **kwargs):
    kwargs.setdefault("x", X)
    kwargs.setdefault("y", Y)
    kwargs.setdefault("z_arriba_seguro", Z_ARRIBA_SEGURO)
    kwargs.setdefault("z_min_seguridad", Z_MIN_SEGURIDAD)
    kwargs.setdefault("dict_params_global", DICT_GRAYSCALE)
    kwargs.setdefault("aumento_deteccion", AUMENTO)
    return _en_silencio(busqueda_muestra.buscar_foco_en_xy, camara, impresora, **kwargs)


def _espiar_movimientos(impresora):
    """Lista de (nombre_del_metodo, x, y, z) con la posición DESPUÉS de cada desp_lin_000 / desp_lin_cam."""

    movimientos = []

    for nombre in ("desp_lin_000", "desp_lin_cam"):
        original = getattr(impresora, nombre)

        def espiar(*args, _original=original, _nombre=nombre, **kwargs):
            resultado = _original(*args, **kwargs)
            movimientos.append((_nombre, float(impresora.posicion_x), float(impresora.posicion_y),
                                float(impresora.posicion_z)))
            return resultado

        setattr(impresora, nombre, espiar)

    return movimientos


def test_buscar_foco_en_xy_encuentra_el_foco_y_configura_la_camara_una_sola_vez():
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)

    resultado = _buscar(camara, impresora)

    assert camara.llamadas_config_de_una == [DICT_GRAYSCALE]
    assert abs(resultado["z_confirmado"] - Z_FOCO) <= PASO_Z + 1e-9
    assert resultado["metrica_confirmada"] == "varianza"
    assert (resultado["x"], resultado["y"]) == (X, Y)
    # la platina queda en el máximo confirmado (autofocus() la deja ahí): el foco final se hace desde afuera
    assert abs(float(impresora.posicion_z) - resultado["z_confirmado"]) < 1e-9


def test_buscar_foco_en_xy_va_a_x_y_por_moverse_a_xy_y_viaja_a_z_seguro_40():
    assert inspect.signature(busqueda_muestra.buscar_foco_en_xy).parameters["z_seguro_xy"].default == 40.0

    # viaje largo: sube a 40 mm, se mueve en XY ARRIBA, y recién después baja a z_arriba_seguro
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)
    impresora.desp_lin_000(50.0, 50.0, Z_ARRIBA_SEGURO)
    movimientos = _espiar_movimientos(impresora)

    _buscar(camara, impresora)

    en_xy = [m for m in movimientos if m[0] == "desp_lin_000"]
    assert len(en_xy) == 1 and (en_xy[0][1], en_xy[0][2]) == (X, Y) and en_xy[0][3] == 40.0, en_xy
    assert movimientos[0][0] == "desp_lin_cam" and movimientos[0][3] == 40.0, movimientos[:2]
    assert movimientos[2][0] == "desp_lin_cam" and abs(movimientos[2][3] - Z_ARRIBA_SEGURO) < 1e-9, movimientos[:4]


def test_buscar_foco_en_xy_viaje_corto_mantiene_el_z_y_z_seguro_xy_se_puede_cambiar():
    # un salto de 0.1 mm (<= DISTANCIA_XY_SIN_SUBIR_MM) no sube: se mueve en XY con el Z que tenía
    impresora, camara = _preparar(z_inicial=27.7)
    movimientos = _espiar_movimientos(impresora)

    _buscar(camara, impresora, x=X + 0.1)

    en_xy = [m for m in movimientos if m[0] == "desp_lin_000"]
    assert len(en_xy) == 1 and abs(en_xy[0][3] - 27.7) < 1e-9, en_xy
    assert max(m[3] for m in movimientos) <= Z_ARRIBA_SEGURO + 1e-9      # nunca subió a 40

    # con otro z_seguro_xy, el viaje largo sube hasta ahí
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)
    impresora.desp_lin_000(50.0, 50.0, Z_ARRIBA_SEGURO)
    movimientos = _espiar_movimientos(impresora)

    _buscar(camara, impresora, z_seguro_xy=30.0)

    assert [m for m in movimientos if m[0] == "desp_lin_000"][0][3] == 30.0


def test_buscar_foco_en_xy_si_la_platina_no_llega_a_x_y_no_baja():
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)
    movimientos = _espiar_movimientos(impresora)

    _tiene_que_fallar(lambda: _buscar(camara, impresora, x=500.0), RuntimeError, "no llegó")

    assert min(m[3] for m in movimientos) >= Z_ARRIBA_SEGURO - 1e-9, movimientos     # a lo sumo subió
    assert camara.llamadas_config_de_una == [] and camara.capturas == 0


def test_buscar_foco_en_xy_la_linea_de_base_son_las_primeras_fotos_y_el_umbral_sale_de_ella():
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)

    resultado = _buscar(camara, impresora, aumento_deteccion=TODOS_LOS_AUMENTOS, pasos_linea_base=4)

    historial = resultado["historial_barrido_grueso"]
    fases = [h["fase"] for h in historial]
    assert fases[:4] == ["linea_base"] * 4 and set(fases[4:]) == {"barrido"}, fases[:8]

    # empieza en z_arriba_seguro y baja de a paso_grueso
    assert [round(h["z"], 4) for h in historial[:4]] == [27.9, 27.895, 27.89, 27.885]

    for nombre in busqueda_muestra.FUNCIONES_METRICA:
        promedio = sum(h[nombre] for h in historial[:4]) / 4
        assert abs(resultado["linea_base"][nombre] - promedio) < 1e-9, nombre

    for nombre, aumento in TODOS_LOS_AUMENTOS.items():
        assert abs(resultado["umbrales_deteccion"][nombre] - resultado["linea_base"][nombre] * (1 + aumento)) < 1e-9

    ultimo = resultado["candidatos"][-1]
    assert ultimo["resultado"] == "confirmado" and ultimo["metrica"] == resultado["metrica_confirmada"]
    assert abs(ultimo["umbral_confirmacion"] - ultimo["umbral_deteccion"] * 1.15) < 1e-9     # margen_confirmacion = 0.15


def test_buscar_foco_en_xy_usa_la_misma_regla_que_detectar_aumento_sobre_las_mediciones_guardadas():
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)

    resultado = _buscar(camara, impresora)

    historial = resultado["historial_barrido_grueso"]
    primero = metricas_imagen.detectar_aumento(historial, AUMENTO, pasos_linea_base=5, fotos_seguidas=2)

    assert primero is not None and primero["metrica"] == "varianza"
    assert abs(historial[primero["indice"]]["z"] - resultado["candidatos"][0]["z_disparo"]) < 1e-9


def test_buscar_foco_en_xy_pide_fotos_seguidas_antes_de_lanzar_autofocus():
    z_disparo = {}

    for seguidas in (1, 2, 3):
        impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)
        resultado = _buscar(camara, impresora, fotos_seguidas_para_detectar=seguidas)
        z_disparo[seguidas] = resultado["candidatos"][0]["z_disparo"]

    # cada foto seguida que se pide agrega un paso_grueso más abajo
    assert abs((z_disparo[1] - z_disparo[2]) - PASO_GRUESO) < 1e-9, z_disparo
    assert abs((z_disparo[2] - z_disparo[3]) - PASO_GRUESO) < 1e-9, z_disparo


def test_buscar_foco_en_xy_descarta_las_senales_falsas_y_cuenta_las_seguidas_desde_cero():
    for seguidas in (1, 3):
        impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)

        resultado = _buscar(camara, impresora, fotos_seguidas_para_detectar=seguidas)

        candidatos = resultado["candidatos"]
        assert len(candidatos) >= 2, "con este aumento tendría que haber señales falsas antes del foco"
        assert [c["resultado"] for c in candidatos[:-1]] == ["pico_no_confirmado"] * (len(candidatos) - 1)
        assert candidatos[-1]["resultado"] == "confirmado"

        # después de descartar, vuelve a pedir TODAS las seguidas: no reutiliza las fotos de antes
        for anterior, siguiente in zip(candidatos, candidatos[1:]):
            assert anterior["z_disparo"] - siguiente["z_disparo"] >= seguidas * PASO_GRUESO - 1e-9, (seguidas, candidatos)


def test_buscar_foco_en_xy_expone_los_parametros_de_la_confirmacion():
    original = busqueda_muestra.autofocus
    llamadas = []

    def espiar(*args, **kwargs):
        llamadas.append(kwargs)
        return original(*args, **kwargs)

    busqueda_muestra.autofocus = espiar

    try:
        # por defecto: pasos_confirmacion = 6, vecinos_para_confirmar = 3, pasos_extension_maxima = 3
        impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)
        _buscar(camara, impresora)
        assert {(k["pasos_arriba"], k["vecinos_para_confirmar"], k["pasos_extension_maxima"]) for k in llamadas} \
            == {(6, 3, 3)}, llamadas

        llamadas.clear()
        impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)
        _buscar(camara, impresora, pasos_confirmacion=4, vecinos_para_confirmar=2, pasos_extension_maxima=5)
        assert {(k["pasos_arriba"], k["vecinos_para_confirmar"], k["pasos_extension_maxima"]) for k in llamadas} \
            == {(4, 2, 5)}, llamadas

        # y siempre con los límites de seguridad
        assert all(k["z_min_seguridad"] == Z_MIN_SEGURIDAD and k["z_max_seguro"] == Z_ARRIBA_SEGURO for k in llamadas)
    finally:
        busqueda_muestra.autofocus = original

    assert inspect.signature(busqueda_muestra.buscar_foco_en_xy).parameters["pasos_confirmacion"].default == 6


def test_buscar_foco_en_xy_no_baja_nunca_de_z_min_seguridad():
    # el foco está por debajo del límite: no se puede llegar sin cruzarlo
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO, z_foco=Z_MIN_SEGURIDAD - 0.05)
    movimientos = _espiar_movimientos(impresora)

    _tiene_que_fallar(lambda: _buscar(camara, impresora), busqueda_muestra.MuestraNoEncontrada)

    assert min(m[3] for m in movimientos) >= Z_MIN_SEGURIDAD - 1e-9, min(m[3] for m in movimientos)


def test_buscar_foco_en_xy_en_un_campo_vacio_levanta_muestra_no_encontrada_con_lo_medido():
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)
    camara.con_muestra = False

    try:
        _buscar(camara, impresora)
    except busqueda_muestra.MuestraNoEncontrada as error:
        parcial = error.resultado_parcial
    else:
        raise AssertionError("un campo vacío tendría que terminar en MuestraNoEncontrada")

    assert parcial["candidatos"] == []
    assert abs(parcial["historial_barrido_grueso"][-1]["z"] - Z_MIN_SEGURIDAD) < 1e-9     # recorrió todo, hasta el límite
    assert parcial["historial_barrido_grueso"][0]["z"] == Z_ARRIBA_SEGURO
    assert set(parcial["linea_base"]) == set(busqueda_muestra.FUNCIONES_METRICA)
    assert (parcial["x"], parcial["y"]) == (X, Y)


def test_buscar_foco_en_xy_los_aumentos_son_obligatorios_y_se_rechazan_los_malos_antes_de_mover():
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)

    # sin los aumentos no corre
    _tiene_que_fallar(
        lambda: _en_silencio(busqueda_muestra.buscar_foco_en_xy, camara, impresora, X, Y, Z_ARRIBA_SEGURO,
                             Z_MIN_SEGURIDAD, DICT_GRAYSCALE),
        TypeError,
    )

    malos = [None, "varianza", {}, {"foco": 0.3}, {"varianza": 0}, {"varianza": -0.1}, {"varianza": "mucho"},
             {"varianza": True}, {"varianza": float("nan")}, {"laplacian": 0.3, "varianza": None}]

    for malo in malos:
        impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)

        _tiene_que_fallar(lambda: _buscar(camara, impresora, aumento_deteccion=malo), (ValueError, TypeError))

        assert impresora.ser.escrituras == [] and camara.llamadas_config_de_una == [], malo


def test_buscar_foco_en_xy_rechaza_los_parametros_malos_antes_de_mover():
    casos = [
        dict(dict_params_global={"pixformat": "GRAYSCALE", "resolucion": "QVGA"}),
        dict(dict_params_global="QVGA"),
        dict(dict_params_global=None),
        dict(z_seguro_xy=Z_ARRIBA_SEGURO - 0.1),        # la altura de viaje tiene que ser >= z_arriba_seguro
        dict(z_seguro_xy=200.0),                        # pasa el máximo de la impresora
        dict(z_min_seguridad=Z_ARRIBA_SEGURO),
        dict(pasos_linea_base=0),
        dict(pasos_linea_base=2.5),
        dict(pasos_linea_base=True),
        dict(pasos_linea_base=100),                     # no entra entre z_arriba_seguro y z_min_seguridad
        dict(fotos_seguidas_para_detectar=0),
        dict(vecinos_para_confirmar=0),
        dict(pasos_confirmacion=-1),
        dict(pasos_extension_maxima=1.5),
        dict(paso_grueso=0),
        dict(margen_confirmacion=-0.1),
    ]

    for kwargs in casos:
        impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)

        _tiene_que_fallar(lambda: _buscar(camara, impresora, **kwargs), (ValueError, TypeError))

        assert impresora.ser.escrituras == [] and camara.llamadas_config_de_una == [], kwargs


def test_buscar_foco_en_xy_sin_home_o_por_debajo_del_limite_falla_antes_de_mover():
    impresora = crear_impresora_falsa(hacer_home=False)
    camara = CamaraFalsa(impresora, z_foco=Z_FOCO)

    _tiene_que_fallar(lambda: _buscar(camara, impresora), RuntimeError, "home")
    assert camara.llamadas_config_de_una == []

    impresora, camara = _preparar(z_inicial=Z_MIN_SEGURIDAD - 0.2)

    _tiene_que_fallar(lambda: _buscar(camara, impresora), busqueda_muestra.RiesgoDeColision)
    assert impresora.ser.escrituras == [] and camara.llamadas_config_de_una == []


def test_buscar_foco_en_xy_ya_no_acepta_lo_de_las_dos_funciones_que_reemplaza():
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)

    for viejos in (dict(verificar_presencia=False), dict(alturas_verificacion=3), dict(umbral_intensidad_fondo=150),
                   dict(hacer_foco_final=False), dict(dict_params_local=DICT_LOCAL), dict(criterios_deteccion={}),
                   dict(foco_final_multimetrica_activado=True), dict(dict_params_color_foco_final=DICT_COLOR),
                   dict(resolucion_global="QVGA"), dict(pixformat_global="GRAYSCALE")):
        _tiene_que_fallar(lambda: _buscar(camara, impresora, **viejos), TypeError)

    assert impresora.ser.escrituras == []

    # y las dos funciones viejas ya no existen (sin alias: las aplicaciones viejas tienen que fallar fuerte)
    assert not hasattr(busqueda_muestra, "buscar_muestra_en_z")
    assert not hasattr(busqueda_muestra, "posicionar_en_muestra")


def test_buscar_foco_en_xy_no_llama_a_verificar_presencia_muestra_que_queda_suelta():
    original = busqueda_muestra.verificar_presencia_muestra
    llamadas = []

    busqueda_muestra.verificar_presencia_muestra = lambda *a, **k: llamadas.append(1) or True

    try:
        impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)
        _buscar(camara, impresora)
    finally:
        busqueda_muestra.verificar_presencia_muestra = original

    assert llamadas == []
    assert "SUELTA" in busqueda_muestra.verificar_presencia_muestra.__doc__
    assert issubclass(busqueda_muestra.CampoVisualSinMuestra, busqueda_muestra.MuestraNoEncontrada)
    assert "SUELTA" in busqueda_muestra.CampoVisualSinMuestra.__doc__


def test_la_verificacion_de_presencia_suelta_sigue_funcionando_por_su_cuenta():
    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)
    capturas = []
    capturar_original = camara.capturar
    camara.capturar = lambda: (capturas.append((camara.pixformat, camara.framesize)), capturar_original())[1]

    hay = _en_silencio(busqueda_muestra.verificar_presencia_muestra, camara, impresora, Z_ARRIBA_SEGURO,
                       Z_MIN_SEGURIDAD, resolucion="VGA", umbral_intensidad_fondo=100)

    assert hay is True and capturas[0] == ("RGB565", "VGA"), capturas[:2]

    impresora, camara = _preparar(z_inicial=Z_ARRIBA_SEGURO)
    camara.con_muestra = False

    assert _en_silencio(busqueda_muestra.verificar_presencia_muestra, camara, impresora, Z_ARRIBA_SEGURO,
                        Z_MIN_SEGURIDAD, umbral_intensidad_fondo=100) is False
