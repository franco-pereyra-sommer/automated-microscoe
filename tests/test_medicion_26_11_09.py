# %%

"""
Pruebas de la aplicación "26.11.09 medicion de aumentos y parametros de camara" (barrido_con_fotos.py,
analisis_de_fotos.py y los scripts de celdas), con la impresora REAL sobre serie simulado y la cámara falsa
(CamaraFalsa, con un plano de foco simulado en Z = 27.75).

Qué se verifica:
  - el barrido baja de z_arriba_seguro a z_min_seguridad con una foto por altura, nunca por debajo de
    z_min_seguridad, configura la cámara UNA vez y vuelve arriba;
  - los nombres de las fotos traen (x, y, z) y info_camara.csv tiene una fila por foto (también si la cámara
    no contesta el info);
  - todo se rechaza ANTES de mover (diccionario de cámara, carpeta con fotos, alturas, platina ya muy abajo);
  - el análisis lee las fotos, ve el pico de las métricas donde está el foco, arma la línea de base como la
    búsqueda y los gráficos se dibujan;
  - el script de medición (con la conexión reemplazada por los simuladores) hace un barrido por cada
    combinación de exposición y ganancia.

Lo que NO prueban: tiempos reales, vibraciones, el sensor con exposición automática ni la óptica reales.
Que pasen no reemplaza correr la medición en el equipo.

Creado el 2026-11-09.
"""

import math
import os
import re
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

from simuladores import RAIZ, CamaraFalsa, crear_impresora_falsa, sin_esperas
from test_estatico import nombres_no_definidos

CARPETA_APP = RAIZ / "26.11.09 medicion de aumentos y parametros de camara"

sys.path.insert(0, str(CARPETA_APP))

import analisis_de_fotos as af  # noqa: E402
import barrido_con_fotos as bf  # noqa: E402
from busqueda_muestra import RiesgoDeColision  # noqa: E402

X, Y = 150.0, 150.0
Z_FOCO = 27.75
Z_ARRIBA_SEGURO = 27.9
Z_MIN_SEGURIDAD = 27.6
PARAMS = {"pixformat": "RGB565", "resolucion": "QVGA", "exposure_us": None, "gain_db": 25, "whitebal": (66.2, 61.4, 60.2)}


def _preparar():
    impresora = crear_impresora_falsa()
    camara = CamaraFalsa(impresora, z_foco=Z_FOCO)
    return impresora, camara


def _barrer(camara, impresora, carpeta, **kwargs):
    argumentos = dict(
        x=X, y=Y, z_arriba_seguro=Z_ARRIBA_SEGURO, z_min_seguridad=Z_MIN_SEGURIDAD,
        dict_params_camara=PARAMS, carpeta_guardado=carpeta,
    )
    argumentos.update(kwargs)

    with sin_esperas():
        return bf.barrido_con_fotos(camara, impresora, **argumentos)


def _espiar_movimientos(impresora):
    """Lista de (nombre_del_metodo, z) con la Z DESPUÉS de cada desp_lin_000 / desp_lin_cam."""

    movimientos = []

    for nombre in ("desp_lin_000", "desp_lin_cam"):
        original = getattr(impresora, nombre)

        def espiar(*args, _original=original, _nombre=nombre, **kwargs):
            resultado = _original(*args, **kwargs)
            movimientos.append((_nombre, float(impresora.posicion_z)))
            return resultado

        setattr(impresora, nombre, espiar)

    return movimientos


def _tiene_que_fallar(excepcion, funcion, *args, **kwargs):
    try:
        funcion(*args, **kwargs)
    except excepcion as motivo:
        return motivo

    raise AssertionError(f"tenía que levantar {excepcion.__name__}")


# ============================================================
# EL BARRIDO
# ============================================================

def test_el_barrido_baja_de_z_arriba_a_z_min_con_una_foto_por_altura_y_vuelve_arriba():
    impresora, camara = _preparar()

    with tempfile.TemporaryDirectory() as tmp:
        resultado = _barrer(camara, impresora, Path(tmp) / "barrido")

        zs = resultado["z"]

        assert len(zs) == 121 == len(resultado["archivos"]) == camara.capturas, (len(zs), camara.capturas)
        assert abs(zs[0] - Z_ARRIBA_SEGURO) < 1e-6 and abs(zs[-1] - Z_MIN_SEGURIDAD) < 1e-6, (zs[0], zs[-1])
        assert all(abs((a - b) - 0.0025) < 1e-6 for a, b in zip(zs, zs[1:])), "los pasos no son de 2.5 µm"
        assert all(archivo.exists() for archivo in resultado["archivos"])

    # terminó arriba, en (x, y)
    assert abs(impresora.posicion_z - Z_ARRIBA_SEGURO) < 1e-6
    assert (impresora.posicion_x, impresora.posicion_y) == (X, Y)


def test_las_fotos_traen_x_y_z_en_el_nombre_y_el_csv_tiene_una_fila_por_foto():
    impresora, camara = _preparar()

    with tempfile.TemporaryDirectory() as tmp:
        resultado = _barrer(camara, impresora, Path(tmp) / "barrido")

        for archivo, z in zip(resultado["archivos"], resultado["z"]):
            x, y, z_nombre = af.extraer_coordenadas(archivo)
            assert (x, y) == (X, Y) and abs(z_nombre - z) < 1e-9, (archivo.name, z)

        info = af._leer_info_camara(resultado["carpeta"])

        assert len(info) == 121 and set(info) == {a.name for a in resultado["archivos"]}
        una = next(iter(info.values()))
        assert una["exposure_us"] == 18000 and una["gain_db"] == 25.0           # exposición automática en la cámara falsa
        assert una["rgb_r_db"] == 66.2 and una["rgb_b_db"] == 60.2
        assert resultado["fotos_sin_info"] == 0


def test_configura_la_camara_una_sola_vez_con_el_diccionario():
    impresora, camara = _preparar()

    with tempfile.TemporaryDirectory() as tmp:
        _barrer(camara, impresora, Path(tmp) / "barrido")

    assert camara.llamadas_config_de_una == [PARAMS], camara.llamadas_config_de_una


def test_nunca_baja_de_z_min_seguridad_y_viaja_en_xy_a_z_seguro():
    impresora, camara = _preparar()
    impresora.desp_lin_000(50.0, 50.0, 30.0)
    movimientos = _espiar_movimientos(impresora)

    with tempfile.TemporaryDirectory() as tmp:
        _barrer(camara, impresora, Path(tmp) / "barrido")

    assert min(z for _, z in movimientos) >= Z_MIN_SEGURIDAD - 1e-9, movimientos[-3:]

    en_xy = [m for m in movimientos if m[0] == "desp_lin_000"]
    assert len(en_xy) == 1 and en_xy[0][1] == 40.0, en_xy         # un solo viaje en XY, a la altura de viaje


def test_un_paso_que_no_divide_el_rango_termina_justo_en_z_min():
    impresora, camara = _preparar()

    with tempfile.TemporaryDirectory() as tmp:
        resultado = _barrer(camara, impresora, Path(tmp) / "barrido", paso_z=0.0125)    # 0.3 mm / 12.5 µm = 24 pasos justos

        assert abs(resultado["z"][-1] - Z_MIN_SEGURIDAD) < 1e-6
        assert len(resultado["z"]) == 25

    impresora, camara = _preparar()

    with tempfile.TemporaryDirectory() as tmp:
        resultado = _barrer(camara, impresora, Path(tmp) / "barrido", paso_z=0.11)      # 0.3 mm no es múltiplo de 0.11

        # el último paso se acorta para no pasar de z_min_seguridad: 27.9, 27.79, 27.68 y 27.6
        assert len(resultado["z"]) == 4 and abs(resultado["z"][-1] - Z_MIN_SEGURIDAD) < 1e-6, resultado["z"]
        assert min(resultado["z"]) >= Z_MIN_SEGURIDAD - 1e-9, resultado["z"]


def test_la_camara_que_no_informa_no_corta_el_barrido():
    impresora, camara = _preparar()
    camara.info_falla = True

    with tempfile.TemporaryDirectory() as tmp:
        resultado = _barrer(camara, impresora, Path(tmp) / "barrido")

        assert resultado["fotos_sin_info"] == 121 and len(resultado["archivos"]) == 121

        info = af._leer_info_camara(resultado["carpeta"])

        assert all(math.isnan(fila["exposure_us"]) for fila in info.values()) and len(info) == 121


def test_sin_guardar_info_no_se_le_pide_nada_a_la_camara():
    impresora, camara = _preparar()

    with tempfile.TemporaryDirectory() as tmp:
        _barrer(camara, impresora, Path(tmp) / "barrido", guardar_info_camara=False)

    assert camara.llamadas_info == 0


# ============================================================
# TODO SE RECHAZA ANTES DE MOVER
# ============================================================

def _no_movio_nada(impresora, camara, posicion_inicial):
    assert camara.capturas == 0 and camara.llamadas_config_de_una == []
    assert (impresora.posicion_x, impresora.posicion_y, impresora.posicion_z) == posicion_inicial


def test_rechaza_antes_de_mover_un_diccionario_malo_alturas_malas_y_una_carpeta_con_fotos():
    impresora, camara = _preparar()
    inicio = (impresora.posicion_x, impresora.posicion_y, impresora.posicion_z)

    with tempfile.TemporaryDirectory() as tmp:
        carpeta = Path(tmp) / "barrido"

        faltante = {k: v for k, v in PARAMS.items() if k != "whitebal"}
        _tiene_que_fallar(ValueError, _barrer, camara, impresora, carpeta, dict_params_camara=faltante)
        _tiene_que_fallar(TypeError, _barrer, camara, impresora, carpeta, dict_params_camara="QVGA")
        _tiene_que_fallar(ValueError, _barrer, camara, impresora, carpeta, paso_z=0)
        _tiene_que_fallar(ValueError, _barrer, camara, impresora, carpeta, z_min_seguridad=27.95)
        _tiene_que_fallar(ValueError, _barrer, camara, impresora, carpeta, z_seguro_xy=27.0)
        _tiene_que_fallar(ValueError, _barrer, camara, impresora, carpeta, z_seguro_xy=500.0)

        carpeta.mkdir()
        (carpeta / "ya_estaba.jpg").write_bytes(b"x")
        _tiene_que_fallar(FileExistsError, _barrer, camara, impresora, carpeta)

    _no_movio_nada(impresora, camara, inicio)


def test_si_la_platina_ya_esta_por_debajo_de_z_min_levanta_riesgo_de_colision_sin_mover():
    impresora, camara = _preparar()
    impresora.desp_lin_000(X, Y, 27.5)
    inicio = (impresora.posicion_x, impresora.posicion_y, impresora.posicion_z)

    with tempfile.TemporaryDirectory() as tmp:
        _tiene_que_fallar(RiesgoDeColision, _barrer, camara, impresora, Path(tmp) / "barrido")

    _no_movio_nada(impresora, camara, inicio)


def test_si_la_platina_no_llega_a_x_y_no_baja():
    impresora, camara = _preparar()

    with tempfile.TemporaryDirectory() as tmp:
        motivo = _tiene_que_fallar(RuntimeError, _barrer, camara, impresora, Path(tmp) / "barrido", x=500.0)

    assert "no llegó" in str(motivo)
    assert camara.capturas == 0 and impresora.posicion_z == 40.0      # se quedó en la altura de viaje


def test_si_la_impresora_se_pasa_de_z_min_levanta_riesgo_de_colision():
    # defensa extra: la platina redondea hacia abajo en la grilla de 2.5 µm; si un movimiento la dejara por debajo de
    # z_min_seguridad, el barrido se detiene en vez de seguir sacando fotos
    impresora, camara = _preparar()
    original = impresora.desp_lin_cam

    def baja_de_mas(x=None, y=None, z=None, **kwargs):
        return original(x, y, z - 0.0025 if z is not None and z < 0 else z, **kwargs)

    impresora.desp_lin_cam = baja_de_mas

    with tempfile.TemporaryDirectory() as tmp:
        motivo = _tiene_que_fallar(RiesgoDeColision, _barrer, camara, impresora, Path(tmp) / "barrido")

    assert "z_min_seguridad" in str(motivo)


# ============================================================
# EL NOMBRE DE LA CARPETA
# ============================================================

def test_el_nombre_de_carpeta_dice_los_parametros_y_distingue_automatico_de_fijo():
    auto = dict(PARAMS, exposure_us=None, gain_db=None, whitebal=None)
    fijo = dict(PARAMS, exposure_us=15000, gain_db=25)

    nombre_auto = bf.nombre_de_carpeta_de_parametros(auto)
    nombre_fijo = bf.nombre_de_carpeta_de_parametros(fijo)

    assert "Exposure auto" in nombre_auto and "Gain auto" in nombre_auto and "Whitebal auto" in nombre_auto
    assert "Exposure 15000" in nombre_fijo and "Gain 25" in nombre_fijo and "Whitebal 66.2-61.4-60.2" in nombre_fijo
    assert nombre_auto != nombre_fijo
    assert nombre_fijo.startswith("Pixformat RGB565 - resolucion QVGA")

    _tiene_que_fallar(ValueError, bf.nombre_de_carpeta_de_parametros, dict(PARAMS, exposure_us=-5))


# ============================================================
# EL ANÁLISIS
# ============================================================

def _medir(tmp, **params):
    impresora, camara = _preparar()
    resultado = _barrer(camara, impresora, Path(tmp) / bf.nombre_de_carpeta_de_parametros(dict(PARAMS, **params)),
                        dict_params_camara=dict(PARAMS, **params))
    return resultado["carpeta"]


def test_el_analisis_ve_el_pico_de_las_metricas_en_el_foco_y_trae_lo_que_informo_la_camara():
    with tempfile.TemporaryDirectory() as tmp:
        resultados = af.analizar_carpeta(_medir(tmp))

    assert len(resultados) == 121
    assert [r["z"] for r in resultados] == sorted(r["z"] for r in resultados)

    for metrica in ("laplacian", "tenengrad", "varianza"):
        z_pico = max(resultados, key=lambda r: r[metrica])["z"]
        assert abs(z_pico - Z_FOCO) < 0.005, (metrica, z_pico)

    assert all(r["gain_db"] == 25.0 for r in resultados)
    assert all(set(r) >= {"intensidad_fondo", "fraccion_rojo", "saturacion_media", "media_r"} for r in resultados)


def test_sin_info_camara_csv_las_columnas_de_la_camara_quedan_en_nan():
    with tempfile.TemporaryDirectory() as tmp:
        carpeta = _medir(tmp)
        (carpeta / bf.NOMBRE_CSV_INFO_CAMARA).unlink()

        resultados = af.analizar_carpeta(carpeta)

    assert len(resultados) == 121 and all(math.isnan(r["exposure_us"]) for r in resultados)


def test_una_carpeta_sin_fotos_se_rechaza():
    with tempfile.TemporaryDirectory() as tmp:
        _tiene_que_fallar(ValueError, af.analizar_carpeta, tmp)
        _tiene_que_fallar(ValueError, af.analizar_medicion, tmp)


def test_serie_del_descenso_va_de_arriba_a_abajo_y_saltea_fotos_para_emular_el_paso_grueso():
    with tempfile.TemporaryDirectory() as tmp:
        resultados = af.analizar_carpeta(_medir(tmp))

    serie = af.serie_del_descenso(resultados, paso_grueso=0.005)
    zs = [fila["z"] for fila in serie]

    assert len(serie) == 61 and abs(zs[0] - Z_ARRIBA_SEGURO) < 1e-6 and abs(zs[-1] - Z_MIN_SEGURIDAD) < 1e-6
    assert all(abs((a - b) - 0.005) < 1e-6 for a, b in zip(zs, zs[1:]))
    assert len(af.serie_del_descenso(resultados, paso_grueso=0.0025)) == 121      # si el paso es el mismo, no se saltea nada


def test_el_resumen_usa_la_misma_linea_de_base_que_la_busqueda():
    import metricas_imagen

    with tempfile.TemporaryDirectory() as tmp:
        resultados = af.analizar_carpeta(_medir(tmp))

    resumen = af.resumen_por_metrica(resultados, pasos_linea_base=5, paso_grueso=0.005)

    serie = af.serie_del_descenso(resultados, 0.005)
    esperada = metricas_imagen.linea_de_base(serie, metricas_imagen.FUNCIONES_METRICA, 5)

    for nombre, datos in resumen.items():
        assert datos["linea_base"] == esperada[nombre]
        assert datos["pico_relativo"] > 3, (nombre, datos)           # en el foco simulado la métrica sube mucho
        assert abs(datos["z_pico"] - Z_FOCO) < 0.006, (nombre, datos)
        assert 0 <= datos["ruido_relativo"] < 0.5, (nombre, datos)

    _tiene_que_fallar(ValueError, af.resumen_por_metrica, resultados, pasos_linea_base=500)


def test_analizar_medicion_arma_una_etiqueta_por_combinacion_con_lo_que_cambia():
    with tempfile.TemporaryDirectory() as tmp:
        _medir(tmp, exposure_us=15000)
        _medir(tmp, exposure_us=50000)
        (Path(tmp) / "una carpeta sin fotos").mkdir()

        resultados_por_etiqueta = af.analizar_medicion(tmp)

    assert sorted(resultados_por_etiqueta) == ["Exposure 15000", "Exposure 50000"], sorted(resultados_por_etiqueta)

    assert af.etiquetas_cortas(["a - b - c", "a - x - c"]) == ["b", "x"]
    assert af.etiquetas_cortas(["a - b", "a - b"]) == ["a - b", "a - b"]
    assert af.etiquetas_cortas(["unica"]) == ["unica"]


def test_los_graficos_se_dibujan_y_el_csv_se_exporta():
    with tempfile.TemporaryDirectory() as tmp:
        _medir(tmp, exposure_us=15000)
        _medir(tmp, exposure_us=50000)

        resultados_por_etiqueta = af.analizar_medicion(tmp)

        figura = af.graficar_metricas(resultados_por_etiqueta, titulo="prueba")
        assert len(figura.axes) == 8
        assert all(len(eje.lines) == 2 for eje in figura.axes[:3])
        plt.close(figura)

        figura = af.graficar_relativo_a_la_linea_de_base(resultados_por_etiqueta, aumentos_candidatos=[0.3, 1.0])
        assert len(figura.axes) == 3 and all(len(eje.lines) >= 2 + 3 for eje in figura.axes)   # 2 curvas + 1 + 2 candidatos
        plt.close(figura)

        af.imprimir_resumen(resultados_por_etiqueta)

        csv_salida = Path(tmp) / "analisis.csv"
        af.exportar_csv(next(iter(resultados_por_etiqueta.values())), csv_salida)

        assert csv_salida.exists() and len(csv_salida.read_text(encoding="utf-8").splitlines()) == 122


# ============================================================
# LOS SCRIPTS DE CELDAS
# ============================================================

def test_los_archivos_de_la_aplicacion_no_usan_nombres_sin_definir_ni_funciones_que_ya_no_existen():
    for nombre in ("barrido_con_fotos", "analisis_de_fotos", "correr_medicion", "correr_analisis"):
        codigo = (CARPETA_APP / f"{nombre}.py").read_text(encoding="utf-8")

        faltan = nombres_no_definidos(codigo, f"{nombre}.py")
        assert not faltan, f"{nombre}.py usa nombres sin definir: {sorted(faltan)}"

        assert not re.search(r"\b(posicionar_en_muestra|buscar_muestra_en_z|metricas_enfoque|desp_lin_abs|desp_lin_rel)\s*\(", codigo), nombre
        assert not re.search(r"^\s*(from|import)\s+(Main|metricas_enfoque)\b", codigo, re.M), nombre


def test_el_script_de_medicion_hace_un_barrido_por_cada_combinacion_de_exposicion_y_ganancia():
    texto = (CARPETA_APP / "correr_medicion.py").read_text(encoding="utf-8")

    # La celda de conexión abriría los puertos reales: se la reemplaza por los simuladores. Se achican las listas
    # para que la prueba sea rápida, y la salida va a una carpeta temporal.
    celdas = re.split(r"^#%%.*$", texto, flags=re.M)
    sin_conexion = [celda for celda in celdas if "TrimakerCOSMOS(\"COM11\")" not in celda]
    assert len(sin_conexion) == len(celdas) - 1

    with tempfile.TemporaryDirectory() as tmp:
        codigo = "\n".join(sin_conexion)
        codigo = re.sub(r"^CARPETA_SALIDA = .*$", f"CARPETA_SALIDA = {str(tmp)!r}", codigo, flags=re.M)
        codigo = re.sub(r"^EXPOSICIONES_US = \[.*?\]", "EXPOSICIONES_US = [None, 15000]", codigo, flags=re.M)
        codigo = re.sub(r"^GANANCIAS_DB = \[.*?\]", "GANANCIAS_DB = [None, 35]", codigo, flags=re.M)
        codigo = re.sub(r"^ESPERA_MOVIMIENTO = [0-9.]+", "ESPERA_MOVIMIENTO = 0", codigo, flags=re.M)

        impresora, camara = _preparar()

        entorno_antes = os.environ.get("PYTHONPATH")
        ruta_antes = list(sys.path)

        try:
            with sin_esperas():
                exec(compile(codigo, "correr_medicion.py", "exec"), {"__name__": "correr_medicion", "grua": impresora, "cam": camara})
        finally:
            sys.path[:] = ruta_antes
            if entorno_antes is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = entorno_antes

        carpetas = sorted(p for p in Path(tmp).rglob("Pixformat*") if p.is_dir())

        assert len(carpetas) == 4, [c.name for c in carpetas]
        assert all(len(list(c.glob("*.jpg"))) == 121 for c in carpetas)
        assert {c.parent.name for c in carpetas} == {"muestra 1 xy 150_150 - pasada 1"}

        # el diccionario de cada barrido sale de los for's: exposición x ganancia, y la cámara se configura una vez por barrido
        combinaciones = [(p["exposure_us"], p["gain_db"]) for p in camara.llamadas_config_de_una]
        assert combinaciones == [(None, None), (None, 35), (15000, None), (15000, 35)], combinaciones
        assert all(p["whitebal"] == (66.22236, 61.43028, 60.206) and p["pixformat"] == "RGB565" for p in camara.llamadas_config_de_una)

        # y el análisis lee todo lo que dejó el script
        resultados_por_etiqueta = af.analizar_medicion(next(iter(carpetas)).parent)
        assert len(resultados_por_etiqueta) == 4


# %%
