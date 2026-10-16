"""
Pruebas con el microscopio simulado:  python test_barrido_simulado.py   (o:  pytest -q)

Qué se verifica:
  1. cuantización: el residuo tras ir a la posición alcanzable es <= 6.25 µm por eje (celda primitiva)
  2. detector: el centro sale a < 1 px del verdadero, y no hay falsos positivos con glóbulos y señuelos magenta
  3. registro de posiciones visitadas: tolerancia de 1 paso y persistencia
  4. calibrar_escala: recupera escala y signo de cada eje
  5. barrido completo: una foto por objeto (incluyendo un par superpuesto), centrada en la celda
  6. escala mal calibrada (25 %): la verificación corrige el grueso (y evita fotos duplicadas); sin verificar, no
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from barrido_puntos import ConfigBarrido, barrer, campos_espiral
from centrado import (PASO_MM, Calibracion, RegistroVisitados, a_pasos, calibrar_escala, centrar_pixel)
from deteccion_puntos import DetectorColor
from simulador_microscopio import MicroscopioSimulado

# Campo visual medido con las fotos 4K: 0.214 x 0.286 mm (x = filas, y = columnas). Acá se lo supone igual en QVGA
# (320 x 240 px  ->  ~0.89 µm/px); calibrar_escala() lo mide de verdad con la cámara real.
NOMINAL = Calibracion.desde_fov_mm(320, 240, fov_x_mm=0.214, fov_y_mm=0.286)
MITAD_UM = PASO_MM * 1000 / 2   # 6.25 µm


# ---------------------------------------------------------------- 1
def test_cuantizacion():
    rng = np.random.default_rng(1)
    for _ in range(2000):
        pos = (a_pasos(rng.uniform(100, 200)) * PASO_MM, a_pasos(rng.uniform(100, 200)) * PASO_MM)
        col, fila = rng.uniform(0, 319), rng.uniform(0, 239)
        o = centrar_pixel(NOMINAL, pos, col, fila)
        assert isinstance(o.pasos_x, int) and isinstance(o.pasos_y, int)
        assert abs(o.residuo_um[0]) <= MITAD_UM + 1e-6 and abs(o.residuo_um[1]) <= MITAD_UM + 1e-6
        assert o.dentro_de_celda
        # el píxel esperado tras mover cae a <= medio paso del centro, medido en píxeles
        assert abs(o.px_esperado[1] - NOMINAL.centro_fila) <= MITAD_UM / NOMINAL.um_por_px_fila + 1e-6
        assert abs(o.px_esperado[0] - NOMINAL.centro_col) <= MITAD_UM / NOMINAL.um_por_px_col + 1e-6
    # el píxel central no pide moverse
    o = centrar_pixel(NOMINAL, (150.0, 150.0), NOMINAL.centro_col, NOMINAL.centro_fila)
    assert (o.pasos_x, o.pasos_y) == (a_pasos(150.0), a_pasos(150.0))
    # signo: píxel por debajo del centro -> x aumenta; píxel a la derecha -> y aumenta
    o = centrar_pixel(NOMINAL, (150.0, 150.0), NOMINAL.centro_col + 100, NOMINAL.centro_fila + 100)
    assert o.x_mm > 150.0 and o.y_mm > 150.0
    print("ok  cuantización")


# ---------------------------------------------------------------- 2
def test_detector():
    pt = (150.020, 149.970)   # +20 µm en x (filas), -30 µm en y (columnas)
    sim = MicroscopioSimulado(NOMINAL, [pt], semilla=3)
    dets = DetectorColor().detectar(sim.capturar_campo())
    assert len(dets) == 1, dets
    d = dets[0]
    esp_fila = NOMINAL.centro_fila + 20 / NOMINAL.um_por_px_fila
    esp_col = NOMINAL.centro_col - 30 / NOMINAL.um_por_px_col
    err = np.hypot(d.col - esp_col, d.fila - esp_fila)
    assert err < 1.0, (d, esp_col, esp_fila, err)
    # campo sin puntos (con glóbulos y señuelos magenta): cero detecciones, en varias posiciones
    vacio = MicroscopioSimulado(NOMINAL, [], semilla=4)
    for dx in (0.0, 0.05, -0.1, 0.2):
        vacio.mover_a(150.0 + dx, 150.0 - dx)
        assert DetectorColor().detectar(vacio.capturar_campo()) == []
    print(f"ok  detector  (error de centro {err:.2f} px = {err * NOMINAL.um_por_px_col:.2f} µm, diámetro {d.diametro_px:.1f} px)")


# ---------------------------------------------------------------- 3
def test_registro():
    r = RegistroVisitados(tol_pasos=1)
    r.registrar([(12000, 12000)], archivo="a.png")
    assert r.ya_visitada(12000, 12000)
    assert r.ya_visitada(12001, 11999)        # 12.5 µm por eje: repetido
    assert not r.ya_visitada(12002, 12000)    # 25 µm: es otro lugar
    assert not r.ya_visitada(12000, 12002)
    with tempfile.TemporaryDirectory() as t:
        r.guardar(Path(t) / "r.json")
        r2 = RegistroVisitados.cargar(Path(t) / "r.json")
    assert len(r2) == 1 and r2.ya_visitada(12001, 12001) and not r2.ya_visitada(12003, 12000)
    print("ok  registro")


# ---------------------------------------------------------------- 4
def test_calibrar_escala():
    for (sf, sc, sgx, sgy) in [(0.90, 0.90, +1, +1), (0.85, 0.95, -1, +1), (0.92, 0.88, +1, -1)]:
        real = Calibracion(320, 240, um_por_px_fila=sf, um_por_px_col=sc, signo_x=sgx, signo_y=sgy)
        sim = MicroscopioSimulado(real, [], semilla=5)
        est = calibrar_escala(sim, 150.0, 150.0, 320, 240)
        assert (est.signo_x, est.signo_y) == (sgx, sgy), (est, real)
        assert abs(est.um_por_px_fila / sf - 1) < 0.03 and abs(est.um_por_px_col / sc - 1) < 0.03, (est, real)
        print(f"ok  calibrar_escala  real=({sf}, {sc}, {sgx:+d}, {sgy:+d})  "
              f"medido=({est.um_por_px_fila:.3f}, {est.um_por_px_col:.3f}, {est.signo_x:+d}, {est.signo_y:+d})")


# ---------------------------------------------------------------- barrido
# Puntos violetas: posición (x, y) en µm respecto de (150, 150) mm. Con los pasos de la espiral (mitad del campo:
# 107 µm en x, 143 µm en y), los bordes de los campos caen en múltiplos de 107 µm (x) y de 143 µm (y).
#   - F (105, 90) queda pegado al borde x = +107 del campo centrado en x = 0: ejercita la regla de ignorar
#     objetos cortados por el borde (se lo fotografía desde el campo vecino, donde queda entero).
#   - G y H están a 6 µm entre sí: se superponen y forman UNA sola mancha (un solo objeto).
_OFF_UM = [(40, 60), (-60, 200), (160, -100), (-170, -220), (90, -330), (105, 90), (-200, 120), (-195, 123.4)]
PUNTOS = [(150.0 + x / 1000, 150.0 + y / 1000) for x, y in _OFF_UM]
GRUPOS = PUNTOS[:6] + [(150.0 + (-200 - 195) / 2 / 1000, 150.0 + (120 + 123.4) / 2 / 1000)]   # objetos a fotografiar


def correr_barrido(real: Calibracion, nominal: Calibracion, verificar: bool, max_corr: int = 2, semilla: int = 7):
    with tempfile.TemporaryDirectory() as t:
        sim = MicroscopioSimulado(real, PUNTOS, semilla=semilla, carpeta_fotos=str(Path(t) / "fotos"))
        cfg = ConfigBarrido(carpeta_salida=str(Path(t) / "salida"), verificar=verificar, max_correcciones=max_corr)
        campos = list(campos_espiral(150.0, 150.0, 0.107, 0.143, n_radial=2))
        res, reg = barrer(sim, DetectorColor(), nominal, campos, cfg, log=lambda *_: None)
        guardado = (Path(cfg.carpeta_salida) / "registro_visitados.json").exists()
        recargado = len(RegistroVisitados.cargar(Path(cfg.carpeta_salida) / "registro_visitados.json"))
    return sim, res, reg, guardado, recargado


def evaluar(sim: MicroscopioSimulado):
    """Asocia cada foto con el objeto más cercano. Devuelve (fotos_por_objeto, residuo_max_um, lejanas)."""
    cuentas = [0] * len(GRUPOS)
    residuos = [[] for _ in GRUPOS]
    lejanas = 0
    for f in sim.fotos:
        pos = (f["pasos"][0] * PASO_MM, f["pasos"][1] * PASO_MM)
        offs = [((g[0] - pos[0]) * 1000, (g[1] - pos[1]) * 1000) for g in GRUPOS]
        k = int(np.argmin([np.hypot(*o) for o in offs]))
        if np.hypot(*offs[k]) > 20:
            lejanas += 1
        cuentas[k] += 1
        residuos[k].append(offs[k])
    peor = max((max(abs(o[0]), abs(o[1])) for rs in residuos for o in rs), default=float("nan"))
    return cuentas, peor, lejanas


def test_barrido_completo():
    sim, res, reg, guardado, recargado = correr_barrido(NOMINAL, NOMINAL, verificar=True)
    cuentas, peor, lejanas = evaluar(sim)
    print(f"ok  barrido  campos={res.campos_visitados} detectados={res.objetos_detectados} fotos={res.fotos} "
          f"repetidos_omitidos={res.omitidos_por_repetidos} borde={res.ignorados_por_borde} "
          f"no_verificados={res.no_verificados} fotos_por_objeto={cuentas} residuo_max={peor:.2f} µm movimientos={sim.movimientos}")
    assert cuentas == [1] * len(GRUPOS), cuentas          # una foto por objeto, incluido el par superpuesto
    assert lejanas == 0
    assert peor <= MITAD_UM + 1.5, peor                   # dentro de la celda (+ error de detección)
    assert res.omitidos_por_repetidos > 0                 # el mecanismo de repetidos se ejercitó de verdad
    assert res.ignorados_por_borde > 0                    # y el de objetos cortados por el borde también
    assert res.no_verificados == 0
    assert guardado and recargado == len(GRUPOS)


def test_escala_mal_calibrada():
    real = Calibracion(320, 240, um_por_px_fila=NOMINAL.um_por_px_fila * 1.25,
                       um_por_px_col=NOMINAL.um_por_px_col * 1.25)     # la real es 25 % mayor que la supuesta
    simv, resv, _, _, _ = correr_barrido(real, NOMINAL, verificar=True)
    cv, pv, lv = evaluar(simv)
    simn, resn, _, _, _ = correr_barrido(real, NOMINAL, verificar=False)
    cn, pn, ln = evaluar(simn)
    print(f"ok  escala -25%  CON verificación: fotos_por_objeto={cv} residuo_max={pv:.2f} µm no_verif={resv.no_verificados} "
          f"movimientos={simv.movimientos}")
    print(f"                 SIN verificación: fotos_por_objeto={cn} residuo_max={pn:.2f} µm")
    # La verificación juzga "centrado" con la escala supuesta, así que hereda su error: queda hasta 25 % por encima de la celda.
    assert pv <= MITAD_UM * 1.25 + 1.5, pv
    assert pn > MITAD_UM * 1.25 + 1.5, "sin verificar y con la escala mal, debería haber objetos fuera de la celda"


if __name__ == "__main__":
    test_cuantizacion()
    test_detector()
    test_registro()
    test_calibrar_escala()
    test_barrido_completo()
    test_escala_mal_calibrada()
    print("\nTodas las pruebas pasaron.")
