"""
Pruebas del detector por contraste local y del calibrador:  python test_deteccion_contraste.py   (o: pytest -q)

Se usan imágenes sintéticas con los colores (en Lab) medidos en las fotos reales de calibración:
plasma verdoso, glóbulos rosados-malva, punto azul oscuro y polvo negro neutro.

  1. un punto: un solo objeto, con el centro a < 1 px del verdadero; los glóbulos no se detectan
  2. cambios de iluminación (brillo 0.7x-1.4x y dominantes cálida/fría): sigue detectando exactamente 1
  3. dos puntos superpuestos: un solo objeto
  4. un punto cortado por el borde se marca toca_borde
  5. polvo fijo: sin máscara se lo detecta; estimar_mascara_fija lo encuentra y el detector lo ignora
     sin tapar los puntos reales
  6. máscara con tamaño equivocado: error claro
  7. calibrar_detector.py de punta a punta con carpetas de PNG: imprime la estadística y elige parámetros
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from deteccion_contraste import (DetectorContraste, ParametrosContraste, cargar_mascara_fija,
                                 estimar_mascara_fija, guardar_mascara_fija)

ALTO, ANCHO = 240, 320
LAB_PLASMA, LAB_GLOBULO, LAB_PUNTO, LAB_POLVO = (175, 125, 128), (140, 138, 122), (100, 139, 108), (40, 128, 128)
POLVO = [(270, 74), (222, 212)]      # (col, fila): fijas en todas las fotos, como en las reales


def _bgr(lab) -> tuple[int, int, int]:
    px = cv2.cvtColor(np.array([[lab]], np.uint8), cv2.COLOR_LAB2BGR)[0, 0]
    return int(px[0]), int(px[1]), int(px[2])


def campo(semilla: int, puntos=(), polvo=True, radio_punto: float = 7.0) -> np.ndarray:
    """Campo sintético. `puntos`: lista de (col, fila) con subpíxel; se dibujan con antialiasing (shift=4)."""
    rng = np.random.default_rng(semilla)
    img = np.full((ALTO, ANCHO, 3), _bgr(LAB_PLASMA), np.uint8)
    for _ in range(170):
        c, f = int(rng.integers(0, ANCHO)), int(rng.integers(0, ALTO))
        lab = tuple(int(np.clip(v + rng.normal(0, 3), 0, 255)) for v in LAB_GLOBULO)
        cv2.circle(img, (c, f), int(rng.integers(6, 9)), _bgr(lab), -1, cv2.LINE_AA)
    for (c, f) in puntos:
        cv2.circle(img, (int(round(c * 16)), int(round(f * 16))), int(round(radio_punto * 16)), _bgr(LAB_PUNTO), -1,
                   cv2.LINE_AA, shift=4)
    if polvo:
        for (c, f) in POLVO:
            cv2.circle(img, (c, f), 3, _bgr(LAB_POLVO), -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    ruido = rng.normal(0, 2.0, img.shape)
    return np.clip(img.astype(np.float32) + ruido, 0, 255).astype(np.uint8)


def _cambiar(img: np.ndarray, gb=1.0, gg=1.0, gr=1.0, brillo=1.0) -> np.ndarray:
    return np.clip(img.astype(np.float32) * np.array([gb, gg, gr], np.float32) * brillo, 0, 255).astype(np.uint8)


def _mascara_de(semilla0: int = 100, n: int = 10) -> np.ndarray:
    """Máscara de manchas fijas estimada con n campos distintos (con puntos en lugares distintos)."""
    rng = np.random.default_rng(semilla0)
    imgs = [campo(semilla0 + k, puntos=[(float(rng.uniform(20, 300)), float(rng.uniform(20, 220)))]) for k in range(n)]
    return estimar_mascara_fija(imgs)


# ---------------------------------------------------------------- 1
def test_un_punto_y_centro():
    det = DetectorContraste(ParametrosContraste(), mascara_fija=_mascara_de())
    errores = []
    for k, (c, f) in enumerate([(100.3, 80.7), (200.5, 150.2), (60.0, 200.0), (250.8, 40.4)]):
        ds = det.detectar(campo(10 + k, puntos=[(c, f)]))
        assert len(ds) == 1, (c, f, ds)
        errores.append(float(np.hypot(ds[0].col - c, ds[0].fila - f)))
        assert not ds[0].toca_borde
    assert max(errores) < 1.0, errores
    assert det.detectar(campo(99, puntos=[])) == []              # solo glóbulos y polvo: nada
    print(f"ok  un punto: error de centro máx {max(errores):.2f} px; campo sin puntos -> 0 detecciones")


# ---------------------------------------------------------------- 2
def test_robusto_a_iluminacion():
    det = DetectorContraste(ParametrosContraste(), mascara_fija=_mascara_de())
    casos = {"brillo 0.7": dict(brillo=0.7), "brillo 1.4": dict(brillo=1.4), "cálido": dict(gb=0.9, gr=1.1),
             "frío": dict(gb=1.1, gr=0.9), "oscuro y cálido": dict(gb=0.9, gr=1.1, brillo=0.7),
             "claro y frío": dict(gb=1.1, gr=0.9, brillo=1.3)}
    for nombre, kw in casos.items():
        con = _cambiar(campo(31, puntos=[(150.0, 120.0)]), **kw)
        sin = _cambiar(campo(32, puntos=[]), **kw)
        assert len(det.detectar(con)) == 1, nombre
        assert det.detectar(sin) == [], nombre
    print(f"ok  robusto a iluminación: {len(casos)} variantes de brillo y color, siempre 1 punto y 0 falsas")


# ---------------------------------------------------------------- 3
def test_superpuestos_son_uno():
    det = DetectorContraste(ParametrosContraste(), mascara_fija=_mascara_de())
    ds = det.detectar(campo(41, puntos=[(150.0, 120.0), (158.0, 124.0)]))
    assert len(ds) == 1, ds
    assert 150.0 < ds[0].col < 158.0 and 120.0 < ds[0].fila < 124.0
    print("ok  dos puntos superpuestos -> un solo objeto, centrado entre ambos")


# ---------------------------------------------------------------- 4
def test_borde():
    det = DetectorContraste(ParametrosContraste(), mascara_fija=_mascara_de())
    ds = det.detectar(campo(51, puntos=[(3.0, 120.0)]))
    assert len(ds) == 1 and ds[0].toca_borde, ds
    print("ok  punto cortado por el borde -> toca_borde")


# ---------------------------------------------------------------- 5
def test_polvo_fijo():
    sin_mascara = DetectorContraste(ParametrosContraste())
    ds = sin_mascara.detectar(campo(61, puntos=[(150.0, 120.0)]))
    assert len(ds) == 3, [(round(d.col), round(d.fila)) for d in ds]    # el punto + las 2 motas de polvo
    mascara = _mascara_de()
    for (c, f) in POLVO:
        assert mascara[f, c], (c, f)
    assert not mascara[100:140, 130:170].any()                          # no tapa la zona de los puntos reales
    con = DetectorContraste(ParametrosContraste(), mascara_fija=mascara).detectar(campo(61, puntos=[(150.0, 120.0)]))
    assert len(con) == 1 and abs(con[0].col - 150.0) < 1 and abs(con[0].fila - 120.0) < 1, con
    # punto real justo al lado de una mota de polvo: se sigue detectando
    cerca = DetectorContraste(ParametrosContraste(), mascara_fija=mascara).detectar(
        campo(62, puntos=[(POLVO[0][0] - 20.0, POLVO[0][1] + 10.0)]))
    assert len(cerca) == 1, cerca
    # con menos de 6 fotos no se estima nada (no hay con qué distinguir polvo de muestra)
    assert not estimar_mascara_fija([campo(70 + k) for k in range(3)]).any()
    with tempfile.TemporaryDirectory() as t:                            # ida y vuelta a disco
        guardar_mascara_fija(mascara, Path(t) / "m.png")
        assert np.array_equal(cargar_mascara_fija(Path(t) / "m.png"), mascara)
    print("ok  polvo fijo: sin máscara 3 detecciones; con máscara estimada, 1 (el punto real); ida y vuelta a PNG")


# ---------------------------------------------------------------- 6
def test_mascara_tamano_equivocado():
    det = DetectorContraste(ParametrosContraste(), mascara_fija=np.zeros((100, 100), bool))
    try:
        det.detectar(campo(81))
    except ValueError as e:
        assert "no coincide" in str(e)
    else:
        raise AssertionError("debería rechazar una máscara de otro tamaño")
    print("ok  máscara de otro tamaño -> ValueError")


# ---------------------------------------------------------------- 7
def test_calibrador_de_punta_a_punta():
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "con").mkdir(), (t / "sin").mkdir()
        for k in range(10):
            pts = [(float(rng.uniform(25, 295)), float(rng.uniform(25, 215)))]
            cv2.imwrite(str(t / "con" / f"c{k:02d}.png"), campo(200 + k, puntos=pts))
        for k in range(12):
            cv2.imwrite(str(t / "sin" / f"s{k:02d}.png"), campo(300 + k, puntos=[]))
        r = subprocess.run([sys.executable, str(Path(__file__).with_name("calibrar_detector.py")), str(t / "con"),
                            "--sin", str(t / "sin"), "--total", "10", "--salida", str(t / "debug")],
                           capture_output=True, text=True, timeout=250)
        assert r.returncode == 0, r.stdout + r.stderr
        out = r.stdout
        assert "En las imágenes SIN puntos violetas se encontraron 0 detecciones (en 0 de 12 fotos)" in out, out
        assert "En las imágenes CON puntos violetas se encontraron 10 detecciones" in out, out
        assert "indicaste que hay 10 puntos" in out and "(100 %)" in out, out
        assert "Manchas fijas" in out and "se ignoran" in out, out
        assert (t / "debug" / "mascara_fija.png").exists()
        assert len(list((t / "debug").glob("con_*.png"))) == 10 and len(list((t / "debug").glob("sin_*.png"))) == 12
    # el mensaje con un total mayor que las detecciones refleja el recall estimado y avisa
    import calibrar_detector as cd
    m = cd.mensaje(cd.estadistica([1] * 10, [0] * 12, total=20))
    assert "(50 %)" in m and "Ni con los mejores parámetros" in m, m
    m = cd.mensaje(cd.estadistica([1, 2, 1, 1], [1, 0, 0, 0], total=4))      # 5 detecciones, 1 falsa estimada por foto sin puntos
    assert "se encontraron 1 detecciones (en 1 de 4 fotos)" in m and "se encontraron 5 detecciones" in m, m
    m = cd.mensaje(cd.estadistica([1] * 10, [0] * 12, total=None))
    assert "indicaste" not in m and "Estimación" not in m, m
    print("ok  calibrar_detector.py: estadística n_sin / n_con, estimación con --total, máscara y fotos marcadas")


if __name__ == "__main__":
    test_un_punto_y_centro()
    test_robusto_a_iluminacion()
    test_superpuestos_son_uno()
    test_borde()
    test_polvo_fijo()
    test_mascara_tamano_equivocado()
    test_calibrador_de_punta_a_punta()
    print("\nTodas las pruebas pasaron.")
