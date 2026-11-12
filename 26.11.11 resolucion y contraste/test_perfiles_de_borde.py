# %%

"""
Pruebas de perfiles_de_borde.py / resolucion_contraste.py / circulos_imagej.py con glóbulos sintéticos
de sigma, radio y contraste conocidos (sin hardware, sin fotos).

Correr:  python test_perfiles_de_borde.py     (desde esta carpeta, o con pixi run python ...)
Que pase no reemplaza mirar las fotos reales: la imagen sintética no tiene halos, aberraciones, suciedad
ni JPEG.
"""

import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from circulos_imagej import leer_circulos_imagej
from perfiles_de_borde import (
    ConfigPerfiles,
    MOTIVO_VECINO,
    refinar_centro_y_radio,
)
from resolucion_contraste import FACTOR_FWHM, FACTOR_W1090, medir_globulo, mtf_gaussiana


SUPERMUESTREO = 4


def globulo_sintetico(alto=320, ancho=320, centro=(160.0, 160.0), radio=50.0, sigma=4.0,
                      fondo=200.0, celula=100.0, vecinos=(), ruido=1.5, semilla=0):
    """Disco de nivel `celula` sobre `fondo`, desenfocado con una gaussiana de desvío `sigma` px, más ruido.

    Se dibuja con supermuestreo y se promedia por bloques (el píxel integra). `vecinos` es una lista de
    (centro_fila, centro_col, radio) de otros discos.
    """
    s = SUPERMUESTREO
    filas, cols = (np.mgrid[0:alto * s, 0:ancho * s] + 0.5) / s - 0.5
    mascara = np.zeros(filas.shape, dtype=bool)
    for cf, cc, r in [(centro[0], centro[1], radio), *vecinos]:
        mascara |= (filas - cf) ** 2 + (cols - cc) ** 2 <= r ** 2
    borroso = gaussian_filter(mascara.astype(float), sigma * s)
    imagen = borroso.reshape(alto, s, ancho, s).mean(axis=(1, 3))
    imagen = fondo + (celula - fondo) * imagen
    imagen += np.random.default_rng(semilla).normal(0.0, ruido, imagen.shape)
    return np.clip(imagen, 0, 255)


def prueba_recupera_sigma():
    for sigma in (2.0, 4.0, 7.0):
        imagen = globulo_sintetico(sigma=sigma)
        res = medir_globulo(imagen, 160.0, 160.0, 50.0)
        error = abs(res.sigma_mediana_px - sigma) / sigma
        print(f"  sigma verdadero {sigma:.1f} -> mediana {res.sigma_mediana_px:.3f} (std {res.sigma_std_px:.3f}, "
              f"{res.n_validos}/{res.n_rayos} rayos), perfil medio {res.sigma_esf_media_px:.3f}")
        assert error < 0.06, f"sigma {sigma}: error relativo {error:.3f}"
        assert abs(res.sigma_esf_media_px - sigma) / sigma < 0.08
        assert res.n_validos == res.n_rayos, "sin vecinos no debería descartarse ningún rayo"


def prueba_polaridad_invertida():
    imagen = globulo_sintetico(sigma=4.0, fondo=60.0, celula=180.0)
    res = medir_globulo(imagen, 160.0, 160.0, 50.0)
    print(f"  célula clara sobre fondo oscuro -> polaridad {res.circulo.polaridad}, sigma {res.sigma_mediana_px:.3f}")
    assert res.circulo.polaridad == -1
    assert abs(res.sigma_mediana_px - 4.0) / 4.0 < 0.06


def prueba_vecino_pegado():
    # un vecino tocando al glóbulo por la derecha (centros a 2 radios): los rayos hacia ese lado deben descartarse
    imagen = globulo_sintetico(sigma=4.0, vecinos=[(160.0, 262.0, 50.0)])
    res = medir_globulo(imagen, 160.0, 160.0, 50.0)
    print(f"  vecino pegado -> {res.n_validos}/{res.n_rayos} rayos válidos, sigma {res.sigma_mediana_px:.3f}, motivos {res.motivos_descarte}")
    assert 0.3 * res.n_rayos < res.n_validos < 0.9 * res.n_rayos
    assert MOTIVO_VECINO in res.motivos_descarte
    assert abs(res.sigma_mediana_px - 4.0) / 4.0 < 0.08
    angulos_validos = [r.angulo_rad for r in res.rayos if r.valido]
    # ningún rayo válido apunta hacia el vecino (angulo 0 = hacia +columna)
    assert all(abs(np.angle(np.exp(1j * a))) > 0.4 for a in angulos_validos)


def prueba_centro_y_radio():
    imagen = globulo_sintetico(sigma=4.0, centro=(150.0, 170.0), radio=48.0)
    circulo = refinar_centro_y_radio(imagen, 156.0, 164.0, 53.0)    # círculo aproximado corrido 6 px y 10 % más grande
    print(f"  centro refinado ({circulo.centro_fila_px:.2f}, {circulo.centro_col_px:.2f}) radio {circulo.radio_px:.2f} "
          f"(verdadero (150, 170) y 48), rms {circulo.rms_px:.2f}")
    assert np.hypot(circulo.centro_fila_px - 150.0, circulo.centro_col_px - 170.0) < 0.7
    assert abs(circulo.radio_px - 48.0) < 1.0


def prueba_contraste():
    imagen = globulo_sintetico(sigma=4.0, fondo=200.0, celula=100.0)
    res = medir_globulo(imagen, 160.0, 160.0, 50.0)
    michelson, weber = 100.0 / 300.0, 100.0 / 200.0       # (F-C)/(F+C) y (F-C)/F
    print(f"  Michelson {res.michelson_mediana:.3f} (esperado {michelson:.3f}), Weber borde {res.weber_borde_mediana:.3f} / "
          f"interior {res.weber_interior:.3f} (esperado {weber:.3f})")
    assert abs(res.michelson_mediana - michelson) < 0.01
    assert abs(res.weber_borde_mediana - weber) < 0.015
    assert abs(res.weber_interior - weber) < 0.015


def prueba_anchos_y_mtf():
    sigma = 4.0
    imagen = globulo_sintetico(sigma=sigma)
    res = medir_globulo(imagen, 160.0, 160.0, 50.0, um_por_px=0.05)
    mtf50_teorica = 0.1874 / sigma
    print(f"  FWHM {res.fwhm_px:.2f} px (= {FACTOR_FWHM:.4f} sigma), w10-90 {res.w1090_px:.2f} px (= {FACTOR_W1090:.4f} sigma), "
          f"MTF50 {res.mtf50_ciclos_px:.4f} (no param. {res.mtf50_no_param_ciclos_px:.4f}) ciclos/px")
    assert abs(res.mtf50_ciclos_px - mtf50_teorica) / mtf50_teorica < 0.06
    assert abs(res.mtf50_no_param_ciclos_px - mtf50_teorica) / mtf50_teorica < 0.05
    assert abs(mtf_gaussiana(res.mtf50_ciclos_px, res.sigma_mediana_px) - 0.5) < 1e-6
    assert abs(res.fwhm_um - res.fwhm_px * 0.05) < 1e-12
    assert abs(res.mtf50_ciclos_um - res.mtf50_ciclos_px / 0.05) < 1e-9


def prueba_pocos_rayos():
    # sin glóbulo no hay nada que medir: no debe devolver métricas inventadas (puede fallar o devolver sin métricas)
    imagen = np.full((200, 200), 150.0) + np.random.default_rng(1).normal(0.0, 1.5, (200, 200))
    try:
        res = medir_globulo(imagen, 100.0, 100.0, 40.0)
    except ValueError as error:
        print(f"  imagen sin glóbulo -> ValueError: {str(error)[:80]}...")
        return
    print(f"  imagen sin glóbulo -> {res.n_validos}/{res.n_rayos} rayos válidos, advertencias: {res.advertencias}")
    assert res.n_validos < 8 and res.advertencias and np.isnan(res.sigma_mediana_px)


def prueba_lector_csv():
    texto = (" ,Area,Mean,Min,Max,X,Y,Feret,FeretX,FeretY,FeretAngle,MinFeret\n"
             "1,6582,143.176,129,162,1225.500,1296.000,93,1179.000,1296,0,90\n"
             "2,10008,164.602,143,182,2099.000,166.000,116,2041.000,166,0,110\n")
    with tempfile.TemporaryDirectory() as carpeta:
        ruta = Path(carpeta) / "circulos.csv"
        ruta.write_text(texto, encoding="utf-8")
        circulos = leer_circulos_imagej(ruta)
        en_pulgadas = leer_circulos_imagej(ruta, px_por_unidad=96.0)
    print(f"  CSV: {len(circulos)} círculos; el primero en numpy (fila, col) = ({circulos[0].centro_fila_px}, {circulos[0].centro_col_px})")
    assert len(circulos) == 2 and circulos[0].indice == 1
    assert (circulos[0].centro_fila_px, circulos[0].centro_col_px, circulos[0].diametro_px) == (1295.5, 1225.0, 93.0)
    assert abs(en_pulgadas[0].diametro_px - 93.0 * 96.0) < 1e-9


PRUEBAS = [prueba_recupera_sigma, prueba_polaridad_invertida, prueba_vecino_pegado, prueba_centro_y_radio,
           prueba_contraste, prueba_anchos_y_mtf, prueba_pocos_rayos, prueba_lector_csv]


if __name__ == "__main__":
    fallaron = []
    for prueba in PRUEBAS:
        print(prueba.__name__)
        try:
            prueba()
            print("  OK")
        except Exception as error:      # se sigue con las demás para ver todas las fallas juntas
            print(f"  FALLÓ: {type(error).__name__}: {error}")
            fallaron.append(prueba.__name__)
    print()
    print("TODO OK" if not fallaron else f"FALLARON: {fallaron}")


# %%
