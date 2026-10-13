#%%

"""
Plano de enfoque aproximado de la muestra: se ajusta un plano
z = a*x + b*y + c a un conjunto de puntos (x, y, z_enfoque), donde
z_enfoque sale de autofocus() en cada campo visual visitado.

Es una ESTIMACIÓN: por vibraciones el sistema óptico se puede
desenfocar solo, y la muestra no es un plano perfecto. Sirve para
saber dónde buscar el foco (arrancar el autofocus() cerca de
plano.z(x, y)), no para reemplazarlo: el z real siempre se confirma y
se ajusta con autofocus().

Unidades: todo en mm. Z creciente = alejarse de la muestra.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PlanoEnfoque:
    a: float                 # dz/dx  [mm/mm]
    b: float                 # dz/dy  [mm/mm]
    c: float                 # z en (0, 0) [mm]
    rms_mm: float            # error RMS de los puntos usados
    n_usados: int
    n_descartados: int
    mascara_usados: np.ndarray = field(repr=False, compare=False, default=None)

    def z(self, x, y):
        """z estimado del plano en (x, y) (acepta escalares o arrays)."""
        return self.a * np.asarray(x, dtype=float) + self.b * np.asarray(y, dtype=float) + self.c

    @property
    def pendientes_um_por_mm(self):
        """(dz/dx, dz/dy) en µm de z por mm desplazado."""
        return (self.a * 1000.0, self.b * 1000.0)

    @property
    def inclinacion_grados(self):
        """Ángulo entre el plano de enfoque y el plano horizontal."""
        return float(np.degrees(np.arctan(np.hypot(self.a, self.b))))


def _dispersion_perpendicular(xy):
    """Dispersión RMS (mm) de los puntos en la dirección en que MENOS
    se extienden: ~0 si todos están sobre una recta."""

    centrados = xy - xy.mean(axis=0)
    _, s, _ = np.linalg.svd(centrados, full_matrices=False)

    return float(s[1] / np.sqrt(len(xy)))


def ajustar_plano(
    puntos,
    descartar_atipicos=True,
    k_sigma=3.0,
    sigma_minima_mm=0.0025,
    max_iteraciones=5,
    dispersion_minima_mm=0.1,
):
    """
    Ajusta z = a*x + b*y + c por cuadrados mínimos a 'puntos'
    (iterable de (x, y, z), en mm).

    Con 'descartar_atipicos', itera: ajusta, calcula los residuos,
    descarta los puntos con |residuo| > k_sigma * sigma (sigma robusta
    por MAD, con piso 'sigma_minima_mm' = un paso de Z de 2.5 µm) y
    reajusta. Sirve contra autofocus() que se "enganchó" en cualquier
    cosa (basura, ruido) en algún campo visual. Nunca descarta de
    más: se frena si quedarían menos de 3 puntos o si los que quedan
    serían casi colineales.

    Levanta ValueError si hay menos de 3 puntos o si los puntos son
    casi colineales (dispersión perpendicular < 'dispersion_minima_mm'):
    con puntos sobre una sola recta, el plano no queda determinado
    (por ejemplo, si se barrió solo en y, falta recorrer algo en x).
    """

    pts = np.asarray(list(puntos), dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("'puntos' tiene que ser una lista de (x, y, z)")

    if len(pts) < 3:
        raise ValueError("Hacen falta al menos 3 puntos para ajustar un plano")

    if _dispersion_perpendicular(pts[:, :2]) < dispersion_minima_mm:
        raise ValueError(
            "Los puntos (x, y) son casi colineales: el plano no queda "
            "determinado. Hay que recorrer también la otra dirección."
        )

    usados = np.ones(len(pts), dtype=bool)

    def _ajustar(mascara):
        A = np.column_stack([pts[mascara, 0], pts[mascara, 1], np.ones(mascara.sum())])
        coef, *_ = np.linalg.lstsq(A, pts[mascara, 2], rcond=None)
        return coef

    coef = _ajustar(usados)

    if descartar_atipicos:

        for _ in range(max_iteraciones):

            residuos = pts[:, 2] - (coef[0] * pts[:, 0] + coef[1] * pts[:, 1] + coef[2])

            r_usados = residuos[usados]
            mad = np.median(np.abs(r_usados - np.median(r_usados)))
            sigma = max(1.4826 * mad, sigma_minima_mm)

            nueva = np.abs(residuos) <= k_sigma * sigma

            if nueva.sum() < 3 or np.array_equal(nueva, usados):
                break

            if _dispersion_perpendicular(pts[nueva, :2]) < dispersion_minima_mm:
                break

            usados = nueva
            coef = _ajustar(usados)

    residuos_finales = pts[usados, 2] - (
        coef[0] * pts[usados, 0] + coef[1] * pts[usados, 1] + coef[2]
    )

    return PlanoEnfoque(
        a=float(coef[0]),
        b=float(coef[1]),
        c=float(coef[2]),
        rms_mm=float(np.sqrt(np.mean(residuos_finales ** 2))),
        n_usados=int(usados.sum()),
        n_descartados=int((~usados).sum()),
        mascara_usados=usados,
    )


#%%
