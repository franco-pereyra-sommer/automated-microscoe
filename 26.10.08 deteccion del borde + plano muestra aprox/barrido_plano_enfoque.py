#%%

"""
Barrido en "T" para reunir puntos (x, y, z_enfoque) y ajustar el plano
de enfoque aproximado de la muestra (ver plano_enfoque.py).

Recorrido, a partir de un campo visual INTERIOR de la muestra (x0, y0):

  1) "+y": de a paso_y_mm (150 µm) hacia +y, hasta llegar a un borde.
  2) Vuelve a (x0, y0) y hace lo mismo hacia "-y".
  3) "medio": va al punto medio en y entre los dos bordes (x0, y_medio).
  4) "+x": desde ahí, de a paso_x_mm (100 µm) hacia +x, hasta un borde.

En cada campo visual:

  - autofocus() (con la cámara en camara.config_autofoco_local()) deja
    la platina en z_enfoque;
  - se saca UNA foto más ahí y con esa foto se calcula
    distancia_normalizada (analisis_campo_visual.caracteristicas_borde);
  - si distancia_normalizada >= umbral_distancia_borde, ese campo visual
    es un borde y la caminata en esa dirección termina.

El resultado (ResultadoExploracion) ya viene listo para plano_enfoque:

    resultado = explorar_plano_enfoque(...)
    plano = resultado.ajustar_plano()        # PlanoEnfoque
    plano.z(x, y)                            # z estimado en (x, y)

Seguridad (regla validada en el laboratorio, ver moverse_a_xy() en
busqueda_muestra.py): parado en un z_enfoque se puede mover hasta ~150
µm en XY sin tocar el Z. Por eso paso_x_mm y paso_y_mm no pueden pasar
de 0.15 mm. Los dos viajes largos (de vuelta al inicio y al punto medio)
suben antes a z_arriba_seguro y recién después vuelven a bajar.

No modifica autofocus() ni ningún otro módulo: todo lo que hace lo
saca de busqueda_muestra.py, analisis_campo_visual.py y plano_enfoque.py.

Unidades: mm. Z creciente = alejarse de la muestra.
"""

# Agrego la raiz del proyecto
import sys
import os

os.environ["PYTHONPATH"] = r"C:\BackUp\Facultad\Labo 7\env_labo7"

sys.path.insert(0, os.environ["PYTHONPATH"])

import csv
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from busqueda_muestra import (
    DISTANCIA_XY_SIN_SUBIR_MM,
    FUNCIONES_METRICA,
    CampoVisualSinMuestra,
    PicoNoConfirmado,
    autofocus,
    bytes_a_imagen,
    intensidad_fondo,
    moverse_a_xy,
    posicionar_en_muestra,
    laplacian_variance
)
from analisis_campo_visual import caracteristicas_borde
from plano_enfoque import ajustar_plano as _ajustar_plano


# ============================================================
# EXCEPCIONES
# ============================================================

class CampoInicialEnBorde(Exception):
    """El campo visual donde arranca la exploración ya es un borde
    (distancia_normalizada >= umbral_distancia_borde): hay que arrancar
    desde un campo visual más adentro de la muestra."""
    pass


# ============================================================
# GRILLA DE LA IMPRESORA
# ============================================================
# La impresora se mueve en múltiplos de 0.0125 mm en x-y y de 0.0025 mm
# en z. Las coordenadas de destino se pasan por la grilla para que la
# posición que registra la impresora coincida con la pedida.

PASO_MINIMO_XY_MM = 0.0125
PASO_MINIMO_Z_MM = 0.0025


def _a_grilla(valor_mm, paso_minimo=PASO_MINIMO_XY_MM):
    """Redondea 'valor_mm' al múltiplo más cercano de 'paso_minimo'."""
    return float(np.round(np.round(valor_mm / paso_minimo) * paso_minimo, 4))


def _z_a_grilla_hacia_arriba(z_mm):
    """Redondea z HACIA ARRIBA (alejándose de la muestra) al múltiplo de
    2.5 µm: así el Z real nunca queda por debajo del pedido."""
    return float(np.round(np.ceil(z_mm / PASO_MINIMO_Z_MM - 1e-6) * PASO_MINIMO_Z_MM, 4))


# ============================================================
# RESULTADO
# ============================================================

CAMPOS_CSV = [
    "indice", "fase", "x", "y", "z_enfoque",
    "distancia_normalizada", "es_borde", "confiable",
    "valor_metrica", "intensidad_fondo",
    "foto_maxima", "carpeta_autofocus",
]


@dataclass
class PuntoEnfoque:
    indice: int                 # número de intento (los saltos marcan autofocus fallidos)
    fase: str                   # "inicio", "+y", "-y", "medio", "+x" o "-x"
    x: float                    # [mm] posición real de la platina
    y: float
    z: float                    # [mm] z_enfoque (máximo del autofocus)
    distancia_normalizada: float
    es_borde: bool              # distancia_normalizada >= umbral_distancia_borde
    confiable: bool             # False si se detectó que no hay muestra (ver umbral_intensidad_fondo)
    valor_metrica: float        # valor de la métrica en el máximo
    intensidad_fondo: float
    ruta_foto_maxima: Optional[str] = None
    carpeta_autofocus: Optional[str] = None


@dataclass
class ResultadoExploracion:
    umbral_distancia_borde: float
    puntos: list = field(default_factory=list)
    # Por qué terminó cada caminata: "borde" (lo normal), "max_pasos",
    # "limite_impresora", "pico_no_confirmado" o "sin_muestra".
    motivo_fin: dict = field(default_factory=dict)
    avisos: list = field(default_factory=list)
    ruta_csv: Optional[Path] = None

    def puntos_xyz(self, incluir_bordes=False):
        """
        Array (N, 3) de [x, y, z_enfoque] en mm, listo para
        plano_enfoque.ajustar_plano().

        Por defecto NO incluye los campos visuales de borde: en un
        campo visual medio vacío la métrica de enfoque es menos
        confiable, y perder esos puntos casi no cambia el ajuste.
        Tampoco incluye los puntos marcados como no confiables.
        """

        elegidos = [
            p for p in self.puntos
            if p.confiable and (incluir_bordes or not p.es_borde)
        ]

        return np.array([[p.x, p.y, p.z] for p in elegidos], dtype=float).reshape(-1, 3)

    def ajustar_plano(self, incluir_bordes=False, **kwargs_ajuste):
        """Atajo para plano_enfoque.ajustar_plano(self.puntos_xyz(...)).
        Los kwargs se pasan tal cual (k_sigma, descartar_atipicos, ...)."""
        return _ajustar_plano(self.puntos_xyz(incluir_bordes), **kwargs_ajuste)

    def resumen(self):
        """Imprime cuántos puntos hay por fase, cómo terminó cada
        caminata y, si alcanza, el plano ajustado."""

        print("\n" + "=" * 60)
        print("RESUMEN DE LA EXPLORACIÓN")
        print("=" * 60)

        for fase in ("inicio", "+y", "-y", "medio", "+x", "-x"):
            pts = [p for p in self.puntos if p.fase == fase]
            if not pts:
                continue
            borde = sum(p.es_borde for p in pts)
            print(
                f"{fase:>6s}: {len(pts):3d} campos visuales"
                f" ({borde} de borde) | "
                f"z de {min(p.z for p in pts):.4f} a {max(p.z for p in pts):.4f} mm"
            )

        for etiqueta, motivo in self.motivo_fin.items():
            print(f"  {etiqueta}: terminó por '{motivo}'")

        for aviso in self.avisos:
            print(f"  [AVISO] {aviso}")

        try:
            plano = self.ajustar_plano()
            print(
                f"\nPlano (sin bordes, {plano.n_usados} puntos usados, "
                f"{plano.n_descartados} descartados): "
                f"dz/dx = {plano.pendientes_um_por_mm[0]:+.2f} µm/mm | "
                f"dz/dy = {plano.pendientes_um_por_mm[1]:+.2f} µm/mm | "
                f"inclinación = {plano.inclinacion_grados:.3f}° | "
                f"rms = {plano.rms_mm * 1000:.1f} µm"
            )
        except ValueError as e:
            print(f"\nTodavía no se puede ajustar el plano: {e}")


# ============================================================
# ORQUESTADOR INTERNO
# ============================================================

class _Exploracion:

    def __init__(
        self,
        camara,
        impresora,
        metrica,
        umbral_distancia_borde,
        z_arriba_seguro,
        z_min_seguridad,
        carpeta_origen,
        guardar_fotos_autofoco,
        guardar_foto_maxima,
        paso_x,
        paso_y,
        sentido_x,
        max_pasos,
        paso_z,
        pasos_foco_local,
        pasos_foco_medio,
        espera_movimiento,
        feedrate_xy,
        umbral_intensidad_fondo,
        margen_descenso_mm,
        tolerancia_ajuste_linea_mm,
        graficar_autofoco,
        kwargs_autofocus,
        kwargs_config_camara,
        kwargs_posicionar,
    ):
        self.camara = camara
        self.impresora = impresora
        self.metrica = metrica
        self.umbral = umbral_distancia_borde
        self.z_arriba_seguro = z_arriba_seguro
        self.z_min_seguridad = z_min_seguridad
        self.carpeta_origen = carpeta_origen
        self.guardar_fotos_autofoco = guardar_fotos_autofoco
        self.guardar_foto_maxima = guardar_foto_maxima
        self.paso_x = paso_x
        self.paso_y = paso_y
        self.sentido_x = sentido_x
        self.max_pasos = max_pasos
        self.paso_z = paso_z
        self.pasos_foco_local = pasos_foco_local
        self.pasos_foco_medio = pasos_foco_medio
        self.espera_movimiento = espera_movimiento
        self.feedrate_xy = feedrate_xy
        self.umbral_intensidad_fondo = umbral_intensidad_fondo
        self.margen_descenso_mm = margen_descenso_mm
        self.tolerancia_ajuste_linea_mm = tolerancia_ajuste_linea_mm
        self.graficar_autofoco = graficar_autofoco
        self.kwargs_autofocus = kwargs_autofocus
        self.kwargs_config_camara = kwargs_config_camara
        self.kwargs_posicionar = kwargs_posicionar

        self.resultado = ResultadoExploracion(umbral_distancia_borde=umbral_distancia_borde)
        self.n_intentos = 0

        self.x0 = self.y0 = self.z0 = None

        if carpeta_origen is not None:
            carpeta_origen.mkdir(parents=True, exist_ok=True)
            marca = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.resultado.ruta_csv = carpeta_origen / f"puntos_enfoque {marca}.csv"

    # --------------------------------------------------------
    # Cámara
    # --------------------------------------------------------

    def _config_camara(self):
        self.camara.config_autofoco_local(**self.kwargs_config_camara)

    # --------------------------------------------------------
    # Un campo visual: autofocus -> foto en z_enfoque -> clasificar
    # --------------------------------------------------------

    def _medir(self, fase, ancho=False):
        """
        Corre autofocus() donde está la platina, saca una foto en el
        z_enfoque que deja, la clasifica y la guarda. 'ancho' usa los
        barridos de Z más grandes (para después de un viaje largo).
        Levanta PicoNoConfirmado si autofocus() no pudo confirmar el máximo.
        """

        imp = self.impresora

        self.n_intentos += 1
        indice = self.n_intentos

        x = float(imp.posicion_x)
        y = float(imp.posicion_y)

        carpeta_af = None

        if self.guardar_fotos_autofoco:
            carpeta_af = self.carpeta_origen / f"{indice:04d} {fase} ({x}, {y})"

        pasos_arriba, pasos_abajo = self.pasos_foco_medio if ancho else self.pasos_foco_local

        kwargs = dict(
            pasos_arriba=pasos_arriba,
            pasos_abajo=pasos_abajo,
            paso_z=self.paso_z,
            espera_movimiento=self.espera_movimiento,
            graficamos=self.graficar_autofoco,
            guardamos=self.guardar_fotos_autofoco,
            carpeta_guardado=carpeta_af,
            z_min_seguridad=self.z_min_seguridad,
            z_max_seguro=self.z_arriba_seguro,
        )
        kwargs.update(self.kwargs_autofocus)

        mejor_z, _, valores = autofocus(
            camara=self.camara,
            impresora=imp,
            metrica=self.metrica,
            **kwargs,
        )

        # autofocus() ya dejó la platina en mejor_z: una foto más ahí.
        image_data = self.camara.capturar()
        imagen = bytes_a_imagen(image_data)

        c = caracteristicas_borde(imagen)
        fondo = float(intensidad_fondo(imagen))

        z = float(mejor_z)

        ruta_foto = None

        if self.guardar_foto_maxima:

            hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            ruta_foto = self.camara.guardar_imagen(
                image_data=image_data,
                carpeta=self.carpeta_origen,
                nombre=f"{indice:04d} {fase} ({x}, {y}, {z}) - {hora}",
                mostrar=False,
            )

        confiable = True

        if self.umbral_intensidad_fondo is not None and fondo <= self.umbral_intensidad_fondo:
            confiable = False

        punto = PuntoEnfoque(
            indice=indice,
            fase=fase,
            x=x,
            y=y,
            z=z,
            distancia_normalizada=float(c["distancia_normalizada"]),
            es_borde=bool(c["distancia_normalizada"] >= self.umbral),
            confiable=confiable,
            valor_metrica=float(np.max(valores)),
            intensidad_fondo=fondo,
            ruta_foto_maxima=None if ruta_foto is None else str(ruta_foto),
            carpeta_autofocus=None if carpeta_af is None else str(carpeta_af),
        )

        self._registrar(punto)

        estado = "BORDE" if punto.es_borde else "interior"
        if not confiable:
            estado = "SIN MUESTRA"

        print(
            f"\n>>> [{fase}] #{indice} ({x}, {y}, {z:.4f}) | "
            f"distancia_normalizada = {punto.distancia_normalizada:.4f} "
            f"(umbral {self.umbral}) -> {estado}\n"
        )

        return punto

    def _registrar(self, punto):

        self.resultado.puntos.append(punto)

        ruta = self.resultado.ruta_csv

        if ruta is None:
            return

        nuevo = not ruta.exists()

        with open(ruta, "a", newline="", encoding="utf-8") as f:

            escritor = csv.DictWriter(f, fieldnames=CAMPOS_CSV)

            if nuevo:
                escritor.writeheader()

            fila = {campo: getattr(punto, campo, "") for campo in CAMPOS_CSV}
            fila["z_enfoque"] = punto.z
            fila["foto_maxima"] = punto.ruta_foto_maxima or ""
            fila["carpeta_autofocus"] = punto.carpeta_autofocus or ""

            escritor.writerow(fila)

    # --------------------------------------------------------
    # Movimientos
    # --------------------------------------------------------

    def _llego(self, x, y):
        tolerancia = PASO_MINIMO_XY_MM / 2 + 1e-9
        return (
            abs(float(self.impresora.posicion_x) - x) <= tolerancia
            and abs(float(self.impresora.posicion_y) - y) <= tolerancia
        )

    def _ir_a(self, x, y, z_destino):
        """
        Viaje a (x, y) con moverse_a_xy(). Si fue un viaje largo (la
        platina quedó en z_arriba_seguro, sin enfocar), baja hasta
        'z_destino' (acotado a [z_min_seguridad, z_arriba_seguro] y
        redondeado hacia arriba a la grilla de 2.5 µm). Si fue corto, el
        Z no se toca. Devuelve True si fue un viaje largo.
        """

        imp = self.impresora

        largo = moverse_a_xy(
            imp, x, y, self.z_arriba_seguro,
            feedrate_xy=self.feedrate_xy,
            espera_movimiento=self.espera_movimiento,
        )

        if not self._llego(x, y):
            raise RuntimeError(
                f"La impresora no llegó a ({x}, {y}): quedó en "
                f"({imp.posicion_x}, {imp.posicion_y}). ¿Fuera de rango?"
            )

        if largo:

            _, _, z_actual = imp.pregunta_posicion_actual()

            z_dest = float(np.clip(z_destino, self.z_min_seguridad, self.z_arriba_seguro))
            z_dest = _z_a_grilla_hacia_arriba(z_dest)

            # Solo baja (nunca sube más de lo que ya subió moverse_a_xy)
            if z_dest < z_actual - 1e-9:
                imp.desp_lin_rel(0, 0, z_dest - z_actual)
                time.sleep(self.espera_movimiento)

        return largo

    # --------------------------------------------------------
    # Caminata en una dirección hasta encontrar un borde
    # --------------------------------------------------------

    def _caminar(self, eje, sentido, paso):

        imp = self.impresora

        etiqueta = f"{'+' if sentido > 0 else '-'}{eje}"
        motivo = "max_pasos"

        for _ in range(self.max_pasos):

            x_obj = float(imp.posicion_x)
            y_obj = float(imp.posicion_y)

            if eje == "x":
                x_obj = _a_grilla(x_obj + sentido * paso)
            else:
                y_obj = _a_grilla(y_obj + sentido * paso)

            largo = moverse_a_xy(
                imp, x_obj, y_obj, self.z_arriba_seguro,
                feedrate_xy=self.feedrate_xy,
                espera_movimiento=self.espera_movimiento,
            )

            if largo:
                # No debería pasar (los pasos se validan <= 0.15 mm). Si
                # pasara, la platina quedó en z_arriba_seguro sin enfocar y
                # un autofocus local sería engañoso: se corta.
                raise RuntimeError(
                    f"[{etiqueta}] moverse_a_xy() lo tomó como viaje largo "
                    f"(la platina quedó en Z = {imp.posicion_z}): hay que "
                    f"reenfocar a mano antes de seguir."
                )

            if not self._llego(x_obj, y_obj):
                aviso = (
                    f"[{etiqueta}] la impresora no llegó a ({x_obj}, {y_obj}) "
                    f"(quedó en {imp.posicion_x}, {imp.posicion_y}): probable límite de rango."
                )
                print(f"[AVISO] {aviso}")
                self.resultado.avisos.append(aviso)
                motivo = "limite_impresora"
                break

            try:
                punto = self._medir(etiqueta)
            except PicoNoConfirmado as e:
                aviso = (
                    f"[{etiqueta}] autofocus() no confirmó el máximo en "
                    f"({imp.posicion_x}, {imp.posicion_y}): {e}"
                )
                print(f"[AVISO] {aviso}")
                self.resultado.avisos.append(aviso)
                motivo = "pico_no_confirmado"
                break

            if not punto.confiable:
                motivo = "sin_muestra"
                break

            if punto.es_borde:
                motivo = "borde"
                break

        else:
            aviso = (
                f"[{etiqueta}] se agotaron los {self.max_pasos} pasos sin "
                f"llegar a un borde."
            )
            print(f"[AVISO] {aviso}")
            self.resultado.avisos.append(aviso)

        self.resultado.motivo_fin[etiqueta] = motivo

    # --------------------------------------------------------
    # Estimación de z sobre la recta ya recorrida (para el punto medio)
    # --------------------------------------------------------

    def _puntos_de_la_recta_y(self):
        return [
            p for p in self.resultado.puntos
            if p.fase in ("inicio", "+y", "-y") and p.confiable
        ]

    def _estimar_z_en_linea(self, y):
        """
        Ajusta z = m*y + n a los puntos interiores ya medidos sobre la
        recta x = x0 y evalúa en 'y'. Devuelve (z_estimado, texto). Si la
        recta no es de fiar devuelve (None, motivo): en ese caso el punto
        medio se reposiciona con posicionar_en_muestra().
        """

        pts = [p for p in self._puntos_de_la_recta_y() if not p.es_borde]

        if len(pts) < 3:
            return None, f"hay solo {len(pts)} puntos interiores sobre la recta (hacen falta 3)"

        ys = np.array([p.y for p in pts])
        zs = np.array([p.z for p in pts])

        if np.ptp(ys) < 0.1:
            return None, "los puntos de la recta abarcan menos de 0.1 mm en y"

        if not (ys.min() - 0.2 <= y <= ys.max() + 0.2):
            return None, "el punto medio caería fuera del tramo medido (habría que extrapolar)"

        m, n = np.polyfit(ys, zs, 1)

        rms = float(np.sqrt(np.mean((zs - (m * ys + n)) ** 2)))

        if rms > self.tolerancia_ajuste_linea_mm:
            return None, (
                f"la recta no ajusta bien (rms = {rms * 1000:.1f} µm > "
                f"{self.tolerancia_ajuste_linea_mm * 1000:.0f} µm)"
            )

        z_est = float(m * y + n)

        if not (self.z_min_seguridad <= z_est <= self.z_arriba_seguro):
            return None, f"z estimado = {z_est:.4f} mm fuera de [z_min_seguridad, z_arriba_seguro]"

        return z_est, f"recta con rms = {rms * 1000:.1f} µm, pendiente = {m * 1000:+.2f} µm/mm"

    # --------------------------------------------------------
    # Fases de la exploración
    # --------------------------------------------------------

    def _volver_al_inicio(self):

        if self._llego(self.x0, self.y0):
            return

        print(f"\nVolviendo a ({self.x0}, {self.y0}) para recorrer la otra dirección...")

        self._ir_a(self.x0, self.y0, self.z0)

    def _ir_al_medio(self):

        pts = self._puntos_de_la_recta_y()
        ys = [p.y for p in pts]

        y_medio = _a_grilla((max(ys) + min(ys)) / 2)

        z_est, texto = self._estimar_z_en_linea(y_medio)

        print(f"\nPunto medio en y: ({self.x0}, {y_medio})")

        if z_est is not None:
            print(f"z estimado ahí = {z_est:.4f} mm ({texto})")
            z_destino = z_est + self.margen_descenso_mm
        else:
            print(f"Sin estimación de z confiable: {texto}.")
            z_destino = self.z_arriba_seguro      # no baja

        largo = self._ir_a(self.x0, y_medio, z_destino)

        if largo and z_est is None:

            print("Reposicionando con posicionar_en_muestra()...")

            kwargs = dict(verificar_presencia=False, hacer_foco_final=False)
            kwargs.update(self.kwargs_posicionar)

            posicionar_en_muestra(
                camara=self.camara,
                impresora=self.impresora,
                x=self.x0,
                y=y_medio,
                z_arriba_seguro=self.z_arriba_seguro,
                z_min_seguridad=self.z_min_seguridad,
                feedrate_xy=self.feedrate_xy,
                espera_movimiento=self.espera_movimiento,
                **kwargs,
            )

            # posicionar_en_muestra() cambia la configuración de la cámara
            self._config_camara()

        punto = self._medir("medio", ancho=largo)

        if punto.es_borde:
            aviso = (
                "El punto medio en y ya es un borde "
                f"(distancia_normalizada = {punto.distancia_normalizada:.4f}): "
                "la muestra es angosta ahí. Se camina igual hacia x."
            )
            print(f"[AVISO] {aviso}")
            self.resultado.avisos.append(aviso)

        if not punto.confiable:
            raise CampoVisualSinMuestra(
                f"El punto medio ({self.x0}, {y_medio}) no tiene muestra."
            )

    # --------------------------------------------------------
    # Todo junto
    # --------------------------------------------------------

    def correr(self):

        r = self.resultado

        try:

            self._config_camara()

            p0 = self._medir("inicio")

            if not p0.confiable:
                raise CampoVisualSinMuestra(
                    "El campo visual inicial no tiene muestra "
                    f"(intensidad_fondo = {p0.intensidad_fondo:.1f})."
                )

            if p0.es_borde:
                raise CampoInicialEnBorde(
                    f"El campo visual inicial ({p0.x}, {p0.y}) ya es un borde: "
                    f"distancia_normalizada = {p0.distancia_normalizada:.4f} >= "
                    f"{self.umbral}. Arrancá desde un campo visual más adentro "
                    f"de la muestra."
                )

            self.x0, self.y0, self.z0 = p0.x, p0.y, p0.z

            self._caminar("y", +1, self.paso_y)

            self._volver_al_inicio()

            self._caminar("y", -1, self.paso_y)

            self._ir_al_medio()

            self._caminar("x", self.sentido_x, self.paso_x)

        except BaseException as e:
            # Si algo corta la exploración a la mitad (incluido Ctrl+C),
            # lo medido hasta ahí queda en e.resultado_parcial (y en el CSV).
            e.resultado_parcial = r
            raise

        r.resumen()

        return r


# ============================================================
# FUNCIÓN PÚBLICA
# ============================================================

def explorar_plano_enfoque(
    camara,
    impresora,
    umbral_distancia_borde,
    z_arriba_seguro,
    z_min_seguridad,
    carpeta_origen=None,
    guardar_fotos_autofoco=False,
    guardar_foto_maxima=False,
    paso_y_mm=0.15,
    paso_x_mm=0.10,
    sentido_x=+1,
    max_pasos=100,
    metrica=laplacian_variance,
    paso_z=0.0025,
    pasos_arriba_foco_local=3,
    pasos_abajo_foco_local=3,
    pasos_arriba_foco_medio=10,
    pasos_abajo_foco_medio=10,
    espera_movimiento=0.5,
    feedrate_xy=3000,
    umbral_intensidad_fondo=None,
    margen_descenso_mm=0.01,
    tolerancia_ajuste_linea_mm=0.02,
    graficar_autofoco=False,
    kwargs_autofocus=None,
    kwargs_config_camara=None,
    kwargs_posicionar=None,
):
    """
    Recorre la muestra en "T" y devuelve un ResultadoExploracion con los
    puntos (x, y, z_enfoque) (ver el docstring del módulo).

    Antes de llamarla la platina tiene que estar PARADA EN UN CAMPO
    VISUAL INTERIOR de la muestra (por ejemplo, después de
    posicionar_en_muestra()). Ese es el campo "inicio": se enfoca ahí y
    desde ahí arrancan las dos caminatas en y. Si resulta ser un borde se
    levanta CampoInicialEnBorde.

    Parámetros que no son obvios:

    - umbral_distancia_borde: un campo visual es "borde" si
      distancia_normalizada >= este valor (calibrado con
      analisis_campo_visual.resumir_calibracion()). Es obligatorio.
    - z_arriba_seguro / z_min_seguridad: los mismos de siempre (ej. 27.9 y
      27.5). Los viajes largos suben a z_arriba_seguro; autofocus() no
      baja de z_min_seguridad ni sube de z_arriba_seguro.
    - carpeta_origen: donde se guarda todo lo que se guarde. Con
      'guardar_fotos_autofoco' cada autofocus() deja sus fotos en una
      subcarpeta propia "0007 +y (x, y)"; con 'guardar_foto_maxima' la
      foto sacada en z_enfoque de cada campo visual queda en
      carpeta_origen, junto con un CSV con todos los puntos. Si es None
      no se guarda nada.
    - paso_y_mm / paso_x_mm: no pueden pasar de 0.15 mm (regla de
      seguridad de moverse_a_xy()). Se redondean a la grilla de 12.5 µm.
    - sentido_x: +1 o -1, hacia dónde se camina en x desde el punto medio.
    - max_pasos: tope de pasos por dirección (si se agota, se avisa).
    - metrica: nombre en FUNCIONES_METRICA ("laplacian", "tenengrad",
      "varianza") o una función imagen -> valor, como en barrido_espiral.
    - pasos_*_foco_local: barrido de Z de cada autofocus() entre campos
      visuales vecinos. pasos_*_foco_medio: el del punto medio, que viene
      de un viaje largo y tiene más incertidumbre en Z.
    - umbral_intensidad_fondo: opcional (None = desactivado). Si se pasa,
      un campo visual con intensidad_fondo <= este valor se considera SIN
      muestra: la caminata termina ahí y ese punto queda marcado como no
      confiable. Sirve por si un paso cae justo en el vacío, donde
      distancia_normalizada da ~0 y parecería "interior". OJO: el 152.5
      de verificar_presencia_muestra() está calibrado con RGB565; con
      GRAYSCALE hay que revisarlo.
    - margen_descenso_mm: en el punto medio, la platina baja hasta el z
      estimado + este margen y el autofocus() (barrido ancho) hace el
      resto.
    - tolerancia_ajuste_linea_mm: si la recta z(y) medida en la caminata
      en y tiene un rms mayor, no se la usa para estimar z en el punto
      medio y se reposiciona con posicionar_en_muestra().
    - kwargs_autofocus: se pasan a autofocus() (vecinos_para_confirmar,
      pasos_extension_maxima, ...).
    - kwargs_config_camara: se pasan a camara.config_autofoco_local()
      (exposure_us, gain_db).
    - kwargs_posicionar: se pasan a posicionar_en_muestra() si hace falta
      el reposicionado del punto medio.

    Si la exploración se corta (error, Ctrl+C), la excepción lleva lo
    medido hasta ahí en el atributo 'resultado_parcial'.
    """

    if z_min_seguridad >= z_arriba_seguro:
        raise ValueError("z_min_seguridad tiene que ser menor que z_arriba_seguro")

    if sentido_x not in (+1, -1):
        raise ValueError("sentido_x tiene que ser +1 o -1")

    if max_pasos < 1:
        raise ValueError("max_pasos tiene que ser >= 1")

    paso_x = _a_grilla(paso_x_mm)
    paso_y = _a_grilla(paso_y_mm)

    if paso_x < PASO_MINIMO_XY_MM or paso_y < PASO_MINIMO_XY_MM:
        raise ValueError(f"paso_x_mm/paso_y_mm tienen que ser >= {PASO_MINIMO_XY_MM} mm")

    if paso_x > DISTANCIA_XY_SIN_SUBIR_MM + 1e-9 or paso_y > DISTANCIA_XY_SIN_SUBIR_MM + 1e-9:
        raise ValueError(
            f"paso_x_mm/paso_y_mm no pueden pasar de {DISTANCIA_XY_SIN_SUBIR_MM} mm: "
            f"más allá de eso moverse_a_xy() tiene que subir a z_arriba_seguro "
            f"y el autofocus local ya no encontraría el foco."
        )

    if abs(paso_x - paso_x_mm) > 1e-9 or abs(paso_y - paso_y_mm) > 1e-9:
        print(
            f"[AVISO] Los pasos se ajustaron a la grilla de {PASO_MINIMO_XY_MM} mm: "
            f"x {paso_x_mm} -> {paso_x} | y {paso_y_mm} -> {paso_y}"
        )

    guarda_algo = guardar_fotos_autofoco or guardar_foto_maxima

    if guarda_algo and carpeta_origen is None:
        raise ValueError(
            "Si 'guardar_fotos_autofoco' o 'guardar_foto_maxima' es True "
            "hay que pasar 'carpeta_origen'"
        )

    if carpeta_origen is not None:
        carpeta_origen = Path(carpeta_origen)

    if isinstance(metrica, str):
        metrica = FUNCIONES_METRICA[metrica]

    if impresora.posicion_x is None or impresora.posicion_y is None or impresora.posicion_z is None:
        raise RuntimeError("Llamá primero a impresora.home()")

    exploracion = _Exploracion(
        camara=camara,
        impresora=impresora,
        metrica=metrica,
        umbral_distancia_borde=umbral_distancia_borde,
        z_arriba_seguro=z_arriba_seguro,
        z_min_seguridad=z_min_seguridad,
        carpeta_origen=carpeta_origen,
        guardar_fotos_autofoco=guardar_fotos_autofoco,
        guardar_foto_maxima=guardar_foto_maxima,
        paso_x=paso_x,
        paso_y=paso_y,
        sentido_x=sentido_x,
        max_pasos=max_pasos,
        paso_z=paso_z,
        pasos_foco_local=(pasos_arriba_foco_local, pasos_abajo_foco_local),
        pasos_foco_medio=(pasos_arriba_foco_medio, pasos_abajo_foco_medio),
        espera_movimiento=espera_movimiento,
        feedrate_xy=feedrate_xy,
        umbral_intensidad_fondo=umbral_intensidad_fondo,
        margen_descenso_mm=margen_descenso_mm,
        tolerancia_ajuste_linea_mm=tolerancia_ajuste_linea_mm,
        graficar_autofoco=graficar_autofoco,
        kwargs_autofocus=kwargs_autofocus or {},
        kwargs_config_camara=kwargs_config_camara or {},
        kwargs_posicionar=kwargs_posicionar or {},
    )

    return exploracion.correr()


#%%


# ============================================================
# EJEMPLO DE USO
# ============================================================

from camara_openmv import OpenMVCamera
from impresora_cosmos import TrimakerCOSMOS


grua = TrimakerCOSMOS("COM11"); grua.home()
cam = OpenMVCamera("COM7"); cam.conectar()


#%%


# 1) Parar la platina en un campo visual INTERIOR de la muestra
posicionar_en_muestra(cam, grua, x=150, y=150,
                    z_arriba_seguro=28.5,
                    z_min_seguridad=27.5, verificar_presencia=False,
                    hacer_foco_final=True, paso_grueso=0.005,
                    metrica_foco_final=laplacian_variance 
                    )


#%%


# 2) Barrido en T
resultado = explorar_plano_enfoque(
    cam, grua,
    umbral_distancia_borde=0.14,
    z_arriba_seguro=27.9,
    z_min_seguridad=27.5,
    carpeta_origen=r"C:\BackUp\Facultad\Labo 7\Fotos\26.10.08 plano",
    guardar_fotos_autofoco=True,
    guardar_foto_maxima=True,
)


#%%


# 3) Plano de enfoque
plano = resultado.ajustar_plano()        # sin los campos de borde
print(plano.z(150.5, 153.2))             # z estimado en (x, y)
xyz = resultado.puntos_xyz()             # array (N, 3) [x, y, z_enfoque]


#%%
