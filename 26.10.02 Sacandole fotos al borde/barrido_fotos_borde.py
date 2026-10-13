#%%

"""
Barrido de medición alrededor de un borde: grilla de campos visuales,
con una foto en COLOR (RGB) enfocada en cada uno.

Para qué sirve: reunir fotos de campos visuales de INTERIOR, BORDE y
VACÍO tomadas con el mismo procedimiento, para después clasificarlas a
mano (carpetas interior / borde / vacio) y calibrar los umbrales con
analisis_campo_visual.resumir_calibracion().

Flujo (para cada punto de la grilla):

  1) La platina se mueve al punto (pasos de <= 0.15 mm: el Z no se toca,
     ver más abajo).
  2) La cámara pasa a escala de grises QVGA (config_autofoco_local()) y
     autofocus() deja la platina en z_enfoque.
  3) La cámara pasa a color (por defecto RGB565 QVGA, calidad 100) y se
     saca UNA foto ahí, que es la que se guarda.
  4) Se calculan las características de esa foto (distancia_normalizada,
     contraste, fraccion_rojos, saturacion_media, ...) y se agrega una
     fila al CSV.

La grilla se recorre en serpentina. El punto (i=0, j=0) es donde está
parada la platina al llamar (o x_inicial / y_inicial); el índice i
avanza en x con 'paso_x_mm' y el j en y con 'paso_y_mm'. Los pasos
tienen SIGNO: uno negativo recorre hacia el otro lado.

Qué queda guardado, dentro de  <carpeta_salida>/<nombre del barrido>/:

  - Las fotos en color, directamente ahí, con nombre
        "NNNN i{i}_j{j} (x, y, z) - hora.jpg"
    (más " SIN FOCO" si en ese punto no hubo enfoque confirmado; ver
    abajo). Son las que hay que repartir en carpetas.
  - 'puntos.csv': una fila por foto con posición, z_enfoque, si hubo
    foco y todas las características de la foto.
  - 'parametros.json': con qué parámetros se hizo el barrido (útil
    para comparar barridos, sobre todo exposición y balance de blancos).
  - 'autofoco/': las fotos de cada autofocus() (solo si
    guardar_fotos_autofoco=True), una subcarpeta por punto.

Seguridad (regla validada en el laboratorio, ver moverse_a_xy() en
busqueda_muestra.py): parado en un z_enfoque se puede mover hasta
~150 µm en XY sin tocar el Z. Por eso paso_x_mm y paso_y_mm no pueden
pasar de 0.15 mm (se rechaza antes de mover nada). Si se pide un punto
inicial a más de 150 µm de donde está la platina, el viaje sube antes a
z_arriba_seguro, se mueve en XY y vuelve a bajar al Z donde estaba (con
un autofocus() más ancho). La grilla entera se valida contra el
rango de la impresora antes de empezar: la impresora ignora en silencio
los movimientos fuera de rango.

Campos sin enfoque: si autofocus() no confirma el máximo (típico en un
campo VACÍO, donde la métrica solo mide ruido), la platina queda en el
Z de antes y la foto se saca igual, marcada SIN FOCO: para clasificar
un campo vacío no hace falta que esté enfocado. Con
umbral_intensidad_fondo se puede evitar directamente correr autofocus()
en los campos claramente vacíos.

No modifica autofocus() ni ningún otro módulo.

Unidades: mm. Z creciente = alejarse de la muestra.
"""

# Agrego la raiz del proyecto
import sys
import os

os.environ["PYTHONPATH"] = r"C:\BackUp\Facultad\Labo 7\env_labo7"

sys.path.insert(0, os.environ["PYTHONPATH"])

import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from busqueda_muestra import (
    DISTANCIA_XY_SIN_SUBIR_MM,
    FUNCIONES_METRICA,
    PicoNoConfirmado,
    autofocus,
    bytes_a_imagen,
    intensidad_fondo,
    laplacian_variance,
    moverse_a_xy,
)


# ============================================================
# GRILLA DE LA IMPRESORA
# ============================================================
# La impresora se mueve en múltiplos de 0.0125 mm en x-y y de 0.0025 mm
# en z. Las coordenadas de destino se pasan por la grilla para que la
# posición que registra la impresora coincida con la pedida. (Se repite
# acá, en vez de importarlo de barrido_plano_enfoque.py, porque ese
# archivo trae celdas de aplicación que se ejecutan al importarlo.)

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
# CARACTERÍSTICAS DE CADA FOTO (módulo opcional)
# ============================================================

CARACTERISTICAS_CSV = (
    "distancia_normalizada",
    "contraste",
    "fraccion_oscura",
    "intensidad_fondo",
    "fraccion_rojos",
    "saturacion_media",
)


def _importar_caracteristicas():
    """
    Devuelve analisis_campo_visual.caracteristicas_borde, o None si no se
    puede importar. El barrido NO depende de eso (el análisis se puede
    hacer después con las fotos), así que un problema de importación
    solo deja el CSV sin esas columnas, con un aviso.

    analisis_campo_visual.py vive en otra carpeta del proyecto: si no se
    encuentra, se busca en las carpetas hermanas de la de este archivo.
    """

    def _intentar():
        from analisis_campo_visual import caracteristicas_borde
        return caracteristicas_borde

    try:
        return _intentar()
    except ImportError:
        pass
    except Exception as e:
        print(f"[AVISO] No se pudo importar analisis_campo_visual: {e!r}")
        return None

    raiz = Path(__file__).resolve().parent.parent

    for archivo in sorted(raiz.glob("*/analisis_campo_visual.py")):
        sys.path.append(str(archivo.parent))

    try:
        return _intentar()
    except Exception as e:
        print(
            f"[AVISO] No se encontró analisis_campo_visual ({e!r}): el CSV "
            f"va a quedar sin las características de cada foto."
        )
        return None


# ============================================================
# RESULTADO
# ============================================================

CAMPOS_CSV = [
    "indice", "i", "j", "x", "y", "z_enfoque", "enfocado", "motivo",
    "valor_metrica", *CARACTERISTICAS_CSV, "hora", "foto", "carpeta_autofocus",
]


@dataclass
class PuntoBarrido:
    indice: int                     # orden en que se recorrió (1, 2, 3, ...)
    i: int                          # índice en x dentro de la grilla
    j: int                          # índice en y dentro de la grilla
    x: float                        # [mm] posición real de la platina
    y: float
    z: float                        # [mm] z_enfoque (o el Z de antes si no hubo foco)
    enfocado: bool
    motivo: str                     # "ok", "pico_no_confirmado" o "vacio"
    valor_metrica: Optional[float]  # valor de la métrica en el máximo (None si no hubo autofocus)
    caracteristicas: dict = field(default_factory=dict)
    hora: str = ""
    ruta_foto: Optional[str] = None
    carpeta_autofocus: Optional[str] = None


@dataclass
class ResultadoBarridoFotos:
    carpeta: Path
    ruta_csv: Path
    puntos: list = field(default_factory=list)
    avisos: list = field(default_factory=list)

    def resumen(self):
        """Imprime cuántas fotos salieron, cuántas sin foco y dónde están."""

        n = len(self.puntos)
        sin_foco = [p for p in self.puntos if not p.enfocado]

        print("\n" + "=" * 60)
        print("RESUMEN DEL BARRIDO DE FOTOS")
        print("=" * 60)

        print(f"Fotos: {n} | con foco: {n - len(sin_foco)} | SIN FOCO: {len(sin_foco)}")

        motivos = {}
        for p in sin_foco:
            motivos[p.motivo] = motivos.get(p.motivo, 0) + 1
        for motivo, cuantas in motivos.items():
            print(f"  sin foco por '{motivo}': {cuantas}")

        enfocados = [p for p in self.puntos if p.enfocado]
        if enfocados:
            zs = [p.z for p in enfocados]
            print(f"z_enfoque de {min(zs):.4f} a {max(zs):.4f} mm")

        for aviso in self.avisos:
            print(f"  [AVISO] {aviso}")

        print(f"\nFotos y CSV en: {self.carpeta}")
        print(
            "Siguiente paso: repartir las fotos en carpetas interior / borde / "
            "vacio y correr analisis_campo_visual.resumir_calibracion()."
        )

    def graficar_mapa(self, caracteristica="distancia_normalizada", mostrar=True):
        """
        Mapa del barrido: un cuadrado por foto, en su posición (x, y) y
        coloreado según 'caracteristica' (cualquiera de CARACTERISTICAS_CSV,
        o "z_enfoque"). Los puntos SIN FOCO llevan una cruz roja y todos
        el número de foto, para poder ubicar el archivo. Ayuda a ver dónde
        quedó el borde dentro de la grilla.
        """

        import matplotlib.pyplot as plt

        if not self.puntos:
            raise ValueError("No hay puntos para graficar.")

        def valor(p):
            if caracteristica == "z_enfoque":
                return p.z
            v = p.caracteristicas.get(caracteristica)
            return np.nan if v is None else float(v)

        xs = np.array([p.x for p in self.puntos])
        ys = np.array([p.y for p in self.puntos])
        vs = np.array([valor(p) for p in self.puntos], dtype=float)

        fig, ax = plt.subplots()

        if np.all(np.isnan(vs)):
            print(f"[AVISO] Ningún punto tiene '{caracteristica}'.")
            ax.scatter(xs, ys, s=300, marker="s", color="lightgray")
        else:
            sc = ax.scatter(xs, ys, c=vs, s=300, marker="s", cmap="viridis")
            fig.colorbar(sc, ax=ax, label=caracteristica)

        sin_foco = [p for p in self.puntos if not p.enfocado]
        if sin_foco:
            ax.scatter(
                [p.x for p in sin_foco], [p.y for p in sin_foco],
                marker="x", color="red", s=80, label="sin foco",
            )
            ax.legend()

        for p in self.puntos:
            ax.annotate(str(p.indice), (p.x, p.y), ha="center", va="center", fontsize=6, color="white")

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"Barrido: {caracteristica}")
        ax.set_aspect("equal")

        if mostrar:
            plt.show()

        return fig


# ============================================================
# ORQUESTADOR INTERNO
# ============================================================

class _BarridoFotos:

    def __init__(self, camara, impresora, xs, ys, **opciones):
        self.camara = camara
        self.impresora = impresora
        self.xs = xs
        self.ys = ys
        self.__dict__.update(opciones)

        self.n_intentos = 0
        self.z_ref = None            # último z_enfoque confirmado
        self.caracteristicas_borde = None

        marca = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        nombre = self.nombre_barrido or f"barrido {marca} ({xs[0]}, {ys[0]})"
        self.carpeta = Path(self.carpeta_salida) / nombre
        self.carpeta.mkdir(parents=True, exist_ok=True)

        self.resultado = ResultadoBarridoFotos(
            carpeta=self.carpeta,
            ruta_csv=self.carpeta / "puntos.csv",
        )

    # --------------------------------------------------------
    # Cámara
    # --------------------------------------------------------

    def _config_gris(self):
        self.camara.config_autofoco_local(**self.kwargs_config_autofoco)

    def _config_color_y_capturar(self):
        """Pasa a color, saca la foto y devuelve sus bytes JPEG."""

        cam = self.camara

        cam.config_de_una(
            pixformat=self.pixformat_foto,
            resolucion=self.resolucion_foto,
            exposure_us=self.exposure_us,
            gain_db=self.gain_db,
            calidad=self.calidad_foto,
        )

        if self.whitebal is None:
            cam.set_whitebal_auto(True)
        else:
            cam.set_whitebal(*self.whitebal)

        if self.frames_extra_descartados > 0:
            cam.descartar_frames(self.frames_extra_descartados)

        return cam.capturar()

    def _restaurar_camara(self):
        """Deja la cámara como la deja posicionar_en_muestra(): gris QVGA,
        todo en automático. Nunca levanta (para no tapar otra excepción)."""

        try:
            self.camara.set_whitebal_auto(True)
            self.camara.config_autofoco_local(**self.kwargs_config_autofoco)
        except Exception as e:
            print(f"[AVISO] No se pudo restaurar la configuración de la cámara: {e!r}")

    # --------------------------------------------------------
    # Movimientos
    # --------------------------------------------------------

    def _llego(self, x, y):
        tolerancia = PASO_MINIMO_XY_MM / 2 + 1e-9
        return (
            abs(float(self.impresora.posicion_x) - x) <= tolerancia
            and abs(float(self.impresora.posicion_y) - y) <= tolerancia
        )

    def _ir_a(self, x, y):
        """
        Lleva la platina a (x, y) con moverse_a_xy(). Si fue un viaje
        largo (> 150 µm: la platina quedó en z_arriba_seguro, sin
        enfocar) baja al último z_enfoque confirmado (acotado a
        [z_min_seguridad, z_arriba_seguro] y redondeado hacia arriba a la
        grilla de 2.5 µm). Si fue corto, el Z no se toca. Devuelve True
        si fue un viaje largo.
        """

        imp = self.impresora

        if self._llego(x, y):
            return False

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

            z_dest = float(np.clip(self.z_ref, self.z_min_seguridad, self.z_arriba_seguro))
            z_dest = _z_a_grilla_hacia_arriba(z_dest)

            # Solo baja (nunca sube más de lo que ya subió moverse_a_xy)
            if z_dest < z_actual - 1e-9:
                imp.desp_lin_rel(0, 0, z_dest - z_actual)
                time.sleep(self.espera_movimiento)

        return largo

    # --------------------------------------------------------
    # Un punto: autofocus (gris) -> foto en color -> características
    # --------------------------------------------------------

    def _medir(self, i, j, ancho):

        imp = self.impresora
        cam = self.camara

        self.n_intentos += 1
        indice = self.n_intentos

        x = float(imp.posicion_x)
        y = float(imp.posicion_y)

        self._config_gris()

        enfocado = True
        motivo = "ok"
        valor_metrica = None
        carpeta_af = None

        # --- ¿campo claramente vacío? entonces no se corre autofocus()
        if self.umbral_intensidad_fondo is not None:

            gris = bytes_a_imagen(cam.capturar())
            fondo = float(intensidad_fondo(gris))

            if fondo <= self.umbral_intensidad_fondo:
                enfocado = False
                motivo = "vacio"

        # --- autofocus
        if enfocado:

            if self.guardar_fotos_autofoco:
                carpeta_af = self.carpeta / "autofoco" / f"{indice:04d} i{i}_j{j} ({x}, {y})"

            pasos_arriba, pasos_abajo = (
                (self.pasos_arriba_foco_ancho, self.pasos_abajo_foco_ancho)
                if ancho
                else (self.pasos_arriba_foco, self.pasos_abajo_foco)
            )

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

            try:
                _, _, valores = autofocus(
                    camara=cam,
                    impresora=imp,
                    metrica=self.metrica,
                    **kwargs,
                )
                valor_metrica = float(np.max(valores))

            except PicoNoConfirmado as e:
                # autofocus() ya restauró el Z de antes de empezar.
                enfocado = False
                motivo = "pico_no_confirmado"
                aviso = f"#{indice} i{i}_j{j} ({x}, {y}): autofocus() no confirmó el máximo: {e}"
                print(f"[AVISO] {aviso}")
                self.resultado.avisos.append(aviso)

        z = float(np.round(imp.posicion_z, 4))

        if enfocado:
            self.z_ref = z

        # --- foto en color
        image_data = self._config_color_y_capturar()

        hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        marca_foco = "" if enfocado else " SIN FOCO"

        ruta_foto = cam.guardar_imagen(
            image_data=image_data,
            carpeta=self.carpeta,
            nombre=f"{indice:04d} i{i}_j{j} ({x}, {y}, {z}){marca_foco} - {hora}",
            mostrar=False,
        )

        # --- características de la foto que se guardó
        caracteristicas = {}

        if self.caracteristicas_borde is not None:
            imagen = bytes_a_imagen(image_data)
            c = self.caracteristicas_borde(imagen)
            c["intensidad_fondo"] = float(intensidad_fondo(imagen))
            caracteristicas = {nombre: c.get(nombre) for nombre in CARACTERISTICAS_CSV}

        punto = PuntoBarrido(
            indice=indice,
            i=i,
            j=j,
            x=x,
            y=y,
            z=z,
            enfocado=enfocado,
            motivo=motivo,
            valor_metrica=valor_metrica,
            caracteristicas=caracteristicas,
            hora=hora,
            ruta_foto=str(ruta_foto),
            carpeta_autofocus=None if carpeta_af is None else str(carpeta_af),
        )

        self._registrar(punto)

        dist = caracteristicas.get("distancia_normalizada")
        texto_dist = "" if dist is None else f" | distancia_normalizada = {dist:.4f}"

        print(
            f"\n>>> #{indice} i{i}_j{j} ({x}, {y}, {z:.4f}) | "
            f"{'foco OK' if enfocado else 'SIN FOCO (' + motivo + ')'}{texto_dist}\n"
        )

        return punto

    def _registrar(self, punto):

        self.resultado.puntos.append(punto)

        ruta = self.resultado.ruta_csv
        nuevo = not ruta.exists()

        with open(ruta, "a", newline="", encoding="utf-8") as f:

            escritor = csv.DictWriter(f, fieldnames=CAMPOS_CSV)

            if nuevo:
                escritor.writeheader()

            fila = {
                "indice": punto.indice,
                "i": punto.i,
                "j": punto.j,
                "x": punto.x,
                "y": punto.y,
                "z_enfoque": punto.z,
                "enfocado": int(punto.enfocado),
                "motivo": punto.motivo,
                "valor_metrica": "" if punto.valor_metrica is None else punto.valor_metrica,
                "hora": punto.hora,
                "foto": punto.ruta_foto or "",
                "carpeta_autofocus": punto.carpeta_autofocus or "",
            }

            for nombre in CARACTERISTICAS_CSV:
                v = punto.caracteristicas.get(nombre)
                fila[nombre] = "" if v is None else v

            escritor.writerow(fila)

    # --------------------------------------------------------
    # Todo junto
    # --------------------------------------------------------

    def correr(self, parametros):

        r = self.resultado

        with open(self.carpeta / "parametros.json", "w", encoding="utf-8") as f:
            json.dump(parametros, f, ensure_ascii=False, indent=2, default=str)

        if self.calcular_caracteristicas:
            self.caracteristicas_borde = _importar_caracteristicas()

        _, _, z_inicial = self.impresora.pregunta_posicion_actual()
        self.z_ref = float(z_inicial)

        n_x, n_y = len(self.xs), len(self.ys)

        try:

            for j in range(n_y):

                orden_i = range(n_x) if j % 2 == 0 else range(n_x - 1, -1, -1)

                for i in orden_i:

                    largo = self._ir_a(self.xs[i], self.ys[j])

                    # Después de un viaje largo no se conoce el Z de enfoque
                    # de ahí: el primer autofocus() barre más ancho.
                    self._medir(i, j, ancho=largo)

        except BaseException as e:
            # Si algo corta el barrido a la mitad (incluido Ctrl+C), lo
            # medido hasta ahí queda en e.resultado_parcial (y en el CSV).
            e.resultado_parcial = r
            raise

        finally:
            self._restaurar_camara()

        r.resumen()

        return r


# ============================================================
# FUNCIÓN PÚBLICA
# ============================================================

def barrer_fotos_borde(
    camara,
    impresora,
    z_arriba_seguro,
    z_min_seguridad,
    carpeta_salida,
    n_x=8,
    n_y=8,
    paso_x_mm=0.075,
    paso_y_mm=0.100,
    x_inicial=None,
    y_inicial=None,
    nombre_barrido=None,
    pixformat_foto="RGB565",
    resolucion_foto="QVGA",
    calidad_foto=100,
    exposure_us=None,
    gain_db=None,
    whitebal=None,
    frames_extra_descartados=1,
    guardar_fotos_autofoco=False,
    metrica=laplacian_variance,
    paso_z=0.0025,
    pasos_arriba_foco=4,
    pasos_abajo_foco=4,
    pasos_arriba_foco_ancho=10,
    pasos_abajo_foco_ancho=10,
    umbral_intensidad_fondo=None,
    calcular_caracteristicas=True,
    espera_movimiento=0.5,
    feedrate_xy=3000,
    graficar_autofoco=False,
    kwargs_autofocus=None,
    kwargs_config_autofoco=None,
):
    """
    Recorre una grilla de n_x por n_y campos visuales y saca una foto en
    color enfocada en cada uno (ver el docstring del módulo). Devuelve un
    ResultadoBarridoFotos.

    Antes de llamarla la platina tiene que estar ENFOCADA en algún lugar
    de la muestra cerca de donde se quiere medir (por ejemplo, después
    de posicionar_en_muestra()): desde ese Z arrancan los autofocus.

    Parámetros que no son obvios:

    - carpeta_salida: se crea adentro una carpeta nueva por barrido
      (llamada 'nombre_barrido', o "barrido <fecha> (x, y)" si no se
      pasa), así que varios barridos pueden compartir 'carpeta_salida'.
    - n_x, n_y, paso_x_mm, paso_y_mm: la grilla. Los pasos tienen signo
      y no pueden pasar de 0.15 mm. Lo ideal es que sean múltiplos de
      0.0125 mm (lo que se mueve la impresora): los defaults (0.075 y
      0.100) lo son.
    - x_inicial, y_inicial: esquina de arranque (i=0, j=0). Si no se
      pasan, es la posición actual de la platina. Más lejos de 0.15 mm
      es un viaje largo (ver el docstring del módulo).
    - pixformat_foto, resolucion_foto, calidad_foto: formato de la foto
      en color (default RGB565, QVGA, calidad JPEG 100).
    - exposure_us, gain_db, whitebal: parámetros FIJOS de la cámara para
      la foto en color; None = automático. Para comparar fraccion_rojos
      y saturacion_media entre fotos conviene fijarlos: el balance de
      blancos y la exposición automáticos se reajustan en cada foto y
      cambian los colores. whitebal es una tupla (r, g, b) de ganancias
      en dB. Al terminar el balance de blancos vuelve a automático.
    - frames_extra_descartados: fotos que se descartan después de pasar
      a color, para que el sensor se acomode (config_de_una() ya
      descarta una).
    - guardar_fotos_autofoco: guarda las fotos de cada autofocus() en
      'autofoco/' (una subcarpeta por punto).
    - metrica: función de enfoque (o una clave de FUNCIONES_METRICA).
    - pasos_*_foco: barrido de Z de autofocus() alrededor del Z actual;
      los '_ancho' se usan solo después de un viaje largo.
    - umbral_intensidad_fondo: si no es None, antes de enfocar se saca
      una foto en gris y, si intensidad_fondo <= este valor (campo
      claro = vacío), no se corre autofocus(): la foto se saca en el Z
      de antes y queda marcada SIN FOCO (motivo "vacio"). Sirve para que
      en los campos vacíos el Z no se desplace siguiendo ruido. Ojo: el
      valor de 152.5 de posicionar_en_muestra() se calibró con RGB565.
    - calcular_caracteristicas: calcula las características de cada foto
      con analisis_campo_visual.caracteristicas_borde() (columnas del
      CSV). Si ese módulo no se puede importar, se avisa y se sigue.
    - kwargs_autofocus / kwargs_config_autofoco: parámetros extra para
      autofocus() y para camara.config_autofoco_local() (por ejemplo
      {"exposure_us": 8000, "gain_db": 0}).
    """

    # ------------------------------------------------------------
    # Validaciones: todas ANTES de mover nada
    # ------------------------------------------------------------

    if impresora.posicion_x is None or impresora.posicion_y is None or impresora.posicion_z is None:
        raise RuntimeError("Llamá primero a impresora.home()")

    if int(n_x) != n_x or int(n_y) != n_y or n_x < 1 or n_y < 1:
        raise ValueError(f"n_x y n_y tienen que ser enteros >= 1 (recibí {n_x} y {n_y}).")

    n_x, n_y = int(n_x), int(n_y)

    for nombre, paso, n in (("paso_x_mm", paso_x_mm, n_x), ("paso_y_mm", paso_y_mm, n_y)):

        if n == 1:
            continue

        if paso == 0 or abs(paso) > DISTANCIA_XY_SIN_SUBIR_MM + 1e-9:
            raise ValueError(
                f"{nombre} = {paso}: tiene que ser distinto de 0 y de a lo sumo "
                f"{DISTANCIA_XY_SIN_SUBIR_MM} mm en valor absoluto (regla de "
                f"seguridad: más lejos que eso, hay que subir antes de moverse)."
            )

        multiplos = abs(paso) / PASO_MINIMO_XY_MM
        if abs(multiplos - round(multiplos)) > 1e-6:
            print(
                f"[AVISO] {nombre} = {paso} no es múltiplo de {PASO_MINIMO_XY_MM} mm: "
                f"la impresora lo redondea y el espaciado real va a ser desparejo."
            )

    if isinstance(metrica, str):
        metrica = FUNCIONES_METRICA[metrica]

    if whitebal is not None and len(whitebal) != 3:
        raise ValueError("whitebal tiene que ser una tupla (r, g, b) o None.")

    _, _, z_actual = impresora.pregunta_posicion_actual()

    if z_actual < z_min_seguridad - 1e-9:
        raise ValueError(
            f"La platina está en Z = {z_actual} mm, por debajo de "
            f"z_min_seguridad = {z_min_seguridad} mm."
        )

    x0 = float(impresora.posicion_x) if x_inicial is None else float(x_inicial)
    y0 = float(impresora.posicion_y) if y_inicial is None else float(y_inicial)

    xs = [_a_grilla(x0 + k * paso_x_mm) for k in range(n_x)]
    ys = [_a_grilla(y0 + k * paso_y_mm) for k in range(n_y)]

    # La impresora IGNORA en silencio un movimiento fuera de rango
    max_x = getattr(impresora, "max_posicion_x", 270)
    max_y = getattr(impresora, "max_posicion_y", 198)

    if min(xs) < 0 or max(xs) > max_x or min(ys) < 0 or max(ys) > max_y:
        raise ValueError(
            f"La grilla se sale del rango de la impresora: x de {min(xs)} a {max(xs)} "
            f"(0 a {max_x}), y de {min(ys)} a {max(ys)} (0 a {max_y})."
        )

    parametros = dict(
        fecha=datetime.now().isoformat(timespec="seconds"),
        n_x=n_x, n_y=n_y, paso_x_mm=paso_x_mm, paso_y_mm=paso_y_mm,
        x_inicial=xs[0], y_inicial=ys[0], x_final=xs[-1], y_final=ys[-1],
        z_inicial=float(z_actual),
        z_arriba_seguro=z_arriba_seguro, z_min_seguridad=z_min_seguridad,
        pixformat_foto=pixformat_foto, resolucion_foto=resolucion_foto,
        calidad_foto=calidad_foto, exposure_us=exposure_us, gain_db=gain_db,
        whitebal=whitebal, frames_extra_descartados=frames_extra_descartados,
        guardar_fotos_autofoco=guardar_fotos_autofoco,
        metrica=getattr(metrica, "__name__", str(metrica)), paso_z=paso_z,
        pasos_arriba_foco=pasos_arriba_foco, pasos_abajo_foco=pasos_abajo_foco,
        pasos_arriba_foco_ancho=pasos_arriba_foco_ancho,
        pasos_abajo_foco_ancho=pasos_abajo_foco_ancho,
        umbral_intensidad_fondo=umbral_intensidad_fondo,
        espera_movimiento=espera_movimiento, feedrate_xy=feedrate_xy,
        kwargs_autofocus=kwargs_autofocus, kwargs_config_autofoco=kwargs_config_autofoco,
    )

    barrido = _BarridoFotos(
        camara, impresora, xs, ys,
        carpeta_salida=carpeta_salida,
        nombre_barrido=nombre_barrido,
        z_arriba_seguro=z_arriba_seguro,
        z_min_seguridad=z_min_seguridad,
        pixformat_foto=pixformat_foto,
        resolucion_foto=resolucion_foto,
        calidad_foto=calidad_foto,
        exposure_us=exposure_us,
        gain_db=gain_db,
        whitebal=None if whitebal is None else tuple(whitebal),
        frames_extra_descartados=frames_extra_descartados,
        guardar_fotos_autofoco=guardar_fotos_autofoco,
        metrica=metrica,
        paso_z=paso_z,
        pasos_arriba_foco=pasos_arriba_foco,
        pasos_abajo_foco=pasos_abajo_foco,
        pasos_arriba_foco_ancho=pasos_arriba_foco_ancho,
        pasos_abajo_foco_ancho=pasos_abajo_foco_ancho,
        umbral_intensidad_fondo=umbral_intensidad_fondo,
        calcular_caracteristicas=calcular_caracteristicas,
        espera_movimiento=espera_movimiento,
        feedrate_xy=feedrate_xy,
        graficar_autofoco=graficar_autofoco,
        kwargs_autofocus=kwargs_autofocus or {},
        kwargs_config_autofoco=kwargs_config_autofoco or {},
    )

    return barrido.correr(parametros)
