# %%

"""
Prueba de centrado -- ¿la impresora deja en el centro de la imagen el píxel que se eligió?

Carpeta: 26.10.29 test de centrado          Inicio: 2026-10-29
Objetivo: comprobar, aislado de la detección y del barrido, que mover la impresora para "centrar" un
    píxel de la imagen lo deja de verdad en el centro de la imagen.
Enfoque (qué se propone para resolverlo): una prueba corta y repetible.
    1) Foto del campo actual (se guarda).
    2) Se elige un píxel al azar: los alto x ancho píxeles tienen la misma probabilidad.
    3) Se guarda la misma foto con el píxel marcado, la cruz del centro y una flecha hacia donde hay
       que llevarlo (el movimiento que se va a pedir, en µm).
    4) La impresora se mueve a la posición alcanzable que lo centra (la cuenta de centrado.py).
    5) Segunda foto, ya centrada (se guarda).
    6) Se busca en la segunda foto lo que rodeaba al píxel en la primera (matchTemplate) y se mide
       a cuántos µm del centro quedó. Así la prueba da un número, no solo imágenes para mirar.
    Se guarda además una imagen que muestra antes y después lado a lado.
    Nada de esto detecta objetos: sirve cualquier campo con textura (glóbulos).
Módulos base que usa: camara_openmv (bytes_a_imagen; OpenMVCamera llega por parámetro),
    impresora_cosmos (moverse_a_xy; TrimakerCOSMOS llega por parámetro y de ahí se lee la grilla,
    impresora.resolucion_x / resolucion_y).
De otra aplicación: centrado.py de 26.10.16 (Calibracion, centrar_pixel, esta_centrado,
    pixel_a_desplazamiento_um). Es la cuenta que usa el barrido y es justo lo que se quiere probar,
    por eso se importa de ahí, sin copiarla ni tocarla.
Propuestas para la base: el campo visual de la cámara por resolución (bloque de abajo).
Estado: EN PRUEBA (todavía no se corrió en el equipo)        Resultado: -
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from camara_openmv import bytes_a_imagen
from impresora_cosmos import moverse_a_xy

# centrado.py vive en la aplicación 26.10.16. Se agrega AL FINAL de sys.path para que esa carpeta no
# tape a ningún módulo de la base.
CARPETA_CENTRADO = Path(__file__).resolve().parent.parent / "26.10.16 detector de puntos violetas y sus fotos"
if str(CARPETA_CENTRADO) not in sys.path:
    sys.path.append(str(CARPETA_CENTRADO))

from centrado import Calibracion, centrar_pixel, esta_centrado, pixel_a_desplazamiento_um  # noqa: E402


# ======================================================================
# PROPUESTAS PARA LA BASE (REGLAS_CODIGO.txt, sección 4)
# ======================================================================
# La base todavía no tiene el campo visual de la cámara. Hasta que lo tenga se define acá, en un solo
# lugar (BITACORA.txt, pendiente 2: pasarlo a camara_openmv.py).
#
#   FOV_X_MM = 0.09263 mm   campo visual a lo largo del eje x de la impresora (= filas de la imagen, 240 px)
#   FOV_Y_MM = 0.1243 mm    campo visual a lo largo del eje y de la impresora (= columnas, 320 px)
#   Valen SOLO en QVGA (320 x 240 px) y con la lente nueva (~0.388 µm/px en filas y en columnas).
#   Origen: medidos el 2026-10-22 moviendo la platina una distancia conocida; ver
#   PROYECTO_CONTEXTO.txt, sección 2.
#   OJO con el orden: x = filas = el lado CORTO (240 px). En 26.10.16 estaban al revés.
FOV_X_MM = 0.09263
FOV_Y_MM = 0.1243
RESOLUCION_DEL_FOV = "QVGA"
ALTO_PX, ANCHO_PX = 240, 320

# ======================================================================
# CONSTANTES DE LA PRUEBA
# ======================================================================
PIXFORMAT = "RGB565"            # color: bytes_a_imagen() lo entrega en BGR
MEDIO_LADO_PARCHE_PX = 24       # el parche que se busca en la foto de después mide (2 * 24 + 1) px de lado
CORRELACION_MINIMA = 0.5        # por debajo de esto no se confía en dónde se ubicó el parche
AVISO_CORRIMIENTO_PX = 4.0      # si lo que se corrió la imagen difiere de lo pedido en más que esto (~1.5 µm; matchTemplate
                                # tiene ~1 px de error) se avisa: la escala o los signos no son los que supone la calibración
AUMENTO = 2                     # las imágenes marcadas se guardan ampliadas (x2, píxel a píxel) para que se lean

_FUENTE = cv2.FONT_HERSHEY_SIMPLEX
_ROJO, _VERDE, _CELESTE, _BLANCO, _MAGENTA = (60, 60, 255), (60, 200, 60), (255, 200, 0), (255, 255, 255), (255, 0, 255)   # BGR


def crear_calibracion() -> Calibracion:
    """Calibracion de QVGA con la lente nueva (escala en µm/px y ejes), a partir de FOV_X_MM y FOV_Y_MM."""
    return Calibracion.desde_fov_mm(ANCHO_PX, ALTO_PX, FOV_X_MM, FOV_Y_MM)


# ======================================================================
# ELEGIR Y MARCAR EL PÍXEL
# ======================================================================
def elegir_pixel_al_azar(alto: int, ancho: int, generador: np.random.Generator) -> tuple[int, int]:
    """Elige un píxel (col, fila) al azar; los alto x ancho píxeles tienen exactamente la misma probabilidad."""
    indice = int(generador.integers(alto * ancho))
    return indice % ancho, indice // ancho


def _coordenada_ampliada(valor: float, aumento: int) -> int:
    """Coordenada de la imagen original -> coordenada en la ampliada (el centro del píxel [i, j] es (j, i))."""
    return int(round((valor + 0.5) * aumento - 0.5))


def _ampliar(imagen: np.ndarray, aumento: int) -> np.ndarray:
    if imagen.ndim == 2:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    return cv2.resize(imagen, None, fx=aumento, fy=aumento, interpolation=cv2.INTER_NEAREST)


def _franja_de_texto(imagen: np.ndarray, lineas: list[str]) -> None:
    """Escribe las líneas arriba a la izquierda, sobre una franja oscura para que se lean sobre cualquier fondo."""
    escala, paso = 0.5, 19
    ancho = max(cv2.getTextSize(linea, _FUENTE, escala, 1)[0][0] for linea in lineas) + 12
    alto = paso * len(lineas) + 8
    franja = imagen[:alto, :ancho]
    franja[:] = (franja * 0.3).astype(np.uint8)
    for k, linea in enumerate(lineas):
        cv2.putText(imagen, linea, (6, 18 + k * paso), _FUENTE, escala, _BLANCO, 1, cv2.LINE_AA)


def _cruz_del_centro(imagen: np.ndarray, calibracion: Calibracion, aumento: int) -> tuple[int, int]:
    centro = (_coordenada_ampliada(calibracion.centro_col, aumento), _coordenada_ampliada(calibracion.centro_fila, aumento))
    cv2.drawMarker(imagen, centro, _BLANCO, cv2.MARKER_CROSS, 11 * aumento, 1, cv2.LINE_AA)
    return centro


def marcar_pixel(imagen_bgr: np.ndarray, col: int, fila: int, calibracion: Calibracion,
                 aumento: int = AUMENTO) -> np.ndarray:
    """Copia ampliada de la foto con el píxel elegido, la cruz del centro y la flecha hacia donde hay que llevarlo."""
    imagen = _ampliar(imagen_bgr, aumento)
    centro = _cruz_del_centro(imagen, calibracion, aumento)
    punto = (_coordenada_ampliada(col, aumento), _coordenada_ampliada(fila, aumento))
    dx_um, dy_um = pixel_a_desplazamiento_um(calibracion, col, fila)

    cv2.arrowedLine(imagen, punto, centro, _CELESTE, 1, cv2.LINE_AA, tipLength=0.05)
    cv2.circle(imagen, punto, 6 * aumento, _VERDE, 1, cv2.LINE_AA)
    cv2.drawMarker(imagen, punto, _ROJO, cv2.MARKER_CROSS, 8 * aumento, 1, cv2.LINE_AA)
    _franja_de_texto(imagen, [
        f"ANTES. Pixel elegido: col={col} fila={fila}",
        f"Hay que mover: dx={dx_um:+.1f} um (x, filas)  dy={dy_um:+.1f} um (y, columnas)",
        "cruz blanca = centro de la imagen; flecha = hacia donde hay que llevarlo",
    ])
    return imagen


def marcar_resultado(imagen_bgr: np.ndarray, calibracion: Calibracion, esperado_px: tuple[float, float],
                     hallado: tuple[float, float, float] | None, residuo_medido_um: tuple[float, float] | None,
                     centrado: bool | None, aumento: int = AUMENTO) -> np.ndarray:
    """Copia ampliada de la foto de después con la cruz del centro, dónde debería haber quedado lo elegido
    (esperado_px, por la grilla de la impresora) y dónde quedó de verdad (hallado)."""
    imagen = _ampliar(imagen_bgr, aumento)
    _cruz_del_centro(imagen, calibracion, aumento)
    cv2.drawMarker(imagen, (_coordenada_ampliada(esperado_px[0], aumento), _coordenada_ampliada(esperado_px[1], aumento)),
                   _MAGENTA, cv2.MARKER_TILTED_CROSS, 8 * aumento, 1, cv2.LINE_AA)
    lineas = ["DESPUES de mover. Cruz blanca = centro de la imagen",
              f"X magenta = donde deberia quedar segun la grilla de {calibracion.paso_um:g} um"]
    if hallado is None:
        lineas.append("No se pudo ubicar el pixel elegido en esta foto (parche sin textura o muy cambiado)")
    else:
        col, fila, correlacion = hallado
        punto = (_coordenada_ampliada(col, aumento), _coordenada_ampliada(fila, aumento))
        cv2.circle(imagen, punto, 6 * aumento, _VERDE, 1, cv2.LINE_AA)
        cv2.drawMarker(imagen, punto, _ROJO, cv2.MARKER_CROSS, 8 * aumento, 1, cv2.LINE_AA)
        rx, ry = residuo_medido_um
        lineas.append(f"Circulo verde = donde quedo el pixel elegido (correlacion {correlacion:.2f})")
        lineas.append(f"Le falta dx={rx:+.1f} um  dy={ry:+.1f} um -> {'CENTRADO' if centrado else 'SIN CENTRAR'}"
                      f" (tolerancia +-{calibracion.paso_um / 2:.2f} um)")
    _franja_de_texto(imagen, lineas)
    return imagen


# ======================================================================
# MEDIR DÓNDE QUEDÓ
# ======================================================================
def buscar_parche(antes: np.ndarray, despues: np.ndarray, col: int, fila: int,
                  medio_lado_px: int = MEDIO_LADO_PARCHE_PX,
                  correlacion_minima: float = CORRELACION_MINIMA) -> tuple[float, float, float] | None:
    """Dónde quedó, en la foto de después, lo que rodeaba al píxel (col, fila) de la foto de antes.

    Recorta de 'antes' un parche centrado en el píxel (más chico si el píxel está cerca del borde) y lo busca en
    'despues' con matchTemplate (correlación normalizada: no le importa un cambio de brillo). Devuelve
    (col, fila, correlacion) del píxel elegido en 'despues', o None si la correlación es menor que
    'correlacion_minima' (parche liso, o la imagen cambió demasiado).
    """
    gris_antes = cv2.cvtColor(antes, cv2.COLOR_BGR2GRAY) if antes.ndim == 3 else antes
    gris_despues = cv2.cvtColor(despues, cv2.COLOR_BGR2GRAY) if despues.ndim == 3 else despues
    alto, ancho = gris_antes.shape
    f0, f1 = max(0, fila - medio_lado_px), min(alto, fila + medio_lado_px + 1)
    c0, c1 = max(0, col - medio_lado_px), min(ancho, col + medio_lado_px + 1)

    resultado = np.nan_to_num(cv2.matchTemplate(gris_despues, gris_antes[f0:f1, c0:c1], cv2.TM_CCOEFF_NORMED),
                              nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, correlacion, _, esquina = cv2.minMaxLoc(resultado)
    if correlacion < correlacion_minima:
        return None
    return float(esquina[0] + (col - c0)), float(esquina[1] + (fila - f0)), float(correlacion)


# ======================================================================
# LA PRUEBA
# ======================================================================
@dataclass
class ResultadoPrueba:
    """Lo que pasó en una prueba. Los µm van en los ejes de la impresora: (x = filas, y = columnas)."""
    numero: int
    col: int                                    # píxel elegido, en la foto de antes
    fila: int
    movimiento_pedido_um: tuple[float, float]   # (dx, dy) que hay que mover para llevarlo al centro, sin redondear a la grilla
    pos_antes_mm: tuple[float, float]
    pos_destino_mm: tuple[float, float]         # la alcanzable que calculó centrar_pixel()
    pos_despues_mm: tuple[float, float]         # donde quedó la impresora según el programa
    residuo_esperado_um: tuple[float, float]    # lo que sobra por la grilla de 12.5 µm (|valor| <= 6.25 µm)
    desplazamiento_pedido_px: tuple[float, float]    # (col, fila): cuánto debería correrse el píxel en la imagen
    hallado: tuple[float, float, float] | None  # (col, fila, correlación) del píxel en la foto de después; None = no se ubicó
    desplazamiento_medido_px: tuple[float, float] | None
    residuo_medido_um: tuple[float, float] | None   # cuánto le falta al píxel elegido para estar en el centro
    centrado: bool | None                       # |residuo medido| <= medio paso en x y en y; None = no se pudo medir
    archivos: dict[str, str] = field(default_factory=dict)

    @property
    def diferencia_de_corrimiento_px(self) -> float | None:
        """Cuánto difiere (en px, el peor de los dos ejes) lo que se corrió la imagen de lo que se pedía; None si no se midió."""
        if self.desplazamiento_medido_px is None:
            return None
        return max(abs(m - p) for m, p in zip(self.desplazamiento_medido_px, self.desplazamiento_pedido_px))


def _capturar(camara, frames_a_descartar: int) -> bytes:
    """Foto de la cámara, tirando antes 'frames_a_descartar' (la cámara puede devolver cuadros viejos)."""
    if frames_a_descartar > 0:
        camara.descartar_frames(cantidad=frames_a_descartar)
    return camara.capturar()


def _guardar_png(ruta: Path, imagen: np.ndarray) -> str:
    if not cv2.imwrite(str(ruta), imagen):
        raise OSError(f"No pude escribir {ruta}")
    return str(ruta)


def _texto_resultado(r: ResultadoPrueba, paso_um: float) -> str:
    lineas = [
        f"Prueba {r.numero:02d}: píxel elegido (col={r.col}, fila={r.fila})",
        f"  pedido: mover dx={r.movimiento_pedido_um[0]:+.1f} µm (x, filas) y dy={r.movimiento_pedido_um[1]:+.1f} µm (y, columnas)",
        f"  posición: antes {r.pos_antes_mm} -> destino {r.pos_destino_mm} -> después {r.pos_despues_mm} mm",
        f"  residuo esperado por la grilla: dx={r.residuo_esperado_um[0]:+.1f} µm, dy={r.residuo_esperado_um[1]:+.1f} µm",
    ]
    if r.hallado is None:
        lineas.append("  NO SE PUDO MEDIR: no se ubicó el píxel elegido en la foto de después (mirá las imágenes).")
    else:
        lineas.append(f"  corrimiento de la imagen (col, fila) en px: pedido ({r.desplazamiento_pedido_px[0]:+.1f}, "
                      f"{r.desplazamiento_pedido_px[1]:+.1f})   medido ({r.desplazamiento_medido_px[0]:+.1f}, "
                      f"{r.desplazamiento_medido_px[1]:+.1f})")
        lineas.append(f"  residuo medido en la foto de después: dx={r.residuo_medido_um[0]:+.1f} µm, "
                      f"dy={r.residuo_medido_um[1]:+.1f} µm (correlación {r.hallado[2]:.2f})  ->  "
                      f"{'CENTRADO' if r.centrado else 'SIN CENTRAR'} (tolerancia ±{paso_um / 2:.2f} µm)")
        if r.diferencia_de_corrimiento_px > AVISO_CORRIMIENTO_PX:
            lineas.append(f"  OJO: la imagen se corrió {r.diferencia_de_corrimiento_px:.1f} px distinto de lo pedido (más de "
                          f"{AVISO_CORRIMIENTO_PX:g} px): revisar la escala o los signos de la calibración, aunque haya quedado centrado.")
    return "\n".join(lineas)


def probar_centrado(camara, impresora, calibracion: Calibracion, z_arriba_seguro: float, carpeta_salida,
                    numero: int = 1, generador: np.random.Generator | None = None,
                    configuracion_camara: dict | None = None, frames_a_descartar: int = 2,
                    volver_al_inicio: bool = True, feedrate_xy: int = 3000, espera_movimiento: float = 0.5,
                    devolveme_la_respuesta: bool = True) -> ResultadoPrueba:
    """Una prueba de centrado: foto, píxel al azar, foto marcada, centrar, foto centrada y medición.

    La cámara tiene que estar sobre la muestra y enfocada, y la impresora con home() hecho. Solo se mueve en XY
    (y como mucho unas decenas de µm: nunca pasa de DISTANCIA_XY_SIN_SUBIR_MM, así que Z no cambia).
    Guarda en carpeta_salida: prueba_NN_1_antes.jpg, prueba_NN_2_antes_marcada.png, prueba_NN_3_despues.jpg y
    prueba_NN_4_comparacion.png (antes y después lado a lado).

    configuracion_camara: opciones extra para camara.config_de_una() (exposure_us, gain_db, whitebal...).
        La cámara se vuelve a configurar en QVGA a color al empezar cada prueba, venga como venga.
    frames_a_descartar: cuadros que se tiran antes de cada foto, para no recibir uno viejo (de antes de que
        la platina frenara). 0 = no tirar ninguno.
    volver_al_inicio: al terminar, la impresora vuelve a donde estaba, así cada prueba arranca en el mismo campo.
    Lanza ValueError si la foto no mide lo que supone la calibración (antes de mover nada) y RuntimeError si
    la impresora no llegó adonde se le pidió.
    """
    if impresora.posicion_x is None or impresora.posicion_y is None:
        raise RuntimeError("Llamá primero a impresora.home()")
    if not (np.isclose(calibracion.paso_mm, impresora.resolucion_x) and np.isclose(calibracion.paso_mm, impresora.resolucion_y)):
        raise ValueError(f"La calibración supone una grilla de {calibracion.paso_mm} mm y la impresora tiene "
                         f"{impresora.resolucion_x} x {impresora.resolucion_y} mm")

    generador = generador if generador is not None else np.random.default_rng()
    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)
    prefijo = f"prueba_{numero:02d}"

    # 1) foto del campo actual
    camara.config_de_una(pixformat=PIXFORMAT, resolucion=RESOLUCION_DEL_FOV, **(configuracion_camara or {}))
    datos_antes = _capturar(camara, frames_a_descartar)
    antes = bytes_a_imagen(datos_antes)
    if antes.shape[:2] != (calibracion.alto_px, calibracion.ancho_px):
        raise ValueError(f"La foto mide {antes.shape[1]} x {antes.shape[0]} px y la calibración supone "
                         f"{calibracion.ancho_px} x {calibracion.alto_px} px ({RESOLUCION_DEL_FOV}). No se movió nada.")

    # 2) píxel al azar y a dónde hay que ir para centrarlo
    col, fila = elegir_pixel_al_azar(calibracion.alto_px, calibracion.ancho_px, generador)
    pos_antes = (float(impresora.posicion_x), float(impresora.posicion_y))
    objetivo = centrar_pixel(calibracion, pos_antes, col, fila)
    movimiento_pedido = pixel_a_desplazamiento_um(calibracion, col, fila)

    # 3) guardar la foto y la foto con el píxel marcado
    archivos = {
        "antes": str(camara.guardar_imagen(datos_antes, carpeta, nombre=f"{prefijo}_1_antes")),
        "antes_marcada": _guardar_png(carpeta / f"{prefijo}_2_antes_marcada.png",
                                      marcar_pixel(antes, col, fila, calibracion)),
    }

    # 4) centrar: moverse_a_xy() es el camino para todo movimiento en XY (REGLAS_CODIGO.txt, sección 7)
    viaje_largo = moverse_a_xy(impresora, objetivo.x_mm, objetivo.y_mm, z_arriba_seguro,
                               feedrate_xy=feedrate_xy, espera_movimiento=espera_movimiento)
    pos_despues = (float(impresora.posicion_x), float(impresora.posicion_y))
    if viaje_largo:
        raise RuntimeError("moverse_a_xy() lo tomó como viaje largo: la platina subió a z_arriba_seguro y no está enfocada.")
    if abs(pos_despues[0] - objetivo.x_mm) > calibracion.paso_mm / 2 or abs(pos_despues[1] - objetivo.y_mm) > calibracion.paso_mm / 2:
        raise RuntimeError(f"La impresora no llegó a {(objetivo.x_mm, objetivo.y_mm)} y quedó en {pos_despues} "
                           "(¿destino fuera de rango? desp_lin_000() cancela esas órdenes avisando solo por pantalla).")

    # 5) foto centrada
    datos_despues = _capturar(camara, frames_a_descartar)
    despues = bytes_a_imagen(datos_despues)
    archivos["despues"] = str(camara.guardar_imagen(datos_despues, carpeta, nombre=f"{prefijo}_3_despues"))

    # 6) medir dónde quedó lo que se eligió
    hallado = buscar_parche(antes, despues, col, fila)
    residuo_medido = desplazamiento_medido = centrado = None
    if hallado is not None:
        residuo_medido = pixel_a_desplazamiento_um(calibracion, hallado[0], hallado[1])
        desplazamiento_medido = (hallado[0] - col, hallado[1] - fila)
        centrado = bool(esta_centrado(calibracion, hallado[0], hallado[1]))

    comparacion = np.hstack([marcar_pixel(antes, col, fila, calibracion),
                             marcar_resultado(despues, calibracion, objetivo.px_esperado, hallado, residuo_medido, centrado)])
    archivos["comparacion"] = _guardar_png(carpeta / f"{prefijo}_4_comparacion.png", comparacion)

    if volver_al_inicio:
        moverse_a_xy(impresora, pos_antes[0], pos_antes[1], z_arriba_seguro,
                     feedrate_xy=feedrate_xy, espera_movimiento=espera_movimiento)

    resultado = ResultadoPrueba(
        numero=numero, col=col, fila=fila, movimiento_pedido_um=movimiento_pedido,
        pos_antes_mm=pos_antes, pos_destino_mm=(objetivo.x_mm, objetivo.y_mm), pos_despues_mm=pos_despues,
        residuo_esperado_um=objetivo.residuo_um,
        desplazamiento_pedido_px=(objetivo.px_esperado[0] - col, objetivo.px_esperado[1] - fila),
        hallado=hallado, desplazamiento_medido_px=desplazamiento_medido, residuo_medido_um=residuo_medido,
        centrado=centrado, archivos=archivos)

    if devolveme_la_respuesta:
        print(_texto_resultado(resultado, calibracion.paso_um))
    return resultado


def resumir_pruebas(resultados: list[ResultadoPrueba]) -> str:
    """Una línea: cuántas pruebas quedaron centradas, sin centrar y sin poder medir."""
    centradas = sum(r.centrado is True for r in resultados)
    sin_centrar = sum(r.centrado is False for r in resultados)
    sin_medir = sum(r.centrado is None for r in resultados)
    return f"{len(resultados)} pruebas: {centradas} centradas, {sin_centrar} sin centrar, {sin_medir} sin poder medir"


# %%
