# %%

"""
Pruebas que NO ejecutan los módulos base: leen su código.

Sirven para atrapar lo que Python recién avisa cuando la línea se ejecuta
(nombres que quedaron sin definir después de mover una función, imports que
rompen la regla de dependencias, nombres viejos que reaparecen) y para
avisar cuando el índice de API de PROYECTO_CONTEXTO.txt quedó desactualizado.

Creado el 2026-10-27.
"""

import ast
import builtins
import importlib.util
import re
import symtable
from pathlib import Path

from simuladores import RAIZ, omitir

MODULOS_BASE = ["camara_openmv", "impresora_cosmos", "metricas_imagen", "busqueda_muestra"]

# Módulos de UN solo instrumento (o de imagen pura): no importan nada del proyecto.
MODULOS_INDEPENDIENTES = ["camara_openmv", "impresora_cosmos", "metricas_imagen"]

_IMPLICITOS = {"__file__", "__name__", "__doc__", "__package__", "__spec__", "__builtins__", "__class__"}


def _codigo(nombre_modulo):
    return (RAIZ / f"{nombre_modulo}.py").read_text(encoding="utf-8")


def _todas_las_tablas(tabla):
    yield tabla
    for hija in tabla.get_children():
        yield from _todas_las_tablas(hija)


def nombres_no_definidos(codigo, nombre_archivo="<codigo>"):
    """
    Nombres que el código usa pero no están definidos en ningún lado (ni
    módulo, ni builtins, ni imports). Un pyflakes mínimo, con la librería estándar.
    """

    raiz = symtable.symtable(codigo, nombre_archivo, "exec")
    tablas = list(_todas_las_tablas(raiz))

    permitidos = set(dir(builtins)) | _IMPLICITOS

    for simbolo in raiz.get_symbols():
        if simbolo.is_assigned() or simbolo.is_imported() or simbolo.is_namespace():
            permitidos.add(simbolo.get_name())

    # Nombres creados con `global x` dentro de una función.
    for tabla in tablas[1:]:
        for simbolo in tabla.get_symbols():
            if simbolo.is_declared_global() and simbolo.is_assigned():
                permitidos.add(simbolo.get_name())

    faltan = set()

    for tabla in tablas:
        es_modulo = tabla.get_type() == "module"
        for simbolo in tabla.get_symbols():
            if not simbolo.is_referenced():
                continue
            if es_modulo or simbolo.is_global():
                if simbolo.get_name() not in permitidos:
                    faltan.add(simbolo.get_name())

    return faltan


def test_el_detector_de_nombres_no_definidos_funciona():
    assert nombres_no_definidos("def f():\n    return g()\n") == {"g"}
    assert nombres_no_definidos("import os\ndef f(x):\n    y = x\n    return os.sep + y\n") == set()
    assert nombres_no_definidos("def f():\n    return [k for k in range(3)] + sin_definir\n") == {"sin_definir"}


def test_ningun_modulo_base_usa_nombres_sin_definir():
    for modulo in MODULOS_BASE:
        faltan = nombres_no_definidos(_codigo(modulo), f"{modulo}.py")
        assert not faltan, f"{modulo}.py usa nombres sin definir: {sorted(faltan)}"


def test_modulos_independientes_no_importan_otros_del_proyecto():
    del_proyecto = {ruta.stem for ruta in RAIZ.glob("*.py")}

    for modulo in MODULOS_INDEPENDIENTES:
        arbol = ast.parse(_codigo(modulo))
        importados = set()

        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Import):
                importados |= {alias.name.split(".")[0] for alias in nodo.names}
            elif isinstance(nodo, ast.ImportFrom) and nodo.module:
                importados.add(nodo.module.split(".")[0])

        indebidos = (importados & del_proyecto) - {modulo}

        assert not indebidos, (
            f"{modulo}.py no debería importar módulos del proyecto: {sorted(indebidos)}"
        )


def test_ningun_base_importa_el_firmware():
    # main_openmv.py corre en la placa (usa csi y pyb): en la PC no se importa.
    for modulo in MODULOS_BASE:
        assert not re.search(r"^\s*(import|from)\s+main_openmv\b", _codigo(modulo), re.M), modulo


def test_los_base_no_llaman_a_los_nombres_viejos_de_desplazamiento():
    # desp_lin_abs -> desp_lin_000 y desp_lin_rel -> desp_lin_cam
    for modulo in MODULOS_BASE:
        viejos = re.findall(r"\bdesp_lin_(?:abs|rel)\s*\(", _codigo(modulo))
        assert not viejos, f"{modulo}.py todavía llama a {sorted(set(viejos))}"


def test_indice_de_api_al_dia():
    contexto = RAIZ / "PROYECTO_CONTEXTO.txt"
    generador = RAIZ / "herramientas" / "generar_indice.py"

    if not contexto.exists() or not generador.exists():
        omitir("todavía no existen PROYECTO_CONTEXTO.txt / herramientas/generar_indice.py")

    spec = importlib.util.spec_from_file_location("generar_indice", generador)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)

    esperado = modulo.sin_marca_de_tiempo(modulo.indice_api())
    actual = modulo.sin_marca_de_tiempo(
        modulo.leer_seccion(contexto.read_text(encoding="utf-8"), "INDICE_API")
    )

    assert esperado == actual, (
        "El índice de API de PROYECTO_CONTEXTO.txt está desactualizado: "
        "correr  python herramientas/generar_indice.py"
    )
