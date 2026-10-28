# %%

"""
Regenera las secciones AUTOGENERADAS de los .txt del proyecto, leyendo el
código (solo librería estándar):

  - PROYECTO_CONTEXTO.txt: índice de la API pública de los módulos base
    (clases, funciones, constantes), entre las marcas  INDICE_API.
  - BITACORA.txt: tabla "qué importa cada aplicación de los módulos base" y si
    esos nombres existen hoy, entre las marcas  TABLA_APPS.

Uso (desde la raíz del repo, con cualquier Python):

    python herramientas/generar_indice.py          # reescribe las dos secciones
    python herramientas/generar_indice.py --ver    # solo las imprime

Cuándo correrlo: después de tocar un módulo base (cambió el índice) y antes
de pasarle los .txt a un chat nuevo. tests/test_estatico.py avisa si el
índice quedó desactualizado.

Limitación de la tabla de aplicaciones: solo ve imports (`from X import a, b`,
`import X` + `X.a`). No ve nombres que lleguen por otro camino.

Creado el 2026-10-27.
"""

import ast
import datetime
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

MODULOS_BASE = ["camara_openmv", "impresora_cosmos", "metricas_imagen", "busqueda_muestra"]

# main_openmv.py es el firmware que corre en la placa OpenMV (usa csi y pyb):
# no se importa desde la PC y su protocolo se documenta a mano en PROYECTO_CONTEXTO.txt.
# metricas_enfoque y Main son módulos descontinuados (carpeta "Oudated code").
MODULOS_RASTREADOS = MODULOS_BASE + ["main_openmv", "metricas_enfoque", "Main"]

CARPETA_APP = re.compile(r"^\d\d\.\d\d\.\d\d ")

ARCHIVO_CONTEXTO = RAIZ / "PROYECTO_CONTEXTO.txt"
ARCHIVO_BITACORA = RAIZ / "BITACORA.txt"


# ----------------------------------------------------------------------
# Texto
# ----------------------------------------------------------------------

# Un punto que cierra una frase: no cuenta el de "X:.." ni el de "etc.." (punto precedido por otro).
_FIN_DE_FRASE = re.compile(r"(?<=[^.\s])\.(?:\s|$)")


def _resumen(docstring, largo=110):
    """Primera frase del docstring, en una línea (hasta 3 líneas si la frase sigue)."""

    if not docstring:
        return ""

    primer_parrafo = docstring.strip().split("\n\n")[0]
    lineas = []

    for linea in primer_parrafo.splitlines()[:3]:
        lineas.append(linea.strip())
        if lineas[-1].endswith((".", ":")):
            break

    texto = " ".join(lineas).rstrip(":").strip()

    fin = _FIN_DE_FRASE.search(texto)
    if fin and fin.start() < largo:
        texto = texto[: fin.start() + 1]

    if len(texto) > largo:
        texto = texto[: largo - 3].rstrip() + "..."

    return texto


def _parametros(argumentos, maximo=6):
    nombres = [a.arg for a in argumentos.posonlyargs + argumentos.args]

    if nombres and nombres[0] in ("self", "cls"):
        nombres = nombres[1:]

    if argumentos.vararg:
        nombres.append("*" + argumentos.vararg.arg)

    nombres += [a.arg for a in argumentos.kwonlyargs]

    if argumentos.kwarg:
        nombres.append("**" + argumentos.kwarg.arg)

    if len(nombres) > maximo:
        return ", ".join(nombres[:maximo]) + f", ... (+{len(nombres) - maximo})"

    return ", ".join(nombres)


def _con_resumen(encabezado, docstring, largo=110):
    resumen = _resumen(docstring, largo)
    return f"{encabezado} -- {resumen}" if resumen else encabezado


def _literal(nodo):
    try:
        return ast.literal_eval(nodo)
    except (ValueError, SyntaxError):
        return None


def _constantes_de_modulo(arbol):
    """NOMBRE=valor para las constantes en MAYÚSCULAS (solo el nombre si el valor no es corto)."""

    constantes = []

    for nodo in arbol.body:
        if isinstance(nodo, ast.Assign):
            for destino in nodo.targets:
                if isinstance(destino, ast.Name) and destino.id.isupper():
                    valor = _literal(nodo.value)
                    texto = repr(valor) if valor is not None else ""
                    constantes.append(
                        f"{destino.id}={texto}" if texto and len(texto) <= 60 else destino.id
                    )

    return constantes


def _atributos_constantes(clase):
    """self.x = <número> en el __init__ (las características fijas del hardware)."""

    for metodo in clase.body:
        if isinstance(metodo, ast.FunctionDef) and metodo.name == "__init__":
            atributos = []
            for nodo in ast.walk(metodo):
                if isinstance(nodo, ast.Assign):
                    for destino in nodo.targets:
                        if (isinstance(destino, ast.Attribute)
                                and isinstance(destino.value, ast.Name)
                                and destino.value.id == "self"):
                            valor = _literal(nodo.value)
                            if isinstance(valor, (int, float)) and not isinstance(valor, bool):
                                atributos.append(f"{destino.attr}={valor}")
            return atributos

    return []


def _es_enum(clase):
    return any(
        (isinstance(base, ast.Attribute) and base.attr == "Enum")
        or (isinstance(base, ast.Name) and base.id == "Enum")
        for base in clase.bases
    )


def _indice_de_un_modulo(ruta):
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    lineas = [_con_resumen(ruta.name, ast.get_docstring(arbol), largo=160)]

    constantes = _constantes_de_modulo(arbol)
    if constantes:
        lineas.append("  Constantes: " + "; ".join(constantes))

    for nodo in arbol.body:

        if isinstance(nodo, ast.ClassDef) and not nodo.name.startswith("_"):

            if _es_enum(nodo):
                miembros = [t.id for n in nodo.body if isinstance(n, ast.Assign)
                            for t in n.targets if isinstance(t, ast.Name)]
                lineas.append(f"  {nodo.name} (Enum): " + ", ".join(miembros))
                continue

            constructor = next((m for m in nodo.body if isinstance(m, ast.FunctionDef)
                                and m.name == "__init__"), None)
            parametros = _parametros(constructor.args) if constructor else ""
            lineas.append(_con_resumen(f"  class {nodo.name}({parametros})", ast.get_docstring(nodo)))

            atributos = _atributos_constantes(nodo)
            if atributos:
                lineas.append("    Atributos fijos: " + ", ".join(atributos))

            for metodo in nodo.body:
                if isinstance(metodo, ast.FunctionDef) and not metodo.name.startswith("_"):
                    lineas.append(_con_resumen(
                        f"    .{metodo.name}({_parametros(metodo.args)})",
                        ast.get_docstring(metodo),
                    ))

        elif isinstance(nodo, ast.FunctionDef) and not nodo.name.startswith("_"):
            lineas.append(_con_resumen(
                f"  {nodo.name}({_parametros(nodo.args)})", ast.get_docstring(nodo)
            ))

    return "\n".join(lineas)


def indice_api():
    """Índice de la API pública de los módulos base (texto)."""

    partes = [f"Generado: {datetime.datetime.now():%Y-%m-%d %H:%M} (herramientas/generar_indice.py)", ""]

    for modulo in MODULOS_BASE:
        partes.append(_indice_de_un_modulo(RAIZ / f"{modulo}.py"))
        partes.append("")

    return "\n".join(partes).rstrip("\n") + "\n"


# ----------------------------------------------------------------------
# Tabla de aplicaciones
# ----------------------------------------------------------------------

def _simbolos_de_modulo(nombre):
    """(definidos, importados) de un módulo de la raíz, o None si no existe ahí."""

    ruta = RAIZ / f"{nombre}.py"
    if not ruta.exists():
        return None

    definidos, importados = set(), set()

    for nodo in ast.parse(ruta.read_text(encoding="utf-8")).body:
        if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definidos.add(nodo.name)
        elif isinstance(nodo, (ast.Assign, ast.AnnAssign)):
            destinos = nodo.targets if isinstance(nodo, ast.Assign) else [nodo.target]
            for destino in destinos:
                definidos |= {n.id for n in ast.walk(destino) if isinstance(n, ast.Name)}
        elif isinstance(nodo, (ast.Import, ast.ImportFrom)):
            importados |= {(a.asname or a.name).split(".")[0] for a in nodo.names}

    return definidos, importados


def _usos_en_la_app(carpeta):
    """({modulo_base: {nombres}}, [archivos que no se pudieron leer])."""

    usos, ilegibles = {}, []

    for ruta in sorted(carpeta.rglob("*.py")):
        if "__pycache__" in ruta.parts:
            continue

        try:
            arbol = ast.parse(ruta.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            ilegibles.append(ruta.name)
            continue

        alias_de_modulo = {}

        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.ImportFrom) and nodo.module and nodo.level == 0:
                raiz_modulo = nodo.module.split(".")[0]
                if raiz_modulo in MODULOS_RASTREADOS:
                    usos.setdefault(raiz_modulo, set()).update(a.name for a in nodo.names)
            elif isinstance(nodo, ast.Import):
                for alias in nodo.names:
                    raiz_modulo = alias.name.split(".")[0]
                    if raiz_modulo in MODULOS_RASTREADOS:
                        alias_de_modulo[alias.asname or alias.name] = raiz_modulo
                        usos.setdefault(raiz_modulo, set())

        if alias_de_modulo:
            for nodo in ast.walk(arbol):
                if (isinstance(nodo, ast.Attribute) and isinstance(nodo.value, ast.Name)
                        and nodo.value.id in alias_de_modulo):
                    usos[alias_de_modulo[nodo.value.id]].add(nodo.attr)

    return usos, ilegibles


def tabla_apps():
    """Qué módulos base importa cada aplicación y qué nombres ya no existen (texto)."""

    simbolos = {m: _simbolos_de_modulo(m) for m in MODULOS_RASTREADOS}

    lineas = [
        f"Generado: {datetime.datetime.now():%Y-%m-%d %H:%M} (herramientas/generar_indice.py)",
        "Leyenda:  nombre~        = funciona solo porque ese módulo lo importa de otro (la definición está en otro lado);",
        "          FALTA          = el nombre ya no existe en el módulo base vigente: la app falla al importarlo;",
        "          DESCONTINUADO  = el módulo ya no está en la raíz (Main.py, metricas_enfoque.py: ver 'Oudated code').",
        "Las apps se actualizan solo si Franco lo pide (ver REGLAS_CODIGO.txt).",
        "",
    ]

    carpetas = sorted(c for c in RAIZ.iterdir() if c.is_dir() and CARPETA_APP.match(c.name))
    con_faltantes = con_descontinuados = 0

    for carpeta in carpetas:
        usos, ilegibles = _usos_en_la_app(carpeta)
        lineas.append(carpeta.name)

        if not usos:
            lineas.append("  (no importa módulos base)")

        faltantes_app = descontinuados_app = 0

        for modulo in MODULOS_RASTREADOS:
            if modulo not in usos:
                continue

            nombres = sorted(usos[modulo])
            actual = simbolos[modulo]

            if (carpeta / f"{modulo}.py").exists():
                lineas.append(f"  {modulo}: usa la COPIA LOCAL de la carpeta ({', '.join(nombres) or 'import completo'})")
                continue

            if actual is None:
                lineas.append(
                    f"  {modulo}: {', '.join(nombres) or '(import completo)'}"
                    f"  [DESCONTINUADO: {modulo}.py ya no está en la raíz]"
                )
                descontinuados_app += 1
                continue

            definidos, importados = actual
            marcados, faltan = [], []

            for nombre in nombres:
                if nombre == "*" or nombre in definidos:
                    marcados.append(nombre)
                elif nombre in importados:
                    marcados.append(nombre + "~")
                else:
                    marcados.append(nombre)
                    faltan.append(nombre)

            lineas.append(f"  {modulo}: {', '.join(marcados) or '(import completo)'}")

            if faltan:
                lineas.append(f"    FALTA en {modulo}: {', '.join(faltan)}")
                faltantes_app += len(faltan)

        if ilegibles:
            lineas.append(f"  (no se pudieron leer: {', '.join(ilegibles)})")

        con_faltantes += bool(faltantes_app)
        con_descontinuados += bool(descontinuados_app)
        lineas.append("")

    lineas.append(
        f"Resumen: de {len(carpetas)} carpetas, {con_descontinuados} importan módulos descontinuados y "
        f"{con_faltantes} importan nombres que ya no existen en los módulos base vigentes."
    )

    return "\n".join(lineas) + "\n"


# ----------------------------------------------------------------------
# Secciones marcadas dentro de los .txt
# ----------------------------------------------------------------------

def _patron(marca):
    return re.compile(
        rf"(^#<<< {marca}\b[^\n]*>>>\n)(.*?)(^#<<< FIN_{marca} >>>)",
        re.DOTALL | re.MULTILINE,
    )


def leer_seccion(texto, marca):
    encontrado = _patron(marca).search(texto)
    if not encontrado:
        raise ValueError(f"No encontré las marcas #<<< {marca} ... #<<< FIN_{marca} >>> en el texto")
    return encontrado.group(2)


def reemplazar_seccion(texto, marca, contenido):
    if not _patron(marca).search(texto):
        raise ValueError(f"No encontré las marcas #<<< {marca} ... #<<< FIN_{marca} >>> en el texto")
    return _patron(marca).sub(lambda m: m.group(1) + contenido + m.group(3), texto, count=1)


def sin_marca_de_tiempo(texto):
    return "\n".join(l for l in texto.splitlines() if not l.startswith("Generado:"))


def _reescribir(archivo, marca, contenido):
    with open(archivo, encoding="utf-8", newline="") as f:
        texto = f.read()
    with open(archivo, "w", encoding="utf-8", newline="") as f:
        f.write(reemplazar_seccion(texto, marca, contenido))


def main(argumentos):
    indice, tabla = indice_api(), tabla_apps()

    if "--ver" in argumentos:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print("=== INDICE_API ===\n" + indice + "\n=== TABLA_APPS ===\n" + tabla)
        return 0

    _reescribir(ARCHIVO_CONTEXTO, "INDICE_API", indice)
    _reescribir(ARCHIVO_BITACORA, "TABLA_APPS", tabla)
    print(f"Actualizado: {ARCHIVO_CONTEXTO.name} (INDICE_API) y {ARCHIVO_BITACORA.name} (TABLA_APPS)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


# %%
