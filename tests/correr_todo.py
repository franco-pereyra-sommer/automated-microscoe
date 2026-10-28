# %%

"""
Corre todas las pruebas de esta carpeta (tests/test_*.py), sin necesidad de
pytest.

    python tests/correr_todo.py              # todas
    python tests/correr_todo.py impresora    # solo los archivos que contengan "impresora"
    python tests/correr_todo.py -v           # muestra también lo que imprimen las pruebas

Cada prueba es una función `test_...` que falla si lanza una excepción
(assert incluido). No hace falta pytest.

Creado el 2026-10-27.
"""

import contextlib
import importlib.util
import io
import sys
import time
import traceback
from pathlib import Path

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA))

from simuladores import Omitida  # noqa: E402


def cargar(ruta):
    spec = importlib.util.spec_from_file_location(ruta.stem, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def main(argumentos):
    verboso = "-v" in argumentos
    filtros = [a for a in argumentos if not a.startswith("-")]

    archivos = sorted(
        ruta for ruta in CARPETA.glob("test_*.py")
        if not filtros or any(f in ruta.stem for f in filtros)
    )

    ok = fallan = omitidas = 0
    t0 = time.time()

    for ruta in archivos:
        try:
            modulo = cargar(ruta)
        except Exception:
            fallan += 1
            print(f"FALLA  {ruta.name}: no se pudo importar\n{traceback.format_exc()}")
            continue

        for nombre in sorted(n for n in vars(modulo) if n.startswith("test_")):
            funcion = getattr(modulo, nombre)
            salida = io.StringIO()

            try:
                with contextlib.redirect_stdout(salida):
                    funcion()
            except Omitida as motivo:
                omitidas += 1
                print(f"omitida {ruta.stem}.{nombre}: {motivo}")
            except Exception:
                fallan += 1
                print(f"FALLA  {ruta.stem}.{nombre}\n{traceback.format_exc()}")
                if salida.getvalue():
                    print("  --- salida de la prueba:\n" + salida.getvalue()[-1500:])
            else:
                ok += 1
                print(f"ok     {ruta.stem}.{nombre}")
                if verboso and salida.getvalue():
                    print(salida.getvalue())

    print(f"\n{ok} ok, {fallan} fallan, {omitidas} omitidas  ({time.time() - t0:.1f} s)")
    return 1 if fallan else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


# %%
