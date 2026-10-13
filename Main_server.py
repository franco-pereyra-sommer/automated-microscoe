#%%



from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from html import escape
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit
import hashlib
import json
import mimetypes
import shutil
import threading


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

CARPETA_FOTOS = Path(
    # r"C:\Users\Mateo\Desktop\Labo 7\Fotos app"
    # r"C:\BackUp\Facultad\Labo 7\Fotos\26.09.03 fotos sv"
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.09.03 fotos sv\ desplazamiento 3"
)

HOST = "0.0.0.0"
PUERTO = 8000

EXTENSIONES = {
    ".jpg",
    ".jpeg",
    ".png"
}

INTERVALO_ACTUALIZACION_MS = 2000

BLOQUEO_ESTADO = threading.Lock()
DETENER_ORGANIZADOR = threading.Event()
CARPETA_ACTIVA = ""


# ==========================================================
# ESTADO Y ARCHIVOS
# ==========================================================

def archivo_estado():
    return CARPETA_FOTOS / "estado_servidor_fotos.json"


def obtener_carpeta_activa():
    with BLOQUEO_ESTADO:
        return CARPETA_ACTIVA


def guardar_carpeta_activa(nombre):
    global CARPETA_ACTIVA

    with BLOQUEO_ESTADO:
        CARPETA_ACTIVA = nombre

        temporal = (
            CARPETA_FOTOS
            / "estado_servidor_fotos.tmp"
        )

        temporal.write_text(
            json.dumps(
                {
                    "carpeta_activa": nombre
                },
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        temporal.replace(
            archivo_estado()
        )


def resolver_carpeta(nombre):
    try:
        raiz = CARPETA_FOTOS.resolve()

        if nombre == "":
            return raiz

        carpeta = (
            CARPETA_FOTOS / nombre
        ).resolve()

        if carpeta.parent != raiz:
            return None

        if not carpeta.is_dir():
            return None

        return carpeta

    except OSError:
        return None


def cargar_carpeta_activa():
    global CARPETA_ACTIVA

    try:
        datos = json.loads(
            archivo_estado().read_text(
                encoding="utf-8"
            )
        )

        nombre = datos.get(
            "carpeta_activa",
            ""
        )

        if not isinstance(nombre, str):
            nombre = ""

    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError
    ):
        nombre = ""

    if nombre and resolver_carpeta(nombre) is None:
        nombre = ""

    with BLOQUEO_ESTADO:
        CARPETA_ACTIVA = nombre


def validar_nombre_carpeta(nombre):
    if not nombre:
        return (
            False,
            "Escribí un nombre para la carpeta."
        )

    if len(nombre) > 80:
        return (
            False,
            "El nombre es demasiado largo."
        )

    caracteres_invalidos = '<>:"/\\|?*'

    if any(
        caracter in nombre
        for caracter in caracteres_invalidos
    ):
        return (
            False,
            "El nombre contiene caracteres no permitidos."
        )

    if any(
        ord(caracter) < 32
        for caracter in nombre
    ):
        return (
            False,
            "El nombre contiene caracteres no permitidos."
        )

    if (
        nombre in {".", ".."}
        or nombre.startswith(".")
    ):
        return (
            False,
            "Ese nombre no es válido."
        )

    if nombre.endswith((".", " ")):
        return (
            False,
            "El nombre no puede terminar con punto o espacio."
        )

    reservados = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9"
    }

    nombre_base = (
        nombre
        .split(".", 1)[0]
        .upper()
    )

    if nombre_base in reservados:
        return (
            False,
            "Ese nombre está reservado por Windows."
        )

    return True, ""


def listar_carpetas():
    try:
        carpetas = [
            ruta
            for ruta in CARPETA_FOTOS.iterdir()
            if ruta.is_dir()
            and not ruta.name.startswith(".")
        ]

    except OSError:
        return []

    return sorted(
        carpetas,
        key=lambda ruta: ruta.name.casefold()
    )


def listar_fotos(carpeta):
    if carpeta is None:
        return []

    resultado = []

    try:
        for ruta in carpeta.iterdir():

            if not ruta.is_file():
                continue

            if ruta.suffix.lower() not in EXTENSIONES:
                continue

            try:
                fecha = ruta.stat().st_mtime_ns

            except OSError:
                continue

            resultado.append(
                (
                    fecha,
                    ruta.name.casefold(),
                    ruta
                )
            )

    except OSError:
        return []

    resultado.sort(
        reverse=True
    )

    return [
        elemento[2]
        for elemento in resultado
    ]


def resolver_foto(
    nombre_carpeta,
    nombre_foto
):
    carpeta = resolver_carpeta(
        nombre_carpeta
    )

    if carpeta is None:
        return None, 404

    try:
        foto = (
            carpeta / nombre_foto
        ).resolve()

        if foto.parent != carpeta.resolve():
            return None, 403

    except OSError:
        return None, 400

    if not foto.is_file():
        return None, 404

    if foto.suffix.lower() not in EXTENSIONES:
        return None, 403

    return foto, 200


def firma_carpeta(nombre_carpeta):
    datos = []

    carpeta = resolver_carpeta(
        nombre_carpeta
    )

    for foto in listar_fotos(carpeta):

        try:
            estado = foto.stat()

            datos.append(
                (
                    foto.name,
                    estado.st_size,
                    estado.st_mtime_ns
                )
            )

        except OSError:
            continue

    texto = json.dumps(
        datos,
        ensure_ascii=False,
        separators=(",", ":")
    )

    return hashlib.sha256(
        texto.encode("utf-8")
    ).hexdigest()


def firma_menu():
    datos = {
        "activa": obtener_carpeta_activa(),
        "carpetas": [
            (
                "",
                firma_carpeta("")
            )
        ]
    }

    for carpeta in listar_carpetas():
        datos["carpetas"].append(
            (
                carpeta.name,
                firma_carpeta(
                    carpeta.name
                )
            )
        )

    texto = json.dumps(
        datos,
        ensure_ascii=False,
        separators=(",", ":")
    )

    return hashlib.sha256(
        texto.encode("utf-8")
    ).hexdigest()


def destino_disponible(
    carpeta,
    nombre
):
    destino = carpeta / nombre

    if not destino.exists():
        return destino

    base = Path(nombre).stem
    extension = Path(nombre).suffix
    numero = 1

    while True:
        destino = (
            carpeta
            / f"{base}_{numero}{extension}"
        )

        if not destino.exists():
            return destino

        numero += 1


# ==========================================================
# MOVER FOTOS NUEVAS A LA CARPETA ACTIVA
# ==========================================================

def firma_archivo(ruta):
    estado = ruta.stat()

    return (
        estado.st_size,
        estado.st_mtime_ns
    )


def organizar_fotos_nuevas():
    procesadas = {}
    pendientes = {}

    # Las fotos que ya existen al iniciar
    # quedan donde estaban.
    for foto in listar_fotos(
        CARPETA_FOTOS
    ):
        try:
            procesadas[foto.name] = (
                firma_archivo(foto)
            )

        except OSError:
            pass

    while not DETENER_ORGANIZADOR.wait(
        0.75
    ):
        actuales = {}

        for foto in listar_fotos(
            CARPETA_FOTOS
        ):
            try:
                firma = firma_archivo(
                    foto
                )

            except OSError:
                continue

            actuales[foto.name] = firma

            if (
                procesadas.get(foto.name)
                == firma
            ):
                pendientes.pop(
                    foto.name,
                    None
                )
                continue

            firma_previa, repeticiones = (
                pendientes.get(
                    foto.name,
                    (None, 0)
                )
            )

            if firma_previa == firma:
                repeticiones += 1

            else:
                repeticiones = 0

            pendientes[foto.name] = (
                firma,
                repeticiones
            )

            # Espera hasta que el archivo
            # termine de copiarse.
            if repeticiones < 2:
                continue

            activa = (
                obtener_carpeta_activa()
            )

            if activa == "":
                procesadas[foto.name] = firma

                pendientes.pop(
                    foto.name,
                    None
                )

                continue

            carpeta_destino = (
                resolver_carpeta(
                    activa
                )
            )

            if carpeta_destino is None:
                guardar_carpeta_activa(
                    ""
                )

                procesadas[foto.name] = firma

                pendientes.pop(
                    foto.name,
                    None
                )

                continue

            destino = destino_disponible(
                carpeta_destino,
                foto.name
            )

            try:
                shutil.move(
                    str(foto),
                    str(destino)
                )

                print(
                    f"Foto guardada en "
                    f"'{activa}': "
                    f"{destino.name}"
                )

                procesadas.pop(
                    foto.name,
                    None
                )

                pendientes.pop(
                    foto.name,
                    None
                )

            except (
                OSError,
                shutil.Error
            ):
                # Si sigue ocupado,
                # vuelve a intentarlo.
                continue

        for nombre in list(
            procesadas
        ):
            if nombre not in actuales:
                procesadas.pop(
                    nombre,
                    None
                )

        for nombre in list(
            pendientes
        ):
            if nombre not in actuales:
                pendientes.pop(
                    nombre,
                    None
                )


# ==========================================================
# APARIENCIA DE LAS PÁGINAS
# ==========================================================

ESTILO = """
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    padding: 16px;
    font-family: Arial, sans-serif;
    background: #f3f5f6;
    color: #222;
}

.contenedor {
    width: 100%;
    max-width: 1000px;
    margin: 0 auto;
}

h1 {
    margin: 0 0 6px;
    font-size: 25px;
    overflow-wrap: anywhere;
}

.subtitulo,
.contador {
    margin: 0 0 15px;
    color: #6c6c6c;
}

.volver {
    display: inline-block;
    margin-bottom: 12px;
    color: #055e6b;
    font-weight: bold;
    text-decoration: none;
}

.destino {
    margin-bottom: 14px;
    padding: 13px;
    border-left: 5px solid #055e6b;
    border-radius: 8px;
    background: #e3f3f5;
}

.destino strong {
    color: #055e6b;
}

.aviso {
    margin-bottom: 13px;
    padding: 11px;
    border-radius: 8px;
}

.correcto {
    background: #dff4e6;
    color: #176b35;
}

.error {
    background: #fde1e1;
    color: #9b2424;
}

.crear {
    display: flex;
    gap: 8px;
    margin-bottom: 19px;
}

.crear input {
    min-width: 0;
    flex: 1;
    padding: 12px;
    border: 1px solid #bbb;
    border-radius: 8px;
    font-size: 16px;
}

button {
    border: 0;
    border-radius: 8px;
    font-size: 15px;
    font-weight: bold;
    cursor: pointer;
}

.crear button {
    padding: 0 15px;
    background: #055e6b;
    color: white;
}

.lista-carpetas {
    display: grid;

    grid-template-columns:
        repeat(
            auto-fill,
            minmax(220px, 1fr)
        );

    gap: 12px;
}

.carpeta {
    overflow: hidden;
    border: 2px solid transparent;
    border-radius: 11px;
    background: white;

    box-shadow:
        0 2px 8px
        rgba(0, 0, 0, 0.12);
}

.carpeta.activa {
    border-color: #055e6b;
}

.abrir-carpeta {
    display: flex;
    align-items: center;
    gap: 12px;
    min-height: 88px;
    padding: 14px;
    color: inherit;
    text-decoration: none;
}

.icono-carpeta {
    font-size: 38px;
}

.datos-carpeta {
    min-width: 0;
}

.datos-carpeta h2 {
    margin: 0 0 4px;
    overflow: hidden;
    font-size: 18px;
    white-space: nowrap;
    text-overflow: ellipsis;
}

.datos-carpeta p {
    margin: 0;
    color: #777;
}

.carpeta form {
    padding: 0 12px 12px;
}

.seleccionar {
    width: 100%;
    padding: 10px;
    background: #e4ecee;
    color: #055e6b;
}

.seleccionado {
    background: #055e6b;
    color: white;
    cursor: default;
}

.encabezado {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 9px;
}

.etiqueta {
    padding: 5px 9px;
    border-radius: 999px;
    background: #d9f1f4;
    color: #055e6b;
    font-size: 13px;
    font-weight: bold;
}

.galeria {
    display: grid;

    grid-template-columns:
        repeat(
            auto-fill,
            minmax(145px, 1fr)
        );

    gap: 12px;
}

.foto {
    overflow: hidden;
    border-radius: 10px;
    background: white;

    box-shadow:
        0 2px 8px
        rgba(0, 0, 0, 0.14);
}

.foto a {
    display: block;
}

.foto img {
    display: block;
    width: 100%;
    aspect-ratio: 4 / 3;
    object-fit: cover;
}

.foto p {
    margin: 0;
    padding: 9px;
    overflow: hidden;
    font-size: 13px;
    white-space: nowrap;
    text-overflow: ellipsis;
}

.vacia {
    grid-column: 1 / -1;
    padding: 28px;
    border-radius: 10px;
    background: white;
    color: #777;
    text-align: center;
}

@media (max-width: 520px) {
    .crear {
        flex-direction: column;
    }

    .crear button {
        min-height: 44px;
    }

    .lista-carpetas {
        grid-template-columns: 1fr;
    }
}
"""


ESTILO_VISOR = """
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    min-height: 100vh;
    font-family: Arial, sans-serif;
    background: #111;
    color: white;
}

.cerrar,
.flecha {
    position: fixed;
    z-index: 20;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.72);
    color: white;
    text-decoration: none;
}

.cerrar {
    top: 15px;
    right: 15px;
    width: 48px;
    height: 48px;
    border-radius: 50%;
    font-size: 36px;
}

.flecha {
    top: 50%;
    width: 46px;
    height: 68px;
    border-radius: 10px;
    font-size: 40px;
    font-weight: bold;

    transform:
        translateY(-50%);
}

.anterior {
    left: 7px;
}

.siguiente {
    right: 7px;
}

.visor {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    padding: 70px 56px 28px;
}

.visor img {
    display: block;
    max-width: 100%;
    max-height: calc(100vh - 150px);
    object-fit: contain;
    border-radius: 6px;
}

.nombre {
    margin: 12px 0 4px;
    color: #ddd;
    text-align: center;
    overflow-wrap: anywhere;
}

.posicion {
    margin: 0;
    color: #999;
    font-size: 14px;
}

@media (max-width: 500px) {
    .flecha {
        width: 41px;
        height: 58px;
        font-size: 33px;
    }

    .visor {
        padding-left: 48px;
        padding-right: 48px;
    }
}
"""


def script_actualizacion(
    firma_inicial,
    url_estado
):
    script = """
    <script>
        const firmaInicial = "__FIRMA__";

        async function comprobarCambios() {
            try {
                const respuesta = await fetch(
                    "__URL__" + Date.now(),
                    {
                        cache: "no-store"
                    }
                );

                if (!respuesta.ok) {
                    return;
                }

                const datos =
                    await respuesta.json();

                if (
                    datos.firma
                    !== firmaInicial
                ) {
                    window.location.reload();
                }

            } catch (error) {
            }
        }

        setInterval(
            comprobarCambios,
            __INTERVALO__
        );
    </script>
    """

    return (
        script
        .replace(
            "__FIRMA__",
            firma_inicial
        )
        .replace(
            "__URL__",
            url_estado
        )
        .replace(
            "__INTERVALO__",
            str(
                INTERVALO_ACTUALIZACION_MS
            )
        )
    )


# ==========================================================
# SERVIDOR HTTP
# ==========================================================

class ServidorFotos(
    BaseHTTPRequestHandler
):

    def enviar(
        self,
        contenido,
        tipo="text/html; charset=utf-8",
        codigo=200
    ):
        if isinstance(contenido, str):
            contenido = contenido.encode(
                "utf-8"
            )

        self.send_response(
            codigo
        )

        self.send_header(
            "Content-Type",
            tipo
        )

        self.send_header(
            "Content-Length",
            str(len(contenido))
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )

        self.send_header(
            "Cache-Control",
            "no-store"
        )

        self.end_headers()

        self.wfile.write(
            contenido
        )

    def redirigir(
        self,
        destino
    ):
        self.send_response(
            303
        )

        self.send_header(
            "Location",
            destino
        )

        self.send_header(
            "Cache-Control",
            "no-store"
        )

        self.end_headers()

    def log_message(
        self,
        formato,
        *argumentos
    ):
        # Evita llenar la consola
        # con las consultas automáticas.
        if not self.path.startswith(
            "/estado"
        ):
            super().log_message(
                formato,
                *argumentos
            )

    # ======================================================
    # PETICIONES POST
    # ======================================================

    def do_POST(self):
        ruta = unquote(
            urlsplit(
                self.path
            ).path
        )

        try:
            longitud = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )

        except ValueError:
            longitud = 0

        if longitud > 10000:
            self.enviar(
                "Petición demasiado grande",
                codigo=413
            )
            return

        cuerpo = self.rfile.read(
            longitud
        ).decode(
            "utf-8",
            errors="replace"
        )

        datos = parse_qs(
            cuerpo,
            keep_blank_values=True
        )

        # ----------------------------------------------
        # CREAR CARPETA
        # ----------------------------------------------

        if ruta == "/crear-carpeta":

            nombre = datos.get(
                "nombre",
                [""]
            )[0].strip()

            valido, error = (
                validar_nombre_carpeta(
                    nombre
                )
            )

            if not valido:
                self.redirigir(
                    "/?error="
                    + quote(
                        error,
                        safe=""
                    )
                )
                return

            carpeta = (
                CARPETA_FOTOS
                / nombre
            )

            try:
                carpeta.mkdir(
                    exist_ok=False
                )

            except FileExistsError:
                if not carpeta.is_dir():
                    self.redirigir(
                        "/?error="
                        + quote(
                            "Ya existe un archivo con ese nombre.",
                            safe=""
                        )
                    )
                    return

            except OSError:
                self.redirigir(
                    "/?error="
                    + quote(
                        "No se pudo crear la carpeta.",
                        safe=""
                    )
                )
                return

            guardar_carpeta_activa(
                nombre
            )

            mensaje = quote(
                f"Carpeta activa: {nombre}",
                safe=""
            )

            self.redirigir(
                "/?mensaje="
                + mensaje
            )
            return

        # ----------------------------------------------
        # SELECCIONAR CARPETA
        # ----------------------------------------------

        if ruta == "/seleccionar-carpeta":

            nombre = datos.get(
                "carpeta",
                [""]
            )[0].strip()

            if (
                nombre
                and resolver_carpeta(nombre)
                is None
            ):
                self.redirigir(
                    "/?error="
                    + quote(
                        "La carpeta no existe.",
                        safe=""
                    )
                )
                return

            guardar_carpeta_activa(
                nombre
            )

            visible = (
                nombre
                or "Sin clasificar"
            )

            self.redirigir(
                "/?mensaje="
                + quote(
                    f"Carpeta activa: {visible}",
                    safe=""
                )
            )
            return

        self.enviar(
            "Ruta no encontrada",
            codigo=404
        )

    # ======================================================
    # PETICIONES GET
    # ======================================================

    def do_GET(self):
        direccion = urlsplit(
            self.path
        )

        ruta = unquote(
            direccion.path
        )

        datos = parse_qs(
            direccion.query,
            keep_blank_values=True
        )

        if ruta == "/":
            self.mostrar_menu(
                datos
            )

        elif ruta == "/galeria":
            self.mostrar_galeria(
                datos.get(
                    "carpeta",
                    [""]
                )[0]
            )

        elif ruta == "/ver":
            self.mostrar_visor(
                datos.get(
                    "carpeta",
                    [""]
                )[0],
                datos.get(
                    "foto",
                    [""]
                )[0]
            )

        elif ruta == "/imagen":
            self.enviar_imagen(
                datos.get(
                    "carpeta",
                    [""]
                )[0],
                datos.get(
                    "foto",
                    [""]
                )[0]
            )

        elif ruta == "/fotos":
            self.enviar_lista_fotos(
                datos.get(
                    "carpeta",
                    [""]
                )[0]
            )

        elif ruta == "/estado":
            self.enviar_estado(
                datos
            )

        # Compatibilidad con
        # la dirección anterior.
        elif ruta.startswith("/foto/"):
            nombre_foto = ruta[
                len("/foto/"):
            ]

            self.enviar_imagen(
                "",
                nombre_foto
            )

        else:
            self.enviar(
                "Ruta no encontrada",
                codigo=404
            )

    # ======================================================
    # TARJETA DE CARPETA
    # ======================================================

    def tarjeta_carpeta(
        self,
        nombre,
        cantidad
    ):
        activa = (
            nombre
            == obtener_carpeta_activa()
        )

        titulo = (
            nombre
            or "Sin clasificar"
        )

        nombre_html = escape(
            nombre,
            quote=True
        )

        nombre_url = quote(
            nombre,
            safe=""
        )

        if activa:
            boton = """
            <button
                class="seleccionar seleccionado"
                disabled
            >
                Carpeta activa
            </button>
            """

        else:
            boton = f"""
            <form
                method="post"
                action="/seleccionar-carpeta"
            >
                <input
                    type="hidden"
                    name="carpeta"
                    value="{nombre_html}"
                >

                <button
                    class="seleccionar"
                    type="submit"
                >
                    Guardar aquí
                </button>
            </form>
            """

        clase = (
            "carpeta activa"
            if activa
            else "carpeta"
        )

        return f"""
        <article class="{clase}">

            <a
                class="abrir-carpeta"
                href="/galeria?carpeta={nombre_url}"
            >
                <div class="icono-carpeta">
                    &#128193;
                </div>

                <div class="datos-carpeta">
                    <h2>
                        {escape(titulo)}
                    </h2>

                    <p>
                        {cantidad} fotos
                    </p>
                </div>
            </a>

            {boton}

        </article>
        """

    # ======================================================
    # MENÚ PRINCIPAL
    # ======================================================

    def mostrar_menu(
        self,
        datos
    ):
        mensaje = datos.get(
            "mensaje",
            [""]
        )[0]

        error = datos.get(
            "error",
            [""]
        )[0]

        aviso = ""

        if mensaje:
            aviso = (
                '<div class="aviso correcto">'
                + escape(mensaje)
                + "</div>"
            )

        elif error:
            aviso = (
                '<div class="aviso error">'
                + escape(error)
                + "</div>"
            )

        activa = (
            obtener_carpeta_activa()
        )

        activa_visible = escape(
            activa
            or "Sin clasificar"
        )

        tarjetas = (
            self.tarjeta_carpeta(
                "",
                len(
                    listar_fotos(
                        CARPETA_FOTOS
                    )
                )
            )
        )

        for carpeta in listar_carpetas():
            tarjetas += (
                self.tarjeta_carpeta(
                    carpeta.name,
                    len(
                        listar_fotos(
                            carpeta
                        )
                    )
                )
            )

        actualizacion = (
            script_actualizacion(
                firma_menu(),
                "/estado?vista=menu&_="
            )
        )

        html = f"""
        <!DOCTYPE html>

        <html lang="es">

        <head>
            <meta charset="UTF-8">

            <meta
                name="viewport"
                content="width=device-width, initial-scale=1.0"
            >

            <title>
                Fotos OpenMV
            </title>

            <style>
                {ESTILO}
            </style>
        </head>

        <body>

            <main class="contenedor">

                <h1>
                    Fotos OpenMV
                </h1>

                <p class="subtitulo">
                    Creá una carpeta o elegí
                    dónde guardar las próximas fotos.
                </p>

                <div class="destino">
                    Las próximas fotos se guardarán en:

                    <strong>
                        {activa_visible}
                    </strong>
                </div>

                {aviso}

                <form
                    class="crear"
                    method="post"
                    action="/crear-carpeta"
                >
                    <input
                        type="text"
                        name="nombre"
                        maxlength="80"
                        placeholder="Nombre de la nueva carpeta"
                        autocomplete="off"
                        required
                    >

                    <button type="submit">
                        Crear y seleccionar
                    </button>
                </form>

                <section class="lista-carpetas">
                    {tarjetas}
                </section>

            </main>

            {actualizacion}

        </body>

        </html>
        """

        self.enviar(
            html
        )

    # ======================================================
    # GALERÍA
    # ======================================================

    def mostrar_galeria(
        self,
        nombre_carpeta
    ):
        carpeta = resolver_carpeta(
            nombre_carpeta
        )

        if carpeta is None:
            self.redirigir(
                "/?error="
                + quote(
                    "La carpeta no existe.",
                    safe=""
                )
            )
            return

        fotos = listar_fotos(
            carpeta
        )

        titulo = (
            nombre_carpeta
            or "Sin clasificar"
        )

        carpeta_url = quote(
            nombre_carpeta,
            safe=""
        )

        tarjetas = ""

        for foto in fotos:
            visible = escape(
                foto.name,
                quote=True
            )

            foto_url = quote(
                foto.name,
                safe=""
            )

            tarjetas += f"""
            <article class="foto">

                <a
                    href="/ver?carpeta={carpeta_url}&amp;foto={foto_url}"
                >
                    <img
                        src="/imagen?carpeta={carpeta_url}&amp;foto={foto_url}"
                        alt="{visible}"
                        loading="lazy"
                    >
                </a>

                <p title="{visible}">
                    {visible}
                </p>

            </article>
            """

        if not tarjetas:
            tarjetas = """
            <div class="vacia">
                Todavía no hay fotos
                en esta carpeta.
            </div>
            """

        etiqueta = ""

        if (
            nombre_carpeta
            == obtener_carpeta_activa()
        ):
            etiqueta = """
            <span class="etiqueta">
                Destino activo
            </span>
            """

        actualizacion = (
            script_actualizacion(
                firma_carpeta(
                    nombre_carpeta
                ),
                (
                    "/estado?"
                    "vista=galeria"
                    f"&carpeta={carpeta_url}"
                    "&_="
                )
            )
        )

        html = f"""
        <!DOCTYPE html>

        <html lang="es">

        <head>
            <meta charset="UTF-8">

            <meta
                name="viewport"
                content="width=device-width, initial-scale=1.0"
            >

            <title>
                {escape(titulo)}
            </title>

            <style>
                {ESTILO}
            </style>
        </head>

        <body>

            <main class="contenedor">

                <a
                    class="volver"
                    href="/"
                >
                    &#10094; Carpetas
                </a>

                <div class="encabezado">

                    <div>
                        <h1>
                            {escape(titulo)}
                        </h1>

                        <p class="contador">
                            {len(fotos)} fotos
                        </p>
                    </div>

                    {etiqueta}

                </div>

                <section class="galeria">
                    {tarjetas}
                </section>

            </main>

            {actualizacion}

        </body>

        </html>
        """

        self.enviar(
            html
        )

    # ======================================================
    # VISOR INDIVIDUAL
    # ======================================================

    def mostrar_visor(
        self,
        nombre_carpeta,
        nombre_foto
    ):
        foto, codigo = resolver_foto(
            nombre_carpeta,
            nombre_foto
        )

        if codigo != 200:
            self.enviar(
                "Foto no encontrada",
                codigo=codigo
            )
            return

        carpeta = resolver_carpeta(
            nombre_carpeta
        )

        fotos = listar_fotos(
            carpeta
        )

        nombres = [
            ruta.name
            for ruta in fotos
        ]

        if nombre_foto not in nombres:
            self.enviar(
                "Foto no encontrada",
                codigo=404
            )
            return

        indice = nombres.index(
            nombre_foto
        )

        carpeta_url = quote(
            nombre_carpeta,
            safe=""
        )

        foto_url = quote(
            nombre_foto,
            safe=""
        )

        flechas = ""

        if len(nombres) > 1:
            anterior_nombre = nombres[
                (indice - 1)
                % len(nombres)
            ]

            siguiente_nombre = nombres[
                (indice + 1)
                % len(nombres)
            ]

            anterior_url = quote(
                anterior_nombre,
                safe=""
            )

            siguiente_url = quote(
                siguiente_nombre,
                safe=""
            )

            flechas = f"""
            <a
                id="anterior"
                class="flecha anterior"
                href="/ver?carpeta={carpeta_url}&amp;foto={anterior_url}"
                aria-label="Foto anterior"
            >
                &#10094;
            </a>

            <a
                id="siguiente"
                class="flecha siguiente"
                href="/ver?carpeta={carpeta_url}&amp;foto={siguiente_url}"
                aria-label="Foto siguiente"
            >
                &#10095;
            </a>
            """

        actualizacion = (
            script_actualizacion(
                firma_carpeta(
                    nombre_carpeta
                ),
                (
                    "/estado?"
                    "vista=galeria"
                    f"&carpeta={carpeta_url}"
                    "&_="
                )
            )
        )

        teclado = """
        <script>
            document.addEventListener(
                "keydown",
                function(evento) {

                    if (
                        evento.key
                        === "ArrowLeft"
                    ) {
                        const boton =
                            document.getElementById(
                                "anterior"
                            );

                        if (boton) {
                            boton.click();
                        }
                    }

                    if (
                        evento.key
                        === "ArrowRight"
                    ) {
                        const boton =
                            document.getElementById(
                                "siguiente"
                            );

                        if (boton) {
                            boton.click();
                        }
                    }
                }
            );
        </script>
        """

        nombre_html = escape(
            nombre_foto,
            quote=True
        )

        html = f"""
        <!DOCTYPE html>

        <html lang="es">

        <head>
            <meta charset="UTF-8">

            <meta
                name="viewport"
                content="width=device-width, initial-scale=1.0"
            >

            <title>
                {nombre_html}
            </title>

            <style>
                {ESTILO_VISOR}
            </style>
        </head>

        <body>

            <a
                class="cerrar"
                href="/galeria?carpeta={carpeta_url}"
                aria-label="Volver a la carpeta"
            >
                &times;
            </a>

            {flechas}

            <div class="visor">

                <img
                    src="/imagen?carpeta={carpeta_url}&amp;foto={foto_url}"
                    alt="{nombre_html}"
                >

                <p class="nombre">
                    {nombre_html}
                </p>

                <p class="posicion">
                    Foto {indice + 1}
                    de {len(nombres)}
                </p>

            </div>

            {teclado}

            {actualizacion}

        </body>

        </html>
        """

        self.enviar(
            html
        )

    # ======================================================
    # ENVIAR IMAGEN
    # ======================================================

    def enviar_imagen(
        self,
        nombre_carpeta,
        nombre_foto
    ):
        foto, codigo = resolver_foto(
            nombre_carpeta,
            nombre_foto
        )

        if codigo != 200:
            self.enviar(
                "Foto no encontrada",
                codigo=codigo
            )
            return

        try:
            contenido = foto.read_bytes()

        except OSError:
            self.enviar(
                "No se pudo leer la foto",
                codigo=500
            )
            return

        tipo = mimetypes.guess_type(
            foto.name
        )[0]

        if tipo is None:
            tipo = (
                "application/octet-stream"
            )

        self.enviar(
            contenido,
            tipo=tipo
        )

    # ======================================================
    # LISTA JSON DE FOTOS
    # ======================================================

    def enviar_lista_fotos(
        self,
        nombre_carpeta
    ):
        carpeta = resolver_carpeta(
            nombre_carpeta
        )

        if carpeta is None:
            self.enviar(
                "Carpeta no encontrada",
                codigo=404
            )
            return

        nombres = [
            foto.name
            for foto in listar_fotos(
                carpeta
            )
        ]

        self.enviar(
            json.dumps(
                nombres,
                ensure_ascii=False
            ),
            tipo=(
                "application/json; "
                "charset=utf-8"
            )
        )

    # ======================================================
    # ESTADO PARA ACTUALIZACIÓN AUTOMÁTICA
    # ======================================================

    def enviar_estado(
        self,
        datos
    ):
        vista = datos.get(
            "vista",
            ["menu"]
        )[0]

        if vista == "galeria":
            nombre = datos.get(
                "carpeta",
                [""]
            )[0]

            if resolver_carpeta(nombre) is None:
                self.enviar(
                    "Carpeta no encontrada",
                    codigo=404
                )
                return

            firma = firma_carpeta(
                nombre
            )

        else:
            firma = firma_menu()

        self.enviar(
            json.dumps(
                {
                    "firma": firma
                }
            ),
            tipo=(
                "application/json; "
                "charset=utf-8"
            )
        )


# ==========================================================
# INICIAR SERVIDOR
# ==========================================================

if __name__ == "__main__":

    CARPETA_FOTOS.mkdir(
        parents=True,
        exist_ok=True
    )

    cargar_carpeta_activa()

    DETENER_ORGANIZADOR.clear()

    hilo_organizador = threading.Thread(
        target=organizar_fotos_nuevas,
        daemon=True
    )

    hilo_organizador.start()

    print()
    print("======================================")
    print("       SERVIDOR DE FOTOS OPENMV")
    print("======================================")
    print()

    print("Carpeta de fotos:")
    print(CARPETA_FOTOS)
    print()

    print("Carpeta activa:")

    print(
        obtener_carpeta_activa()
        or "Sin clasificar"
    )

    print()
    print(
        f"Servidor iniciado "
        f"en el puerto {PUERTO}"
    )
    print()

    print("Desde esta PC:")
    print(
        f"http://localhost:{PUERTO}"
    )
    print()

    print("Desde el celular:")
    print(
        f"http://IP_DE_LA_PC:{PUERTO}"
    )
    print()

    print(
        "Presioná CTRL+C "
        "para detener."
    )
    print()

    servidor = ThreadingHTTPServer(
        (
            HOST,
            PUERTO
        ),
        ServidorFotos
    )

    try:
        servidor.serve_forever()

    except KeyboardInterrupt:
        print(
            "\nServidor detenido."
        )

    finally:
        DETENER_ORGANIZADOR.set()
        servidor.server_close()


# %%
