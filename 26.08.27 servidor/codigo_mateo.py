#%%



from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote
import json


# ==========================================================
# CARPETA DONDE SE GUARDAN LAS FOTOS
# ==========================================================

CARPETA_FOTOS = Path(
    r"C:\BackUp\Facultad\Labo 7\Fotos\26.08.25 conf de parametros de la camara"
)


# ==========================================================
# CONFIGURACIÓN DEL SERVIDOR
# ==========================================================

HOST = "0.0.0.0"
PUERTO = 8000


# ==========================================================
# MANEJADOR DE PETICIONES
# ==========================================================

class ServidorFotos(BaseHTTPRequestHandler):

    def enviar_respuesta(self, contenido, tipo_contenido, codigo=200):

        self.send_response(codigo)

        self.send_header(
            "Content-Type",
            tipo_contenido
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )

        self.end_headers()

        self.wfile.write(contenido)


    # ======================================================
    # GET
    # ======================================================

    def do_GET(self):

        ruta = unquote(self.path)

        # --------------------------------------------------
        # LISTA DE FOTOS
        # --------------------------------------------------

        if ruta == "/fotos":

            fotos = sorted(
                CARPETA_FOTOS.glob("*.jpg")
            )

            nombres = [
                foto.name
                for foto in fotos
            ]

            datos = json.dumps(
                nombres,
                ensure_ascii=False
            ).encode("utf-8")

            self.enviar_respuesta(
                datos,
                "application/json; charset=utf-8"
            )

            return


        # --------------------------------------------------
        # OBTENER UNA FOTO
        # --------------------------------------------------

        if ruta.startswith("/foto/"):

            nombre = ruta[len("/foto/"):]

            archivo = CARPETA_FOTOS / nombre

            # Seguridad: evitar acceder fuera de la carpeta
            try:
                archivo = archivo.resolve()
                carpeta = CARPETA_FOTOS.resolve()

                if carpeta not in archivo.parents:
                    self.enviar_respuesta(
                        b"Acceso no permitido",
                        "text/plain; charset=utf-8",
                        403
                    )
                    return

            except Exception:

                self.enviar_respuesta(
                    b"Archivo invalido",
                    "text/plain; charset=utf-8",
                    400
                )

                return


            if not archivo.exists() or not archivo.is_file():

                self.enviar_respuesta(
                    b"Foto no encontrada",
                    "text/plain; charset=utf-8",
                    404
                )

                return


            with open(archivo, "rb") as f:

                imagen = f.read()


            self.enviar_respuesta(
                imagen,
                "image/jpeg"
            )

            return


        # --------------------------------------------------
        # PAGINA PRINCIPAL
        # --------------------------------------------------

        if ruta == "/":

            fotos = sorted(
                CARPETA_FOTOS.glob("*.jpg")
            )

            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Fotos OpenMV</title>
            </head>

            <body>

                <h1>Fotos OpenMV</h1>

                <p>
                    Fotos encontradas: %d
                </p>

                <hr>

            """ % len(fotos)


            for foto in fotos:

                html += f"""
                <div style="margin-bottom:30px">

                    <p>{foto.name}</p>

                    <img
                        src="/foto/{foto.name}"
                        style="max-width:500px"
                    >

                </div>
                """


            html += """
            </body>
            </html>
            """


            self.enviar_respuesta(
                html.encode("utf-8"),
                "text/html; charset=utf-8"
            )

            return


        # --------------------------------------------------
        # RUTA DESCONOCIDA
        # --------------------------------------------------

        self.enviar_respuesta(
            b"Ruta no encontrada",
            "text/plain; charset=utf-8",
            404
        )


# ==========================================================
# ARRANCAR SERVIDOR
# ==========================================================

if __name__ == "__main__":

    print()
    print("======================================")
    print("       SERVIDOR DE FOTOS OPENMV")
    print("======================================")
    print()

    print(f"Carpeta de fotos:")
    print(CARPETA_FOTOS)

    print()
    print(f"Servidor iniciado en el puerto {PUERTO}")
    print()
    print("Desde esta PC:")
    print(f"http://localhost:{PUERTO}")
    print()
    print("Desde el celular:")
    print("http://IP_DE_LA_PC:8000")
    print()
    print("Presioná CTRL+C para detener.")
    print()

    servidor = HTTPServer(
        (HOST, PUERTO),
        ServidorFotos
    )

    servidor.serve_forever()
# %%
