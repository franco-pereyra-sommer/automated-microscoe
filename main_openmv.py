import csi
from pyb import USB_VCP

usb = USB_VCP()
cam = csi.CSI()

cam.reset()
cam.pixformat(csi.JPEG)
cam.framesize(csi.VGA)
cam.quality(90)

cam.auto_exposure(True)
cam.auto_gain(True)
cam.auto_whitebal(True)

cam.snapshot(time=2000)

formats = {
    "JPEG": csi.JPEG,
    "RGB565": csi.RGB565,
    "GRAYSCALE": csi.GRAYSCALE
}

sizes = {
    "QQVGA": csi.QQVGA,
    "QVGA": csi.QVGA,
    "VGA": csi.VGA,
    "HD": csi.HD,
    "FHD": csi.FHD,
    "WQXGA2": csi.WQXGA2
}

# Estado que el firmware necesita recordar entre comandos, porque
# capture() necesita saber esto y no lo puede volver a preguntar.
# Antes capture() ignoraba los dos: mandaba siempre compress(quality=90)
# sin mirar current_quality (así que set_quality no hacía nada en la
# práctica), y volvía a comprimir con .compress() incluso cuando el
# pixformat ya era JPEG (snapshot() en ese modo ya devuelve una imagen
# comprimida por el sensor; recomprimirla con .compress() tiene
# comportamiento poco confiable).
current_pixformat = "JPEG"
current_quality = 90


def send_ok():
    usb.write(b"OK\n")


def send_error():
    usb.write(b"ERR\n")


def capture():
    img = cam.snapshot()

    if current_pixformat == "JPEG":
        # El sensor ya entrega la imagen comprimida en este modo:
        # comprimirla de nuevo no respeta current_quality y puede no
        # achicarla como se espera.
        jpeg = img
    else:
        jpeg = img.compress(quality=current_quality)

    data = jpeg.bytearray()

    usb.write(str(len(data)) + "\n")

    while not usb.any():
        pass

    usb.readline()
    usb.write(data)


def set_exposure_auto(enable=True, exposure_us=None):

    if enable:
        cam.auto_exposure(True)
    else:
        if exposure_us is None:
            raise ValueError("Falta exposure_us")

        cam.auto_exposure(
            False,
            exposure_us=int(exposure_us)
        )


def set_gain_auto(enable=True, gain_db=None):

    if enable:
        cam.auto_gain(True)
    else:
        if gain_db is None:
            raise ValueError("Falta gain_db")

        cam.auto_gain(
            False,
            gain_db=float(gain_db)
        )


def set_whitebal_auto(
    enable=True,
    rgb_gain_db=None
):

    if enable:
        cam.auto_whitebal(True)
    else:
        if rgb_gain_db is None:
            raise ValueError(
                "Falta rgb_gain_db"
            )

        cam.auto_whitebal(
            False,
            rgb_gain_db=rgb_gain_db
        )


def process(cmd):

    global current_pixformat, current_quality

    parts = cmd.split()

    if not parts:
        return

    try:

        if parts[0] == "capture":
            capture()
            return

        if parts[0] == "set_pixformat":
            cam.pixformat(formats[parts[1]])
            current_pixformat = parts[1]
            send_ok()
            return

        if parts[0] == "set_framesize":
            cam.framesize(sizes[parts[1]])
            send_ok()
            return

        if parts[0] == "set_quality":
            cam.quality(int(parts[1]))
            current_quality = int(parts[1])
            send_ok()
            return

        if parts[0] == "set_exposure_auto":

            enable = bool(int(parts[1]))

            if enable:
                set_exposure_auto(True)
            else:
                set_exposure_auto(
                    False,
                    int(parts[2])
                )

            send_ok()
            return

        if parts[0] == "set_gain_auto":

            enable = bool(int(parts[1]))

            if enable:
                set_gain_auto(True)
            else:
                set_gain_auto(
                    False,
                    float(parts[2])
                )

            send_ok()
            return

        if parts[0] == "set_whitebal_auto":

            enable = bool(int(parts[1]))

            if enable:
                set_whitebal_auto(True)
            else:
                set_whitebal_auto(
                    False,
                    (
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4])
                    )
                )

            send_ok()
            return

        if parts[0] == "info":
            usb.write(
                (
                    str(cam.exposure_us()) + " " +
                    str(cam.gain_db()) + " " +
                    str(cam.rgb_gain_db()) + "\n"
                ).encode()
            )
            return

        send_error()

    except Exception:
        send_error()


while True:

    if usb.any():

        cmd = usb.readline()

        if cmd:
            process(cmd.decode().strip())
