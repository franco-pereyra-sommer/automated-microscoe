import sensor
import image
import time
from pyb import USB_VCP

usb = USB_VCP()

# Inicializar cámara
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.VGA)
# sensor.set_framesize(sensor.QVGA)  # Set frame size to QVGA (320x240)
# sensor.set_framesize(sensor.HD)
# sensor.set_framesize(sensor.FHD)
# sensor.set_framesize(sensor.WQXGA2)

sensor.skip_frames(time=2000)

while True:

    if usb.any():
        cmd = usb.readline()

        if cmd:
            cmd = cmd.decode().strip()

            if cmd == "capture":

                img = sensor.snapshot()

                # Comprimir JPEG
                jpeg = img.compress(quality=90)

                data = jpeg.bytearray()

                # Enviar tamaño primero
                size = len(data)
                usb.write(str(size) + "\n")

                # Esperar pequeño ACK
                while not usb.any():
                    pass

                usb.readline()

                # Enviar imagen
                usb.write(data)

#%%

# Main minimal

from pyb import USB_VCP

usb = USB_VCP()

while True:

    if usb.any():

        cmd = usb.readline()

        if cmd:

            cmd = cmd.decode().strip()

            if cmd == "capture":

                usb.write("123\n")


#%%


import sensor
import time
from pyb import USB_VCP

usb = USB_VCP()

# Inicializar cámara
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)

sensor.skip_frames(time=2000)

while True:

    if usb.any():

        cmd = usb.readline()

        if cmd:

            cmd = cmd.decode().strip()

            if cmd == "capture":

                try:

                    img = sensor.snapshot()

                    # Crear JPEG
                    jpg = img.compress(quality=80)

                    # Bytes JPG
                    data = jpg.bytearray()

                    # Enviar tamaño
                    usb.write(str(len(data)) + "\n")

                    # Esperar ACK
                    while not usb.any():
                        pass

                    usb.readline()

                    # Enviar JPG
                    usb.write(data)

                except Exception as e:

                    usb.write("ERR\n")