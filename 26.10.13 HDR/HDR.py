#%%
pasada = 8
# Agrego la raiz del proyecto
import sys
import os

from datetime import datetime

os.environ["PYTHONPATH"] = r"C:\BackUp\Facultad\Labo 7\env_labo7"

sys.path.insert(0, os.environ["PYTHONPATH"])


# import cv2
# import numpy as np
# from camara_openmv import OpenMVCamera, PixFormat, Framesize

# class GeneradorHDR:
#     def __init__(self, camara: OpenMVCamera):
#         self.cam = camara

#     def tomar_foto_hdr(self, framesize_original=Framesize.QVGA, mostrar=False, guardar_como=None):
#         """
#         Interrumpe el flujo normal para sacar una foto HDR de máxima calidad (WQXGA2).
#         Luego restaura los parámetros exactos (formato, resolución, exposición y ganancias)
#         para que el código principal pueda seguir operando sin enterarse del cambio.
        
#         Args:
#             framesize_original (Framesize): La resolución a la que debe volver la cámara.
#             mostrar (bool): Si es True, bloquea el código mostrando la imagen fusionada.
#             guardar_como (str): Ruta donde guardar el archivo .jpg final.
#         """
#         print("\n--- INICIANDO CAPTURA HDR ---")
#         # 1. GUARDAR EL ESTADO ORIGINAL
#         # Leer valores actuales del hardware
#         info_original = self.cam.info()
#         exp_original = info_original["exposure_us"]
#         ganancia_original = info_original["gain_db"]
#         rgb_gain_original = info_original["rgb_gain_db"]
        
#         # Leer valores guardados en el software
#         pixformat_original = self.cam.pixformat
#         if pixformat_original is None:
#             pixformat_original = PixFormat.RGB565 # Fallback por si nunca se seteó
            
#         try:
#             # 2. CONFIGURAR PARA MÁXIMA CALIDAD HDR
#             print("Configurando cámara a máxima resolución (WQXGA2)...")
#             self.cam.set_pixformat(PixFormat.RGB565)
#             # WQXGA2 es la máxima resolución disponible en el enum de camara_openmv.py
#             self.cam.set_framesize(Framesize.WQXGA2) 
#             self.cam.set_quality(100) # Máxima calidad JPEG de transmisión
            
#             # Dejar que la cámara mida la luz de la escena a esta nueva resolución
#             self.cam.set_exposure_auto(True)
#             self.cam.set_gain_auto(True)
#             self.cam.set_whitebal_auto(True)
#             self.cam.descartar_frames(cantidad=4)
            
#             # Obtener el tiempo de exposición ideal medido
#             info_actual = self.cam.info()
#             exp_base = info_actual["exposure_us"]
            
#             # Congelar parámetros para que las 3 fotos coincidan
#             self.cam.set_whitebal_auto(False)
#             self.cam.set_gain_auto(False)
            
#             # Tiempos de bracketing (-2EV, 0, +2EV aprox) ----> cada paso entero de exposición (EV) duplica o divide por dos el tiempo que el sensor recibe luz
#             exp_oscura = int(exp_base / 4)
#             exp_clara = int(exp_base * 4)
            
#             exposiciones = [exp_oscura, exp_base, exp_clara]
#             imagenes_cv = []
            
#             print("Capturando bracketing (3 exposiciones)...")
#             for exp in exposiciones:
#                 self.cam.set_exposure(exp)
#                 self.cam.descartar_frames(cantidad=2) # Esperar a que el hardware aplique el cambio
                
#                 img_bytes = self.cam.capturar()
                
#                 nparr = np.frombuffer(img_bytes, np.uint8)
#                 img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
#                 if img is not None:
#                     imagenes_cv.append(img)
#                 else:
#                     print(f"Error decodificando frame en {exp}us")
                    
#             if len(imagenes_cv) != 3:
#                 raise Exception("Fallo en la captura de las 3 imágenes.")
                
#             print("Fusionando (Algoritmo Mertens)...") # ----> Combina los mapas de peso y las imágenes mediante una pirámide Laplaciana para evitar los molestos halos o artefactos que suelen aparecer en las zonas de alto contraste. La pirámide Laplaciana básicamente descompone una imagen en diferentes bandas de frecuencia
#             merge_mertens = cv2.createMergeMertens()
#             hdr_fusion = merge_mertens.process(imagenes_cv)
            
#             # Escalar float32 [0-1] a uint8 [0-255]
#             hdr_8bit = np.clip(hdr_fusion * 255, 0, 255).astype('uint8')
            
#             if guardar_como:
#                 cv2.imwrite(guardar_como, hdr_8bit)
#                 print(f"HDR guardado en: {guardar_como}")
                
#             if mostrar:
#                 # Redimensionar la ventana para que entre en la pantalla (WQXGA2 es muy grande)
#                 h, w = hdr_8bit.shape[:2]
#                 img_mostrar = cv2.resize(hdr_8bit, (w//3, h//3))
#                 cv2.imshow("HDR Alta Calidad", img_mostrar)
#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()
                
#             return hdr_8bit

#         finally:
#             # 3. RESTAURAR ABSOLUTAMENTE TODO EL ESTADO ORIGINAL
#             print("Restaurando parámetros originales...")
            
#             # Volver a la resolución y formato que tenía el código principal
#             self.cam.set_pixformat(pixformat_original)
#             self.cam.set_framesize(framesize_original)
            
#             # Volver a los valores de exposición, ganancia y color exactos
#             self.cam.set_exposure(exp_original)
#             self.cam.set_gain(ganancia_original)
#             self.cam.set_whitebal(rgb_gain_original[0], rgb_gain_original[1], rgb_gain_original[2])
            
#             # Descartar frames para que el sensor se reacomode al estado viejo
#             self.cam.descartar_frames(cantidad=2)
#             print("--- CÁMARA LISTA PARA CONTINUAR ---")



import cv2
import numpy as np
from camara_openmv import OpenMVCamera, PixFormat, Framesize

class GeneradorHDR:
    def __init__(self, camara: OpenMVCamera):
        self.cam = camara

    def tomar_foto_hdr(self, framesize_original=Framesize.QVGA, mostrar=False, guardar_como=None):
        """
        Interrumpe el flujo normal para sacar una foto HDR de máxima calidad.condicion_para_foto_hd
        Luego restaura exactamente exposición, gain, white balance y resolución.
        """
        print("\n--- INICIANDO CAPTURA HDR ---")
        
        # 1. Obtener los valores originales con la extracción limpia
        info_orig = self.cam.info()
        exp_orig = info_orig["exposure_us"]
        gain_orig = info_orig["gain_db"]
        r_orig, g_orig, b_orig = info_orig["rgb_gain_db"]
        
        pixformat_orig = self.cam.pixformat or PixFormat.RGB565
        
        try:
            # 2. Configurar para máxima resolución
            self.cam.set_pixformat(PixFormat.RGB565)
            self.cam.set_framesize(Framesize.WQXGA2)
            self.cam.set_quality(100)
            
            # Auto-ajuste temporal para medir la luz de la escena
            self.cam.set_exposure_auto(True)
            self.cam.set_gain_auto(True)
            self.cam.set_whitebal_auto(True)
            self.cam.descartar_frames(cantidad=2)
            
            # Medir exposición base
            info_actual = self.cam.info()
            exp_base = info_actual["exposure_us"]
            
            # Congelar para las 3 tomas
            self.cam.set_whitebal(info_actual["rgb_gain_db"][0], info_actual["rgb_gain_db"][1], info_actual["rgb_gain_db"][2])
            self.cam.set_exposure(info_actual["exposure_us"])
            self.cam.set_gain(info_actual["gain_db"])

            array_exp = np.linspace(0.05, 2.5, 50)

            exposiciones = exp_base * array_exp
            
            # exposiciones = [int(exp_base / 4), exp_base, int(exp_base * 4)]
            imagenes_cv = []
            
            print(f"Capturando {len(exposiciones)} exposiciones...")
            for exp in exposiciones:
                exp = int(exp)
                self.cam.set_exposure(exp)
                self.cam.descartar_frames(cantidad=2)
                
                img_bytes = self.cam.capturar()

                carpeta_guardado = rf"C:\BackUp\Facultad\Labo 7\Fotos\HDR 4K pasada {pasada}"

                hora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                cam.guardar_imagen(
                    image_data = img_bytes,
                    carpeta=carpeta_guardado,
                    nombre = rf"{exp} ({grua.posicion_x}, {grua.posicion_y}, {grua.posicion_z}) {hora}",
                    mostrar=False
                )
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    imagenes_cv.append(img)
            
            # if len(imagenes_cv) != 3:
            #     raise Exception("Error durante la captura de las 3 imagenes")
                
            print("Fusionando imágenes...")
            merge_mertens = cv2.createMergeMertens()
            hdr_fusion = merge_mertens.process(imagenes_cv)
            hdr_8bit = np.clip(hdr_fusion * 255, 0, 255).astype('uint8')
            
            if guardar_como:
                cv2.imwrite(guardar_como, hdr_8bit)
                
            if mostrar:
                h, w = hdr_8bit.shape[:2]
                cv2.imshow("Resultado HDR", cv2.resize(hdr_8bit, (w//4, h//4)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            return hdr_8bit

        finally:
            # 3. RESTAURACIÓN GARANTIZADA DE PARÁMETROS ORIGINALES
            print("Restaurando parámetros originales...")
            self.cam.set_pixformat(pixformat_orig)
            self.cam.set_framesize(framesize_original)
            
            # Restaurar valores numéricos exactos de exposición, gain y balance de blancos
            self.cam.set_exposure(exp_orig)
            self.cam.set_gain(gain_orig)
            self.cam.set_whitebal(r_orig, g_orig, b_orig)
            
            self.cam.descartar_frames(cantidad=2)
            print("--- CÁMARA RESTAURADA AL ESTADO PREVIO ---")


#%%


# ============================================================
# EJEMPLO DE USO
# ============================================================

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

from camara_openmv import OpenMVCamera
from impresora_cosmos import TrimakerCOSMOS


grua = TrimakerCOSMOS("COM11"); grua.home()
cam = OpenMVCamera("COM7"); cam.conectar()


#%%


# 1) Parar la platina en un campo visual INTERIOR de la muestra
posicionar_en_muestra(cam, grua, x=150, y=150,
                    z_arriba_seguro=27.9,
                    z_min_seguridad=27.5, verificar_presencia=False,
                    hacer_foco_final=True, paso_grueso=0.005,
                    metrica_foco_final=laplacian_variance 
                    )


#%%


cam.config_autofoco_local()

mejor_z, posiciones_z, valores_metrica = autofocus(
    cam,
    grua,
    laplacian_variance,
    pasos_abajo=5,
    pasos_arriba=5,
    paso_z=0.0025,
    espera_movimiento=0.5,
    graficamos=True,
    guardamos=False,
    carpeta_guardado=r"C:\BackUp\Facultad\Labo 7\Fotos\26"
    )

cam.config_de_una()
# cam.config_autofoco_local()
time.sleep(0.5)
cam.mostrar(cam.capturar())
grua.pregunta_posicion_actual()


#%%


if __name__ == "__main__":
    # cam = OpenMVCamera(puerto="COM7")
    # cam.conectar()

    # 1. Configuración rápida/normal de tu programa principal
    estado_normal_resolucion = Framesize.QVGA
    cam.config_de_una(
        pixformat=PixFormat.GRAYSCALE, 
        resolucion=estado_normal_resolucion,
        gain_db= 25,
        exposure_us= 50000,
        whitebal= [66.22236, 61.43028, 60.206]
        )
    
    hdr_tool = GeneradorHDR(cam)
    
    try:
        while True:
            # Tu bucle normal de trabajo rápido
            img_normal = cam.capturar()
            print("Procesando frame rápido en escala de grises...")
            
            # Supongamos que aquí tu código decide que es un buen momento 
            # para sacar la foto de alta calidad:
            condicion_para_foto_hd = True 
            
            if condicion_para_foto_hd:
                # Llamamos a la clase. Le pasamos el Framesize para que sepa a dónde volver.
                img_alta_calidad = hdr_tool.tomar_foto_hdr(
                    framesize_original=estado_normal_resolucion, 
                    mostrar=True, 
                    guardar_como="foto_final.jpg"
                )
                
                # Una vez que esta línea termina, la cámara YA ESTÁ en 
                # GRAYSCALE y QVGA de nuevo automáticamente.
                print("Continuando con el flujo normal...")
                break # o seguir iterando

    finally:
        # cam.desconectar()
        print("termino")
        pasada += 1


# %%
