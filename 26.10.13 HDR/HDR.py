#%%

# Agrego la raiz del proyecto
import sys
import os

os.environ["PYTHONPATH"] = r"C:\BackUp\Facultad\Labo 7\env_labo7"

sys.path.insert(0, os.environ["PYTHONPATH"])


import cv2
import numpy as np
from camara_openmv import OpenMVCamera, PixFormat, Framesize

class GeneradorHDR:
    def __init__(self, camara: OpenMVCamera):
        self.cam = camara

    def tomar_foto_hdr(self, framesize_original=Framesize.QVGA, mostrar=False, guardar_como=None):
        """
        Interrumpe el flujo normal para sacar una foto HDR de máxima calidad (WQXGA2).
        Luego restaura los parámetros exactos (formato, resolución, exposición y ganancias)
        para que el código principal pueda seguir operando sin enterarse del cambio.
        
        Args:
            framesize_original (Framesize): La resolución a la que debe volver la cámara.
            mostrar (bool): Si es True, bloquea el código mostrando la imagen fusionada.
            guardar_como (str): Ruta donde guardar el archivo .jpg final.
        """
        print("\n--- INICIANDO CAPTURA HDR ---")
        # 1. GUARDAR EL ESTADO ORIGINAL
        # Leer valores actuales del hardware
        info_original = self.cam.info()
        exp_original = info_original["exposure_us"]
        ganancia_original = info_original["gain_db"]
        rgb_gain_original = info_original["rgb_gain_db"]
        
        # Leer valores guardados en el software
        pixformat_original = self.cam.pixformat
        if pixformat_original is None:
            pixformat_original = PixFormat.RGB565 # Fallback por si nunca se seteó
            
        try:
            # 2. CONFIGURAR PARA MÁXIMA CALIDAD HDR
            print("Configurando cámara a máxima resolución (WQXGA2)...")
            self.cam.set_pixformat(PixFormat.RGB565)
            # WQXGA2 es la máxima resolución disponible en el enum de camara_openmv.py
            self.cam.set_framesize(Framesize.WQXGA2) 
            self.cam.set_quality(100) # Máxima calidad JPEG de transmisión
            
            # Dejar que la cámara mida la luz de la escena a esta nueva resolución
            self.cam.set_exposure_auto(True)
            self.cam.set_gain_auto(True)
            self.cam.set_whitebal_auto(True)
            self.cam.descartar_frames(cantidad=4)
            
            # Obtener el tiempo de exposición ideal medido
            info_actual = self.cam.info()
            exp_base = info_actual["exposure_us"]
            
            # Congelar parámetros para que las 3 fotos coincidan
            self.cam.set_whitebal_auto(False)
            self.cam.set_gain_auto(False)
            
            # Tiempos de bracketing (-2EV, 0, +2EV aprox) ----> cada paso entero de exposición (EV) duplica o divide por dos el tiempo que el sensor recibe luz
            exp_oscura = int(exp_base / 4)
            exp_clara = int(exp_base * 4)
            
            exposiciones = [exp_oscura, exp_base, exp_clara]
            imagenes_cv = []
            
            print("Capturando bracketing (3 exposiciones)...")
            for exp in exposiciones:
                self.cam.set_exposure(exp)
                self.cam.descartar_frames(cantidad=2) # Esperar a que el hardware aplique el cambio
                
                img_bytes = self.cam.capturar()
                
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    imagenes_cv.append(img)
                else:
                    print(f"Error decodificando frame en {exp}us")
                    
            if len(imagenes_cv) != 3:
                raise Exception("Fallo en la captura de las 3 imágenes.")
                
            print("Fusionando (Algoritmo Mertens)...") # ----> Combina los mapas de peso y las imágenes mediante una pirámide Laplaciana para evitar los molestos halos o artefactos que suelen aparecer en las zonas de alto contraste. La pirámide Laplaciana básicamente descompone una imagen en diferentes bandas de frecuencia
            merge_mertens = cv2.createMergeMertens()
            hdr_fusion = merge_mertens.process(imagenes_cv)
            
            # Escalar float32 [0-1] a uint8 [0-255]
            hdr_8bit = np.clip(hdr_fusion * 255, 0, 255).astype('uint8')
            
            if guardar_como:
                cv2.imwrite(guardar_como, hdr_8bit)
                print(f"HDR guardado en: {guardar_como}")
                
            if mostrar:
                # Redimensionar la ventana para que entre en la pantalla (WQXGA2 es muy grande)
                h, w = hdr_8bit.shape[:2]
                img_mostrar = cv2.resize(hdr_8bit, (w//3, h//3))
                cv2.imshow("HDR Alta Calidad", img_mostrar)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            return hdr_8bit

        finally:
            # 3. RESTAURAR ABSOLUTAMENTE TODO EL ESTADO ORIGINAL
            print("Restaurando parámetros originales...")
            
            # Volver a la resolución y formato que tenía el código principal
            self.cam.set_pixformat(pixformat_original)
            self.cam.set_framesize(framesize_original)
            
            # Volver a los valores de exposición, ganancia y color exactos
            self.cam.set_exposure(exp_original)
            self.cam.set_gain(ganancia_original)
            self.cam.set_whitebal(rgb_gain_original[0], rgb_gain_original[1], rgb_gain_original[2])
            
            # Descartar frames para que el sensor se reacomode al estado viejo
            self.cam.descartar_frames(cantidad=2)
            print("--- CÁMARA LISTA PARA CONTINUAR ---")



#%%


if __name__ == "__main__":
    cam = OpenMVCamera(puerto="COM7")
    cam.conectar()
    
    # 1. Configuración rápida/normal de tu programa principal
    estado_normal_resolucion = Framesize.QVGA
    cam.config_de_una(pixformat=PixFormat.GRAYSCALE, resolucion=estado_normal_resolucion)
    
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
        cam.desconectar()



# %%
