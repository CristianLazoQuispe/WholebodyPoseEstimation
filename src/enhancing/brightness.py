import cv2
import numpy as np

def adjust_gamma(image, gamma=1.0):
    # Construir la tabla de búsqueda
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    
    # Aplicar la corrección de gamma usando la tabla de búsqueda
    return cv2.LUT(image, table)
def log_transform_rgb(image, c=1.0):
    # Convertir la imagen a float32 y normalizar a [0, 1]
    image_float = image.astype(np.float32) / 255.0
    
    # Aplicar la transformación logarítmica a cada canal
    image_log = c * np.log(image_float + 1)
    
    # Escalar los valores de píxel al rango [0, 255]
    image_log = np.clip(image_log * 255, 0, 255).astype(np.uint8)
    
    return image_log

def adjust_brightness_rgb(image, gamma=1.0):
    # Convertir la imagen a float32 y normalizar a [0, 1]
    image_float = image.astype(np.float32) / 255.0
    
    # Aplicar la transformación exponencial inversa a cada canal
    image_adjusted = np.power(image_float, gamma)
    
    # Escalar los valores de píxel al rango [0, 255]
    image_adjusted = (image_adjusted * 255).astype(np.uint8)
    
    return image_adjusted