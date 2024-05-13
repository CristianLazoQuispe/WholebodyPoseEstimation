

import numpy as np
keypoints_buffer = []

def process_keypoints(keypoints, keypoints_buffer, window_size=5):
    # Agregar los keypoints actuales al buffer
    keypoints_buffer.append(keypoints)
    
    # Si el buffer tiene suficientes keypoints, aplicar los filtros
    if len(keypoints_buffer) >= window_size:
        # Convertir el buffer a un array NumPy
        keypoints_array = np.array(keypoints_buffer)
        
        # Aplicar el filtro de media móvil
        smoothed_keypoints = np.mean(keypoints_array, axis=0)
        
        # Aplicar el filtro de mediana para eliminar outliers
        median_filtered_keypoints = np.median(keypoints_array, axis=0)
        
        # Combinar los keypoints suavizados y filtrados
        combined_keypoints = (smoothed_keypoints + median_filtered_keypoints) / 2
        
        #keypoints['points'][:,0] = savgol_filter(keypoints['points'][:,0], 5, 2, mode='interp')#nearest')
        #keypoints['points'][:,1] = savgol_filter(keypoints['points'][:,1], 5, 2, mode='interp')#nearest')#interp

        # Eliminar el keypoint más antiguo del buffer
        keypoints_buffer.pop(0)
        
        # Retornar los keypoints combinados como un vector
        return combined_keypoints.tolist()
    
    # Si el buffer no tiene suficientes keypoints, retornar los keypoints originales
    return keypoints