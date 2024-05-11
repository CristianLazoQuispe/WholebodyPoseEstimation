import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

def initialize_kalman_filters(num_keypoints):
    kalman_filters = []
    for _ in range(num_keypoints):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.zeros(4)  # Inicializar estado con valores predeterminados
        kf.F = np.array([[1, 1, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]])  # Matriz de transición de estados
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0]])  # Matriz de observación
        kf.P = np.eye(4) * 1000#1000#1000  # Matriz de covarianza del estado inicial
        kf.R = np.eye(2) *1#* 1000.0  # Matriz de covarianza del ruido de observación
        kf.Q = np.eye(4) * 0.01  # Matriz de covarianza del ruido del proceso
        kalman_filters.append(kf)
    queues_filters = [[] for _ in range(num_keypoints)]  # Cola para almacenar los valores de la ventana
    return kalman_filters,queues_filters

def estimate_missing_keypoints(keypoints, scores, kalman_filters,queues_filters):
    num_keypoints = keypoints.shape[1]
    estimated_keypoints = np.zeros_like(keypoints)
    window_size = 5  # Tamaño de la ventana para el filtro de media móvil

    for i in range(num_keypoints):
        
        flag = False
        kalman_filters[i].update(keypoints[0, i])
        if scores[0, i] > 5.5:#qqq or scores[0,i]<1.5:
            estimated_keypoints[0, i] = keypoints[0, i]
        else:
            flag = True
            #kalman_filters[i].predict()
            #value = kalman_filters[i].x
            #estimated_keypoints[0, i] = value[[0,2]]
            
        #queues_filters[i].append(estimated_keypoints[0, i])
        if len(queues_filters[i]) > window_size:
            queues_filters[i].pop(0)
        if flag:
            estimated_keypoints[0, i] = np.mean(queues_filters[i], axis=0)
            #queues_filters[i][-1] = estimated_keypoints[0, i]            
            #kalman_filters[i].update(estimated_keypoints[0, i])

        queues_filters[i].append(estimated_keypoints[0, i])

    return estimated_keypoints

def initialize_kalman_filters_scores(num_keypoints):
    kalman_filters = []
    for _ in range(num_keypoints):
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.zeros(2)  # Inicializar estado con valores predeterminados
        kf.F = np.array([[1, 1],
                         [0, 1]])  # Matriz de transición de estados
        kf.H = np.array([[1, 0]])  # Matriz de observación
        kf.P = np.eye(2) * 1000#1000#1000  # Matriz de covarianza del estado inicial
        kf.R = np.array([[1]])*1  # Matriz de covarianza del ruido de observación
        kf.Q = np.array([[1, 0],
                         [0, 1]]) * 0.01  # Matriz de covarianza del ruido del proceso
        kalman_filters.append(kf)
    queues_filters = [[] for _ in range(num_keypoints)]  # Cola para almacenar los valores de la ventana
    return kalman_filters,queues_filters

def estimate_keypoint_scores(keypoints,scores, kalman_filters,queues_filters):
    num_keypoints = scores.shape[1]
    estimated_scores = np.zeros_like(scores)
    
    window_size = 5
    
    for i in range(num_keypoints):
        flag = False

        kalman_filters[i].update(scores[0, i])
        if scores[0, i] > 5.5:#  or scores[0,i]<1.5:
            estimated_scores[0, i] = scores[0, i]
        else:
            flag = True
            #kalman_filters[i].predict()     
            #value = kalman_filters[i].x[0]
            #if i==99:
            #    print("queues_filters[i]:",queues_filters[i])
            #    print(f"i={i}, score={value} , key = {keypoints[0,i]}")
            #estimated_scores[0, i] = value
                
        #queues_filters[i].append(estimated_scores[0, i])
        if len(queues_filters[i]) > window_size:
            queues_filters[i].pop(0)
        if flag:
            estimated_scores[0, i] = np.mean(queues_filters[i], axis=0)
            #queues_filters[i][-1] = estimated_scores[0, i]
            #kalman_filters[i].update(estimated_scores[0, i])
        queues_filters[i].append(estimated_scores[0, i])

    return estimated_scores
# Inicializar los filtros de Kalman
num_keypoints = 133  # Número total de keypoints en tu modelo
kalman_filters,queues_filters = initialize_kalman_filters(num_keypoints)
kalman_filters_scores,queues_filters_scores = initialize_kalman_filters_scores(num_keypoints)