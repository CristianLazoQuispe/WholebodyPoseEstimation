
import cv2
import mediapipe
import mediapipe as mp
import time

#############################################
import cv2
import onepose
import time
from scipy.signal import savgol_filter


#model = onepose.create_model('ViTPose+_base_coco_wholebody').to("cuda")
#model = onepose.create_model('ViTPose_huge_mpii').to("cuda")
model  = onepose.create_model('ViTPose+_huge_coco_wholebody').to("cuda")
#model = onepose.create_model('ViTPose+_large_coco_wholebody').to("cuda")
  

#########################################################################
drawingModule = mediapipe.solutions.drawing_utils
handsModule   = mediapipe.solutions.hands

#capture = cv2.VideoCapture("../Testeos/ninho.webm")#1)
capture = cv2.VideoCapture(0)#1)
## initialize pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
#mp_face = mp.solutions.face_mesh


pose = mp_pose.Pose(static_image_mode=True,
                    min_detection_confidence=0.3, min_tracking_confidence=0.3,
                    model_complexity=2)

hands = handsModule.Hands(static_image_mode=True, min_detection_confidence=0.3, min_tracking_confidence=0.3, max_num_hands=2,
                          model_complexity=1)

#face = mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5,max_num_faces=1)
flag_time = True

import numpy as np

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

keypoints_buffer = []
import copy

while (True):

    ret, frame = capture.read()
    frame_vit = copy.deepcopy(frame)

    frame_rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    pose_results = pose.process(frame_rgb)
    #face_resutls = face.process(frame_rgb)
    if results.multi_hand_landmarks != None:
        for handLandmarks in results.multi_hand_landmarks:
            drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    keypoints = model(frame_vit)
    if len(keypoints['points'])>6:
        keypoints['points'] = process_keypoints(keypoints['points'], keypoints_buffer,window_size=3)

    onepose.visualize_keypoints(frame_vit, keypoints, model.keypoint_info, model.skeleton_info)
    

    #if face_resutls.multi_face_landmarks != None:
    #  for faceLandmarks in face_resutls.multi_face_landmarks:
    #    mp_drawing.draw_landmarks(frame, faceLandmarks, mp_face.FACEMESH_CONTOURS)
    vis = np.concatenate((frame, frame_vit), axis=1)

    cv2.imshow('Test hand', cv2.flip(vis, 1))
    if flag_time:
        #time.sleep(5)
        flag_time  = False
    else:
        pass
        #time.sleep(1)

    if cv2.waitKey(1) == 27:
        break
 
cv2.destroyAllWindows()
capture.release()
