###############################################
import cv2

from rtmlib import Wholebody, draw_skeleton

device = 'cuda'  # cpu, cuda
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

wholebody = Wholebody(to_openpose=openpose_skeleton,
                      mode='performance',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                      backend=backend, device=device)
num_keypoints = 190
cnt = 0
threshold = 2.55
def filter_ids(list_ids,list_body,list_ids_remove,text,threshold=0.5):
    list_body= list(list_body)
    cnt_ids = np.sum(scores[0][list_ids] > threshold)
    if cnt_ids != len(list_ids):
        #print(f"remove: {text}")
        scores[0][list_body]= 0 
        scores[0][list_ids_remove]= 0 

######################################
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
#capture = cv2.VideoCapture(0)#1)
filename = "/media/cristian/12FF1F6D0CD48422/Research/Gloss/Gloss/Datasets/PUCP/PruebasLSP/LSP-testeo 8 de marzo/cristian_lazo/guardar.webm"
capture = cv2.VideoCapture(0) 
## initialize pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh


pose = mp_pose.Pose(static_image_mode=True,
                    min_detection_confidence=0.3, min_tracking_confidence=0.3,
                    model_complexity=2)

hands = handsModule.Hands(static_image_mode=True, min_detection_confidence=0.3, min_tracking_confidence=0.3, max_num_hands=2,
                          model_complexity=1)

face = mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5,max_num_faces=1)
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

cap = cv2.VideoCapture("input.mp4") 

out = cv2.VideoWriter( 
    "output.avi", cv2.VideoWriter_fourcc(*'MPEG'), 30, (640*3,480)) 
while (True):

    ret, frame = capture.read()

    if not ret:
        break
    #print(frame.shape)
    #(480, 640, 3)
    frame_bgr = copy.deepcopy(frame)
    frame_vit = copy.deepcopy(frame)

    frame_rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    ###################
    hand_results = hands.process(frame_rgb)
    pose_results = pose.process(frame_rgb)
    face_resutls = face.process(frame_rgb)

    if face_resutls.multi_face_landmarks != None:
      for faceLandmarks in face_resutls.multi_face_landmarks:
        mp_drawing.draw_landmarks(frame, faceLandmarks, mp_face.FACEMESH_CONTOURS,landmark_drawing_spec= mp_drawing.DrawingSpec(circle_radius=1,thickness=1,color=(255,0,0)))

    if hand_results.multi_hand_landmarks != None:
        for handLandmarks in hand_results.multi_hand_landmarks:
            drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)

    if pose_results.pose_landmarks != None:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        

    ###################
    keypoints = model(frame_vit)
    #if len(keypoints['points'])>6:
    #    keypoints['points'] = process_keypoints(keypoints['points'], keypoints_buffer,window_size=3)

    onepose.visualize_keypoints(frame_vit, keypoints, model.keypoint_info, model.skeleton_info)
    
    ######################
    keypoints, scores = wholebody(frame_rgb)
    
    foot_left_ids = [11,13,15]
    filter_ids(foot_left_ids,range(17,20),[15],"foot left",threshold=threshold)
    foot_right_ids = [12,14,16]
    filter_ids(foot_right_ids,range(20,23),[16],"foot right",threshold=threshold)

    hand_right_ids = [6,8,10,112]
    filter_ids(hand_right_ids,range(112,133),[10],"hand right",threshold=threshold)
    hand_left_ids = [5,7,9,91]
    filter_ids(hand_left_ids,range(91,112),[9],"hand left",threshold=threshold)
            
            
    frame_rrtm = draw_skeleton(frame_bgr, keypoints, scores, kpt_thr=threshold,line_width=2,radius=3)
    


    frame_rrtm = cv2.flip(frame_rrtm, 1)
    frame_vit = cv2.flip(frame_vit, 1)
    frame = cv2.flip(frame, 1)

    cv2.putText(frame, "MEDIAPIPE", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame_vit, "VITPOSE", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(frame_rrtm, "RTM", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    vis = np.concatenate((frame,frame_vit,frame_rrtm), axis=1)

    cv2.imshow('Test hand', vis)
    if flag_time:
        #time.sleep(5)
        flag_time  = False
    else:
        pass
    
    #time.sleep(1)
    out.write(vis)

    if cv2.waitKey(1) == 27:
        break
 
cv2.destroyAllWindows()
capture.release()
out.release()
