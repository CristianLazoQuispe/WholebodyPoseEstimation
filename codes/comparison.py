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

import sys
import cv2
import os


module_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(module_path,'../src/'))
from models.mediapipe.map_mp2coco import MP2COCO
from models.mediapipe.model import mediapipe_model

maper = MP2COCO()
model_mediapipe = mediapipe_model()

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
filename = "/media/cristian/12FF1F6D0CD48422/Research/Gloss/Gloss/Datasets/videos_largos_pucp/Gramática de la LSP： Morfología (4).mp4"

filename = "/media/cristian/12FF1F6D0CD48422/Research/Gloss/Gloss/Datasets/PUCP/5. Segundo avance (corregido)/DINERO/DINERO_ORACION_2.mp4"
capture = cv2.VideoCapture(filename)

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

cv2.namedWindow("Testing models", cv2.WINDOW_NORMAL) 

width  = 640#int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
height = 480#int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height

print("width :",width)
print("height:",height)
out = cv2.VideoWriter( 
    "output.avi", cv2.VideoWriter_fourcc(*'MPEG'), 15, (width*2,height*2)) 

cnt = 0
def draw_text_with_background(frame, text, position, font, scale, text_color, background_color, thickness):
   text_size, _ = cv2.getTextSize(text, font, scale, thickness)
   text_width, text_height = text_size
   x, y = position
   
   cv2.rectangle(frame, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), background_color, -1)
   cv2.putText(frame, text, position, font, scale, text_color, thickness)

while (True):

    ret, frame = capture.read()
    cnt+=1
    if not ret:
        break
    #print(frame.shape)
    #(480, 640, 3)
    frame = cv2.resize(frame,(640,480))
    #if cnt<60*80:
    #    continue
    frame_bgr = copy.deepcopy(frame)
    frame_vit = copy.deepcopy(frame)

    frame_rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    ###################
    results = model_mediapipe.predict(frame_rgb)
    
   
    #frame = model.draw_mediapipe(frame,results)

    keypoints, scores = maper.process(frame,results)
    #print("keypoints",keypoints.shape)
    #print("scores",scores.shape)
    frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5,
                                line_width=4,radius=3)

    ###################
    keypoints = model(frame_vit)

    #onepose.visualize_keypoints(frame_vit, keypoints, model.keypoint_info, model.skeleton_info,
    #                    radius=1,
    #                    skeleton_thickness=4,
    #                    keypoint_thickness=2,)
    if len(keypoints['points'])>6:
        #keypoints['points'] = process_keypoints(keypoints['points'], keypoints_buffer,window_size=3)
        scores = np.moveaxis( keypoints['confidence'], 0, 1)
        keypoints = np.expand_dims(keypoints['points'], axis=0)

        #height, width, _ = frame_bgr.shape
        #keypoints = np.round(keypoints * [width, height]).astype(int)
        """        
        try:
            print(scores.shape)
            print(keypoints.shape)
            print(scores[0,:5])
            print(keypoints[0,:5,0])
        except:
            pass

        """
        frame_vit = draw_skeleton(copy.deepcopy(frame_vit), keypoints,scores, kpt_thr=0.5,line_width=4,radius=3)
    
    ######################
    keypoints, scores = wholebody(frame_rgb)

    frame_rrtm_original = draw_skeleton(copy.deepcopy(frame_bgr), keypoints, scores, kpt_thr=threshold,line_width=8,radius=6)
    
    foot_left_ids = [11,13,15]
    filter_ids(foot_left_ids,range(17,20),[15],"foot left",threshold=threshold)
    foot_right_ids = [12,14,16]
    filter_ids(foot_right_ids,range(20,23),[16],"foot right",threshold=threshold)

    hand_right_ids = [6,8,10,112]
    filter_ids(hand_right_ids,range(112,133),[10],"hand right",threshold=threshold)
    hand_left_ids = [5,7,9,91]
    filter_ids(hand_left_ids,range(91,112),[9],"hand left",threshold=threshold)
            
    #print(keypoints.shape)
    #print(scores.shape)
    frame_rrtm_our = draw_skeleton(copy.deepcopy(frame_bgr), keypoints, scores, kpt_thr=threshold,line_width=4,radius=3)
    


    #frame_rrtm = cv2.flip(frame_rrtm, 1)
    #frame_vit = cv2.flip(frame_vit, 1)
    #frame = cv2.flip(frame, 1)
    
    #cv2.putText(frame_bgr, "ORIGINAL", (250,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)
    #cv2.putText(frame, "MEDIAPIPE", (250,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5)
    #cv2.putText(frame_vit, "VITPOSE", (250,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 5)
    #cv2.putText(frame_rrtm_our, "OUR", (250,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 5)

    # Dibujar el texto con fondo negro
    draw_text_with_background(frame_bgr, "ORIGINAL", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), (0, 0, 0), 4)
    draw_text_with_background(frame, "MEDIAPIPE", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), (0, 0, 0), 4)
    draw_text_with_background(frame_vit, "VITPOSE", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), (0, 0, 0), 4)
    draw_text_with_background(frame_rrtm_our, "OUR", (290, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), (0, 0, 0), 4)

    vis1 = np.concatenate((frame_bgr,frame), axis=1)
    vis2 = np.concatenate((frame_vit,frame_rrtm_our), axis=1)
    vis = np.concatenate((vis1,vis2), axis=0)

    cv2.imshow('Testing models', vis)
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
