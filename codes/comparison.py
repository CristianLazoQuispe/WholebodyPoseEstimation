import sys
import cv2
import copy
import numpy as np
sys.path.insert(1,'../src/')

from wholebodypose.utils.vision import draw_text,DrawerPose

from wholebodypose.models.mediapipe.model import MediapipeModel
from wholebodypose.models.rtmpose.model   import RTMPoseModel
from wholebodypose.models.vitpose.model   import VITPoseModel


draw_skeleton   = DrawerPose()
model_mediapipe = MediapipeModel(image_mode=True,mode_coco=True,use_thresholding=True,kpt_thr=0.5)

model_rtmpose   = RTMPoseModel(mode='performance',backend='onnxruntime',
                     use_thresholding=True,filter_noise=True,kpt_thr=2.5)

model_vitpose   = VITPoseModel(device='cuda', model_name='ViTPose+_huge_coco_wholebody',
                 use_thresholding=True,kpt_thr=0.5)

filename = "/media/cristian/12FF1F6D0CD48422/Research/Gloss/Gloss/Datasets/PUCP/5. Segundo avance (corregido)/TELEFONO/TELEFONO_ORACION_2.mp4"

filename = "/media/cristian/12FF1F6D0CD48422/Research/Gloss/Gloss/Datasets/wlasl-complete/videos/16096.mp4"
capture = cv2.VideoCapture(filename)

cv2.namedWindow("Testing models", cv2.WINDOW_NORMAL) 
width  = 640#int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
height = 480#int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height
print("width :",width)
print("height:",height)

out = cv2.VideoWriter( 
    "../results/output_camera.avi", cv2.VideoWriter_fourcc(*'MPEG'), 15, (width*2,height*2)) 

cnt = 0

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
    frame_med = copy.deepcopy(frame)
    frame_rtm = copy.deepcopy(frame)
    frame_vit = copy.deepcopy(frame)

    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    keypoints, scores = model_mediapipe.predict(frame_rgb)
    frame_med = draw_skeleton(frame_med, keypoints, scores, kpt_thr=0.5,line_width=2,radius=2)
    
    keypoints, scores = model_rtmpose.predict(frame_rgb)
    frame_rtm = draw_skeleton(frame_rtm, keypoints, scores, kpt_thr=0.5,line_width=2,radius=2)
    
    keypoints, scores = model_vitpose.predict(frame_rgb)
    frame_vit = draw_skeleton(frame_vit, keypoints, scores, kpt_thr=0.5,line_width=2,radius=2)
    
    # Dibujar el texto con fondo negro
    draw_text(frame, "ORIGINAL", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 
              0.75, (255, 255, 255), (0, 0, 0), 4)
    draw_text(frame_med, "MEDIAPIPE", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 
              0.75, (0, 255, 255), (0, 0, 0), 4)
    draw_text(frame_vit, "VITPOSE", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 
              0.75, (255, 0, 255), (0, 0, 0), 4)
    draw_text(frame_rtm, "OURS", (290, 50), cv2.FONT_HERSHEY_SIMPLEX, 
              0.75, (255, 255, 0), (0, 0, 0), 4)

    vis1 = np.concatenate((frame,frame_med), axis=1)
    vis2 = np.concatenate((frame_vit,frame_rtm), axis=1)
    vis  = np.concatenate((vis1,vis2), axis=0)

    cv2.imshow('Testing models', vis)
    
    #time.sleep(1)
    out.write(vis)

    if cv2.waitKey(1) == 27:
        break
 
cv2.destroyAllWindows()
capture.release()
out.release()
