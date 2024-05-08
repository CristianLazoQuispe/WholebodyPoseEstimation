import sys
import cv2
import os


module_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(module_path,'../../src/'))

import matplotlib.pyplot as plt
from models.mediapipe.model import mediapipe_model
from models.mediapipe.map_mp2coco import MP2COCO
import time
from rtmlib import Wholebody, draw_skeleton
import copy 
# import the opencv library 
import cv2 



model = mediapipe_model()
maper = MP2COCO()


vid = cv2.VideoCapture(0) 
while(True): 
    
    ret, frame = vid.read() 
    if ret is None:
        break

    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = model.predict(frame_rgb)
    
   
    #frame = model.draw_mediapipe(frame,results)

    keypoints, scores = maper.process(frame,results)

    frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.5,
                                line_width=6,radius=6)

    cv2.imshow('frame', frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
