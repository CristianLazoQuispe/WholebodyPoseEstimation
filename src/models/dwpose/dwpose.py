import json
import os
from PIL import Image

from DWPOSE_MODELS.DWPose.src.dwpose import DwposeDetector
model = DwposeDetector.from_pretrained_default()

import numpy as np






import cv2
import time

#capture = cv2.VideoCapture("../../Testeos/no.webm")#1)
capture = cv2.VideoCapture(0)

while (True):

    ret, frame = capture.read()

    frame_rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #frame = dwpose(np.asarray(frame))

    imgOut,j,source = model(frame,
        include_hand=True,
        include_face=True,
        include_body=True,
        image_and_json=True,
        detect_resolution=256)

    cv2.imshow('Test hand', cv2.flip(np.array(imgOut), 1))
    time.sleep(1)
    if cv2.waitKey(1) == 27:
        break
 
cv2.destroyAllWindows()
capture.release()
