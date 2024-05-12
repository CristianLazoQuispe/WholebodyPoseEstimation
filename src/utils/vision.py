import cv2

def draw_text(frame, text, position, font, scale, text_color, background_color, thickness):
   text_size, _ = cv2.getTextSize(text, font, scale, thickness)
   text_width, text_height = text_size
   x, y = position
   
   cv2.rectangle(frame, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), background_color, -1)
   cv2.putText(frame, text, position, font, scale, text_color, thickness)


import math

import cv2
import numpy as np
from .coco133 import coco133

def draw_bbox(img, bboxes, color=(0, 255, 0)):
    for bbox in bboxes:
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])), color, 2)
    return img


class DrawerPose:
   def __init__(self,mode_coco:bool=True):
      """

      Args:
          mode_coco (bool): Mode coco is according to COCO Wholebody format. Defaults to True.
            When mode_coco is False : it creates:
               new point -> middle_chest = 0.5('left_shoulder', 'right_shoulder')
               new point -> middle_hip   = 0.5('left_hip', 'right_hip')
               new link  -> ('middle_chest', 'nose'). 
               new link  -> ('middle_hip', 'middle_chest'). 
      """
      self.skeleton_dict = coco133
      self.keypoint_info = self.skeleton_dict['keypoint_info']
      self.skeleton_info = self.skeleton_dict['skeleton_info']
      self.mode_coco = mode_coco

      self.name2id = {}
      for i, kpt_info in self.keypoint_info.items():
         self.name2id[kpt_info['name']] = kpt_info['id']
      
      self.n_skeleton = 65 if self.mode_coco is True else 67
      
   def __call__(self,img,
                     keypoints,
                     scores,
                     kpt_thr=0.5,
                     radius=2,
                     line_width=2):
      
      if len(keypoints.shape) == 2:
         keypoints = keypoints[None, :, :]
         scores = scores[None, :, :]

      num_instance = keypoints.shape[0]
      for i in range(num_instance):
         img = self.draw_mmpose(img, keypoints[i], scores[i],
                              kpt_thr, radius, line_width)
      return img


   def draw_mmpose(self,img,
                  keypoints,
                  scores,
                  kpt_thr=0.5,
                  radius=2,
                  line_width=2):
      assert len(keypoints.shape) == 2

      if not self.mode_coco:
         x1 = int((keypoints[5][0]+keypoints[6][0])*0.5)
         y1 = int((keypoints[5][1]+keypoints[6][1])*0.5)
         score1 = (scores[5]+scores[6])*0.5
         
         x2 = int((keypoints[11][0]+keypoints[12][0])*0.5)
         y2 = int((keypoints[11][1]+keypoints[12][1])*0.5)
         score2 = (scores[11]+scores[12])*0.5

         keypoints = np.append(keypoints, [[x1,y1],[x2,y2]], axis=0)
         scores    = np.append(scores, [score1,score2], axis=0)

      vis_kpt = scores >= kpt_thr#[scores >= kpt_thr for s in scores]
      n = len(keypoints)
      
      for i in range(self.n_skeleton):
         ske_info   = self.skeleton_info[i]
         link       = ske_info['link']
         link_color = ske_info['color']
         pt0, pt1 = self.name2id[link[0]], self.name2id[link[1]]

         if not self.mode_coco:
            if (pt0==3 and pt1==5) or (pt0==4 and pt1==6):
               continue
            if (pt0==5 and pt1==11) or (pt0==6 and pt1==12):
               continue
         if vis_kpt[pt0] and vis_kpt[pt1]:
            kpt0 = keypoints[pt0]
            kpt1 = keypoints[pt1]

            img = cv2.line(img, (int(kpt0[0]), int(kpt0[1])),
                           (int(kpt1[0]), int(kpt1[1])),
                           link_color,
                           thickness=line_width)

      for i in range(n):
         kpt_info = self.keypoint_info[i]
         kpt_color = tuple(kpt_info['color'])
         kpt = keypoints[i]
         if vis_kpt[i]:
               img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius),
                              kpt_color, -1)
               
      return img

