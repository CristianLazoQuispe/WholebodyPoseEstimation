import numpy as np
import mediapipe as mp
import os
import json

# Obtener la ruta del módulo actual
module_path = os.path.dirname(os.path.abspath(__file__))



class MP2COCO:
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.visibility_threshold = self.mp_drawing._VISIBILITY_THRESHOLD
        self.presence_threshold = self.mp_drawing._PRESENCE_THRESHOLD
        self.meta_data = None
        # Construir la ruta completa del archivo JSON
        json_path = os.path.join(module_path, "meta_data.json")

        # Leer el archivo JSON
        with open(json_path, "r") as file:
            self.meta_data = json.load(file)
            
        self.mp2coco_pose = self.meta_data['meta_info']['mp2coco']['pose']
        self.mp2coco_face = self.meta_data['meta_info']['mp2coco']['face']

            
    def process(self, img_rgb,results):
        keypoints = np.zeros((133, 2), dtype=np.float32)
        scores = np.zeros((133,), dtype=np.float32)
        keypoints, scores = self.process_hand(results, keypoints, scores)
        keypoints, scores = self.process_pose(results, keypoints, scores)
        keypoints, scores = self.process_face(results, keypoints, scores)

        keypoints = self.rescale(img_rgb, keypoints)

        scores = np.array(scores).reshape((133,1))
        scores = np.moveaxis(scores, 0, 1)
        keypoints = np.expand_dims(keypoints, axis=0)

        return keypoints, scores

    def process_hand(self, results, keypoints, scores):
        if results['hands'].multi_hand_landmarks:
            left_hand_landmarks = results['hands'].multi_hand_landmarks[0]
            keypoints[91:112], scores[91:112] = self.get_xy_landmarks(left_hand_landmarks)

            if len(results['hands'].multi_hand_landmarks) == 2:
                right_hand_landmarks = results['hands'].multi_hand_landmarks[1]
                keypoints[112:133], scores[112:133] = self.get_xy_landmarks(right_hand_landmarks)
        return keypoints,scores
    def process_pose(self, results, keypoints, scores):
        if results['pose'].pose_landmarks != None:
            pose_landmarks = results['pose'].pose_landmarks
            keypoints_pose,scores_pose = self.get_xy_landmarks(pose_landmarks)

            for i, (kp, sc) in enumerate(zip(keypoints_pose, scores_pose)):
                coco_id = self.mp2coco_pose.get(str(i))
                if coco_id is not None:
                    if isinstance(coco_id, list):
                        keypoints[coco_id] = kp
                        scores[coco_id] = sc
                    else:
                        keypoints[coco_id] = kp
                        scores[coco_id] = sc
        return keypoints,scores

    def process_face(self, results, keypoints, scores):
        if results['face'].multi_face_landmarks != None:
            faces_landmarks = results['face'].multi_face_landmarks
            if len(faces_landmarks)>=0:
                face_landmark = faces_landmarks[0]
                keypoints_face,scores_face = self.get_xy_landmarks(face_landmark)
                for k,v in self.mp2coco_face.items():
                    keypoints[int(v)] = keypoints_face[int(k)]
                    scores[int(v)] = 1
        return keypoints,scores
    
    def rescale(self, img_rgb, keypoints):
        height, width, _ = img_rgb.shape
        keypoints_new = np.round(keypoints * [width, height]).astype(int)
        return keypoints_new
    
    
    def get_xy_landmarks(self, list_landmarks):
        list_points = np.zeros((len(list_landmarks.landmark), 2), dtype=np.float32)
        list_scores = np.zeros((len(list_landmarks.landmark),), dtype=np.float32)

        for idx, value in enumerate(list_landmarks.landmark):
            if ((value.HasField('visibility') and
                 value.visibility < self.visibility_threshold) or
                (value.HasField('presence') and
                 value.presence < self.presence_threshold)):
                continue

            list_points[idx] = [value.x, value.y]
            list_scores[idx] = 1

       
        return list_points, list_scores