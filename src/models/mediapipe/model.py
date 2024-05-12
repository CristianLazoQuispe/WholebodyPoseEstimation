from .map_mp2coco import MP2COCO
from typing import List,Dict
import mediapipe as mp
import copy

# Default configurations
default_face_config = {
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'max_num_faces': 1,
    'refine_landmarks':False
}
default_hands_config = {
    'min_detection_confidence': 0.3,
    'min_tracking_confidence': 0.3,
    'max_num_hands': 2,
    'model_complexity': 1
}
default_pose_config = {
    'min_detection_confidence': 0.3,
    'min_tracking_confidence': 0.3,
    'model_complexity': 2,
    'smooth_landmarks':True,
    'enable_segmentation':False,
    'smooth_segmentation':True
}

default_draw_config = {
    'circle_radius':1,
    'thickness':1,
    'color':(255,0,0)
}

class mediapipe_model:
    def __init__(self, models: List[str] = ['hands', 'face', 'pose'], image_mode: bool = True, 
                 face_config: Dict = None, hands_config: Dict = None, pose_config: Dict = None,
                 mode_coco : bool = False,use_thresholding: bool=False,kpt_thr:float=0.5) -> None:
        """
        WholeBodyPoseEstimation using mediapipe
        Args:
            models (List[str]): Models to use. Defaults to ['hand', 'face', 'pose']
            image_mode (bool): image_mode False takes frames as video stream with tracking to reduce latency
            face_config (Dict): Configuration for the face model. Defaults to None
            hands_config (Dict): Configuration for the hands model. Defaults to None
            pose_config (Dict): Configuration for the pose model. Defaults to None
            mode_coco (bool): Configuration of output format. mode_coco True returns 133 points in coco format
            use_thresholding (bool): Configuration of thresholding.
                ```
                scores[scores>=self.kpt_thr]   = 1
                scores[scores<self.kpt_thr]    = 0
                keypoints[scores<self.kpt_thr] = 0
                ```
            kpt_thr (int): threshold to filtering
        """
        self.kpt_thr = kpt_thr
        self.use_thresholding = use_thresholding
        self.image_mode = image_mode
        # initialize drawing mediapipe
        self.mp_drawing = mp.solutions.drawing_utils

        # Check if models list is valid
        valid_models = ['hands', 'face', 'pose']
        if not 1 <= len(models) <= 3 or not all(model in valid_models for model in models):
            raise ValueError("The 'models' list must have at least one element and a maximum of three elements from among 'hands', 'face', and 'pose'.")
        
        # choose models hand face pose
        self.models = list(set(valid_models)&set(models))
        
        if face_config is not None:
            default_face_config.update({k: v for k, v in face_config.items() if k in default_face_config})

        if pose_config is not None:
            default_pose_config.update({k: v for k, v in pose_config.items() if k in default_pose_config})

        if hands_config is not None:
            default_hands_config.update({k: v for k, v in hands_config.items() if k in default_hands_config})
       
        
        # Load selected models
        if 'face' in self.models:
            self.face_model  = mp.solutions.face_mesh.FaceMesh(static_image_mode=self.image_mode,
                                        min_detection_confidence=default_face_config['min_detection_confidence'], 
                                        min_tracking_confidence=default_face_config['min_tracking_confidence'],
                                        max_num_faces=default_face_config['max_num_faces'])
        
        if 'hands' in self.models:
            self.hands_model = mp.solutions.hands.Hands(static_image_mode=self.image_mode,
                                        min_detection_confidence=default_hands_config['min_detection_confidence'],    
                                        min_tracking_confidence=default_hands_config['min_tracking_confidence'],
                                        max_num_hands=default_hands_config['max_num_hands'],
                                        model_complexity=default_hands_config['model_complexity'])
        
        if 'pose' in self.models:
            self.pose_model  = mp.solutions.pose.Pose(static_image_mode=self.image_mode, 
                                        min_detection_confidence=default_pose_config['min_detection_confidence'],
                                        min_tracking_confidence=default_pose_config['min_tracking_confidence'],
                                        model_complexity=default_pose_config['model_complexity'])
        self.maper = MP2COCO()
        self.mode_coco = mode_coco

    def predict(self,frame_rgb):
        results = {}
        if 'face' in self.models:
            results['face']= self.face_model.process(frame_rgb)
        if 'hands' in self.models:
            results['hands']= self.hands_model.process(frame_rgb)
        if 'pose' in self.models:
            results['pose']= self.pose_model.process(frame_rgb)
    
        if self.mode_coco:
            keypoints, scores = self.maper.process(frame_rgb,results)
            if self.use_thresholding:
                keypoints, scores = self.thresholding(keypoints,scores)
            return keypoints,scores
        return results
    
    def thresholding(self,keypoints,scores):
        keypoints[scores<self.kpt_thr] = 0
        scores[scores<self.kpt_thr]    = 0
        scores[scores>=self.kpt_thr]   = 1
        return keypoints,scores
        
    def draw_mediapipe(self,frame_rgb,results: Dict,landmark_config:Dict=None,
                       connection_config:Dict=None):

        default_ldk_config = copy.deepcopy(default_draw_config)
        default_con_config = copy.deepcopy(default_draw_config)

        if landmark_config is not None:
            default_ldk_config.update({k: v for k, v in landmark_config.items() if k in default_ldk_config})
        if connection_config is not None:
            default_con_config.update({k: v for k, v in connection_config.items() if k in default_con_config})
                              
        frame_rgb = copy.deepcopy(frame_rgb)

        ldk_config = self.mp_drawing.DrawingSpec(circle_radius=default_ldk_config['circle_radius'],
                                                    thickness=default_ldk_config['thickness'],
                                                    color=default_ldk_config['color'])

        con_config = self.mp_drawing.DrawingSpec(circle_radius=default_con_config['circle_radius'],
                                                    thickness=default_con_config['thickness'],
                                                    color=default_con_config['color'])
                
        if 'face' in self.models:
            if results['face'].multi_face_landmarks != None:
                for faceLandmarks in results['face'].multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(frame_rgb, faceLandmarks,
                    mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec= ldk_config,
                    connection_drawing_spec = con_config
                        )
        if 'hands' in self.models:
            if results['hands'].multi_hand_landmarks != None:
                for handLandmarks in results['hands'].multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame_rgb, handLandmarks,
                                    mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec= ldk_config,
                    connection_drawing_spec = con_config
                                    )
        if 'pose' in self.models:
            if results['pose'].pose_landmarks != None:
                self.mp_drawing.draw_landmarks(frame_rgb, results['pose'].pose_landmarks,
                                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec= ldk_config,
                    connection_drawing_spec = con_config
                                               )
        return frame_rgb
    
    
        