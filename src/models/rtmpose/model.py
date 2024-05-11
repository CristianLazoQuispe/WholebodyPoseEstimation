from rtmlib import Wholebody, draw_skeleton
import numpy as np
import torch

class RTMPoseModel:
    def __init__(self, device='cuda', backend='onnxruntime', 
                 mode='performance', to_openpose=False, threshold=2.55,filtering=False) -> None:
        """
        Initialize the RTMPoseModel.

        Args:
            device (str): The device to run the model on. Options: 'cpu', 'cuda'. Default: 'cuda'.
            backend (str): The backend to use for inference. Options: 'opencv', 'onnxruntime', 'openvino'. Default: 'onnxruntime'.
            mode (str): The mode of the model. Options: 'performance', 'lightweight', 'balanced'. Default: 'performance'.
            to_openpose (bool): Whether to convert the output to OpenPose-style. Default: False.
            threshold (float): The threshold for pose detection. Default: 2.55.
        """
        self.device = device
        self.backend = backend
        self.mode = mode
        self.to_openpose = to_openpose
        self.threshold = threshold
        self.filtering = filtering

        # Check if CUDA is available if device is set to 'cuda'
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA is not available. Falling back to CPU.")
            self.device = 'cpu'

        self.model = Wholebody(to_openpose=self.to_openpose, 
                               mode=self.mode, backend=self.backend,
                               device=self.device)
        
        self.foot_left_ids = [11,13,15]
        self.foot_right_ids = [12,14,16]
        self.hand_right_ids = [6,8,10,112]
        self.hand_left_ids = [5,7,9,91]

    def predict(self,frame_rgb):
        keypoints, scores = self.model(frame_rgb)
        if self.filtering:
            keypoints,scores = self.filter_scores(keypoints,scores)
        return keypoints, scores

    def filter_ids(self,list_ids,list_body,list_ids_remove,scores,keypoints):
        list_body= list(list_body)
        cnt_ids = np.sum(scores[0][list_ids] > self.threshold)
        if cnt_ids != len(list_ids):
            scores[0][list_body]= 0 
            #keypoints[0][list_body] = 0
            
            scores[0][list_ids_remove]= 0 
            #keypoints[0][list_ids_remove] = 0
        return keypoints,scores

    def filter_scores(self,keypoints,scores):
        keypoints,scores = self.filter_ids(self.foot_left_ids,range(17,20),[15],
                                 scores,keypoints)
        keypoints,scores = self.filter_ids(self.foot_right_ids,range(20,23),[16],
                                 scores,keypoints)

        keypoints,scores = self.filter_ids(self.hand_right_ids,range(112,133),[10],
                                 scores,keypoints)
        keypoints,scores = self.filter_ids(self.hand_left_ids,range(91,112),[9],
                                 scores,keypoints)
        return keypoints,scores



