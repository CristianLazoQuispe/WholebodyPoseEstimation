import cv2
import onepose
import time
from scipy.signal import savgol_filter
import numpy as np
import torch

class VITPoseModel:
    def __init__(self, device:str='cuda', model_name:str='ViTPose+_huge_coco_wholebody',
                 use_thresholding:bool=False,kpt_thr:float=2.5) -> None:
        """
        Initialize the VITPoseModel.

        Args:
            device (str): The device to run the model on. Options: 'cpu', 'cuda'. Default: 'cuda'.
            model_name (str): The pose model : 'ViTPose+_base_coco_wholebody',
                'ViTPose_huge_mpii','ViTPose+_huge_coco_wholebody','ViTPose+_large_coco_wholebody'
            use_thresholding (bool): Configuration of thresholding.
                ```
                scores[scores>=self.kpt_thr]   = 1
                scores[scores<self.kpt_thr]    = 0
                keypoints[scores<self.kpt_thr] = 0
                ```
            kpt_thr (int): threshold to filtering
        """
        self.kpt_thr=kpt_thr
        self.use_thresholding = use_thresholding
        self.device = device
        self.model_name = model_name
        # Check if CUDA is available if device is set to 'cuda'
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA is not available. Falling back to CPU.")
            self.device = 'cpu'


        self.model  = onepose.create_model(model_name=self.model_name).to(self.device)
        
    def predict(self,frame_rgb):

        keypoints = self.model(frame_rgb)
        scores    = np.moveaxis( keypoints['confidence'], 0, 1)
        keypoints = np.expand_dims(keypoints['points'], axis=0)
        if self.use_thresholding:
            keypoints,scores = self.thresholding(keypoints,scores)
        return keypoints, scores

    def thresholding(self,keypoints,scores):
        keypoints[scores<self.kpt_thr] = 0
        scores[scores<self.kpt_thr]    = 0
        scores[scores>=self.kpt_thr]   = 1
        return keypoints,scores
    

  
