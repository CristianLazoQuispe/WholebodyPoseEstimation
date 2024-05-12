import cv2
import onepose
import time
from scipy.signal import savgol_filter

import torch

class VITPoseModel:
    def __init__(self, device='cuda', model_name='ViTPose+_huge_coco_wholebody',
                 kpt_thr=0.5) -> None:
        """
        Initialize the VITPoseModel.

        Args:
            device (str): The device to run the model on. Options: 'cpu', 'cuda'. Default: 'cuda'.
            model_name (str): The pose model : 'ViTPose+_base_coco_wholebody',
                'ViTPose_huge_mpii','ViTPose+_huge_coco_wholebody','ViTPose+_large_coco_wholebody'
            kpt_thr (int): threshold to filtering
        """
        self.kpt_thr=kpt_thr
        
        self.device = device
        self.model_name = model_name
        # Check if CUDA is available if device is set to 'cuda'
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA is not available. Falling back to CPU.")
            self.device = 'cpu'


        self.model  = onepose.create_model().to(self.device)
        

        keypoints = model(frame_vit)
        scores = np.moveaxis( keypoints['confidence'], 0, 1)
        keypoints = np.expand_dims(keypoints['points'], axis=0)

  
