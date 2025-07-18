import cv2
import math
import copy
import torch
import random
import logging
import numpy as np


class AugmentationSpatial:
    def __init__(self,device='gpu'):
        self.device = device

    def _random_pass(self, prob: float) -> bool:
        return random.random() < prob

    def _rotate_torch(self, origin, points, angle: float):
        if isinstance(origin, tuple):
            ox, oy = origin
        else:
            ox, oy = origin[:, 0].unsqueeze(1), origin[:, 1].unsqueeze(1)

        rotated = torch.zeros_like(points)#.to(device=self.device)
        rotated[:, :, 0] = ox + torch.cos(angle) * (points[:, :, 0] - ox) - torch.sin(angle) * (points[:, :, 1] - oy)
        rotated[:, :, 1] = oy + torch.sin(angle) * (points[:, :, 0] - ox) + torch.cos(angle) * (points[:, :, 1] - oy)
        return rotated

    def augment_rotate(self, keypoints: torch.Tensor, angle_range: tuple, origin=(0.5, 0.5)) -> torch.Tensor:
        angle = torch.tensor(math.radians(random.uniform(*angle_range)))
        return self._rotate_torch(origin, keypoints, angle)

    def normalize_keypoints(self,keypoints_np):
        """
        """

        if keypoints_np.ndim == 2:
            keypoints_np = keypoints_np[None, ...]  # shape: (1, N, 2)

        x = keypoints_np[..., 0]
        y = keypoints_np[..., 1]

        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
    
        x_range = max(x_max - x_min, 1e-5)
        y_range = max(y_max - y_min, 1e-5)

        # Normalize
        x_norm = (x - x_min) / x_range
        y_norm = (y - y_min) / y_range

        normed = np.stack([x_norm, y_norm], axis=-1)

        return normed, (x_min, x_range), (y_min, y_range)


    def denormalize_keypoints(normed_kp, kp_min, kp_range):
        return normed_kp * kp_range + kp_min

    def augment_shear(self, keypoints: torch.Tensor, shear_type: str, squeeze_ratio: tuple, p= 0.5) -> torch.Tensor:
        src = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.float32)
        keypoints_np = keypoints.cpu().numpy().astype(np.float32)
        mini_ = np.min(keypoints_np)
        maxi_ = np.max(keypoints_np)

        if shear_type == "squeeze":
            move_l = random.uniform(*squeeze_ratio)
            move_r = random.uniform(*squeeze_ratio)

            if random.random() > p:
                dest = np.array([[0 + move_l, 1], [1 - move_r, 1], [0 + move_l, 0], [1 - move_r, 0]], dtype=np.float32)
            else:
                dest = np.array([[0, 1 + move_l], [1, 1 + move_l], [0, 0 - move_r], [1, 0 - move_r]], dtype=np.float32)

        elif shear_type == "perspective":
            mini_ = np.min(keypoints_np)
            maxi_ = np.max(keypoints_np)
            if mini_ <0 or maxi_>1:
                keypoints_np, kp_min, kp_range = self.normalize_keypoints(keypoints_np)

            dest = np.array([
                [0 + random.uniform(*squeeze_ratio), 1 + random.uniform(*squeeze_ratio)],
                [1 + random.uniform(*squeeze_ratio), 1 + random.uniform(*squeeze_ratio)],
                [0 + random.uniform(*squeeze_ratio), 0 + random.uniform(*squeeze_ratio)],
                [1 + random.uniform(*squeeze_ratio), 0 + random.uniform(*squeeze_ratio)]
            ], dtype=np.float32)

        elif shear_type == "zoom":
            move_xy = random.uniform(*squeeze_ratio)

            x_max, y_max = keypoints_np[0].max(axis=0)
            x_min, y_min = keypoints_np[0].min(axis=0)

            width  = x_max - x_min
            height = y_max - y_min
            size   = max(width, height)
            MIN_SIZE = 0.3
            MAX_SIZE = 1

            # Avoid zoom-in if already too large
            if size >= MAX_SIZE and move_xy > 0:
                move_xy = 0.0

            # Avoid zoom-out if already too small
            if size <= MIN_SIZE and move_xy < 0:
                move_xy = 0.0

            # Also protects against overflow outside [0, 1]
            safe_margin = 0.175
            if ((y_max)*(1+move_xy))>(1+safe_margin):
                move_xy = (1+safe_margin)/(y_max)-1
            if ((height)*(1+2*move_xy))<MIN_SIZE:
                move_xy = ((MIN_SIZE)/(height)-1)/2
            dest = np.array([
                [0 - move_xy, 1 + move_xy], [1 + move_xy, 1 + move_xy],
                [0 - move_xy, 0 - move_xy], [1 + move_xy, 0 - move_xy]
            ], dtype=np.float32)
        else:
            logging.error(f"Invalid shear type: {shear_type}")
            return keypoints

        mtx = cv2.getPerspectiveTransform(src, dest)
        transformed = cv2.perspectiveTransform(keypoints_np, mtx)
        #transformed, kp_min, kp_range = self.normalize_keypoints(transformed)

        mini = np.min(transformed)
        maxi = np.max(transformed)
        mini_ = np.min(keypoints_np)
        maxi_ = np.max(keypoints_np)

        if mini < -1000 or maxi > 1000:
            print("[Warning SHEAR] For inputs_total has extreme values.")
            if mini < -100 or maxi > 100:
                print(f"[Warning SHEAR] Extreme values detected in inputs")
        if mini < -5 or maxi > 5:
            transformed = keypoints_np
    
        return torch.from_numpy(transformed)#.to(device=self.device)



class Augmentation:
    def __init__(self, device='gpu'):
        self.aug_spatial  = AugmentationSpatial(device)
        self.device = device

        # Diccionarios de augmentaciones espaciales
        self.spatial_augmentations = {
            0: self._rotate,
            1: self._shear_perspective,
            2: self._shear_squeeze,
            3: self._zoom,
        }
            
    # --------- SPATIAL FUNCTIONS ----------
    def _rotate(self, x): return self.aug_spatial.augment_rotate(x, angle_range=(-20, 20))
    def _shear_perspective(self, x): return self.aug_spatial.augment_shear(x, shear_type="perspective", squeeze_ratio=(-0.3, 0.3))
    def _shear_squeeze(self, x): return self.aug_spatial.augment_shear(x, shear_type="squeeze", squeeze_ratio=(0.3, -0.3))
    def _zoom(self, x): return self.aug_spatial.augment_shear(x, shear_type="zoom", squeeze_ratio=(-0.5, 0.5))

    # --------- APPLY METHODS ----------
    def apply_spatial(self, selected_aug: int, depth_map_original: torch.Tensor) -> torch.Tensor:
        func = self.spatial_augmentations.get(selected_aug, lambda x: x)
        return func(depth_map_original)


if __name__ == "__main__":
    pass