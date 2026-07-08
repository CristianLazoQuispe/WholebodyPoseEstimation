"""Vision utilities for drawing pose keypoints and bounding boxes."""

import cv2
import numpy as np

from .coco133 import coco133


def valid_keypoint_mask(keypoints: np.ndarray, sentinel: float = 0.0) -> np.ndarray:
    """Boolean mask of DETECTED keypoints.

    Undetected keypoints are stored as the sentinel coordinate (both x and y equal `sentinel`, e.g.
    (0, 0) in raw image space). Use this to build a `scores` array for the drawers instead of the
    near-origin proximity heuristic `(x + y) / 2 < thr`, which breaks once the sentinels have been
    moved by centering/augmentation. Accepts (..., K, 2) and returns (..., K) bool.
    """
    x = keypoints[..., 0]
    y = keypoints[..., 1]
    return ~((x == sentinel) & (y == sentinel))


def draw_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.6,
    text_color: tuple[int, int, int] = (255, 255, 255),
    background_color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
) -> None:
    """
    Draw text with background rectangle on frame.
    
    Args:
        frame: Image to draw on
        text: Text to display
        position: (x, y) coordinates for text
        font: OpenCV font type
        scale: Font scale
        text_color: RGB color for text
        background_color: RGB color for background
        thickness: Text thickness
    """
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_width, text_height = text_size
    x, y = position
    
    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x - 10, y - text_height - 10),
        (x + text_width + 10, y + 10),
        background_color,
        -1
    )
    
    # Draw text
    cv2.putText(frame, text, position, font, scale, text_color, thickness)


def draw_bbox(
    img: np.ndarray,
    bboxes: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        img: Image to draw on
        bboxes: Array of bounding boxes with shape (N, 4) where each box is [x1, y1, x2, y2]
        color: RGB color for boxes
        thickness: Line thickness
        
    Returns:
        Image with drawn bounding boxes
    """
    for bbox in bboxes:
        cv2.rectangle(
            img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color,
            thickness
        )
    return img


class DrawerPose:
    """
    Draw pose keypoints and skeleton connections.
    
    Supports COCO Wholebody format (133 keypoints).
    """
    
    def __init__(self, mode_coco: bool = True):
        """
        Initialize pose drawer.
        
        Args:
            mode_coco: If True, uses standard COCO Wholebody format.
                      If False, adds custom middle points:
                        - middle_chest = 0.5 * (left_shoulder + right_shoulder)
                        - middle_hip = 0.5 * (left_hip + right_hip)
                      And additional links:
                        - middle_chest -> nose
                        - middle_hip -> middle_chest
        """
        self.skeleton_dict = coco133
        self.keypoint_info = self.skeleton_dict['keypoint_info']
        self.skeleton_info = self.skeleton_dict['skeleton_info']
        self.mode_coco = mode_coco
        
        # Build name to ID mapping
        self.name2id = {
            kpt_info['name']: kpt_info['id']
            for i, kpt_info in self.keypoint_info.items()
        }
        
        self.n_skeleton = 65 if self.mode_coco else 67
    
    def __call__(
        self,
        img: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        kpt_thr: float = 0.5,
        radius: int = 2,
        line_width: int = 2
    ) -> np.ndarray:
        """
        Draw pose keypoints and skeleton on image.
        
        Args:
            img: Image to draw on
            keypoints: Keypoint coordinates with shape (N, K, 2) or (K, 2)
                      where N is number of instances, K is number of keypoints
            scores: Keypoint confidence scores with shape (N, K) or (K,)
            kpt_thr: Confidence threshold for drawing keypoints
            radius: Radius of keypoint circles
            line_width: Width of skeleton lines
            
        Returns:
            Image with drawn poses
        """
        # Ensure 3D shape (N, K, 2)
        if len(keypoints.shape) == 2:
            keypoints = keypoints[None, :, :]
            scores = scores[None, :]
        
        num_instances = keypoints.shape[0]
        
        for i in range(num_instances):
            img = self._draw_single_pose(
                img, keypoints[i], scores[i], kpt_thr, radius, line_width
            )
        
        return img
    
    def _draw_single_pose(
        self,
        img: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        kpt_thr: float = 0.5,
        radius: int = 2,
        line_width: int = 2
    ) -> np.ndarray:
        """
        Draw a single pose instance.
        
        Args:
            img: Image to draw on
            keypoints: Keypoint coordinates with shape (K, 2)
            scores: Keypoint confidence scores with shape (K,)
            kpt_thr: Confidence threshold for drawing keypoints
            radius: Radius of keypoint circles
            line_width: Width of skeleton lines
            
        Returns:
            Image with drawn pose
        """
        assert len(keypoints.shape) == 2, "Keypoints must be 2D array"
        
        # Add custom middle points if not in COCO mode
        if not self.mode_coco:
            keypoints, scores = self._add_middle_points(keypoints, scores)
        
        # Determine visible keypoints
        vis_kpt = scores >= kpt_thr
        n_keypoints = len(keypoints)
        
        # Draw skeleton connections
        for i in range(self.n_skeleton):
            img = self._draw_skeleton_link(
                img, keypoints, vis_kpt, i, line_width
            )
        
        # Draw keypoints
        for i in range(n_keypoints):
            if vis_kpt[i]:
                kpt_info = self.keypoint_info[i]
                kpt_color = tuple(kpt_info['color'])
                kpt = keypoints[i]
                
                cv2.circle(
                    img,
                    (int(kpt[0]), int(kpt[1])),
                    radius,
                    kpt_color,
                    -1
                )
        
        return img
    
    def _add_middle_points(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Add custom middle chest and hip points.
        
        Args:
            keypoints: Keypoint coordinates (K, 2)
            scores: Keypoint confidence scores (K,)
            
        Returns:
            Tuple of (updated_keypoints, updated_scores)
        """
        # Middle chest = average of left and right shoulders (indices 5, 6)
        middle_chest_x = int((keypoints[5][0] + keypoints[6][0]) * 0.5)
        middle_chest_y = int((keypoints[5][1] + keypoints[6][1]) * 0.5)
        middle_chest_score = (scores[5] + scores[6]) * 0.5
        
        # Middle hip = average of left and right hips (indices 11, 12)
        middle_hip_x = int((keypoints[11][0] + keypoints[12][0]) * 0.5)
        middle_hip_y = int((keypoints[11][1] + keypoints[12][1]) * 0.5)
        middle_hip_score = (scores[11] + scores[12]) * 0.5
        
        # Append new points
        keypoints = np.append(
            keypoints,
            [[middle_chest_x, middle_chest_y], [middle_hip_x, middle_hip_y]],
            axis=0
        )
        scores = np.append(scores, [middle_chest_score, middle_hip_score])
        
        return keypoints, scores
    
    def _draw_skeleton_link(
        self,
        img: np.ndarray,
        keypoints: np.ndarray,
        vis_kpt: np.ndarray,
        link_idx: int,
        line_width: int
    ) -> np.ndarray:
        """
        Draw a single skeleton connection.
        
        Args:
            img: Image to draw on
            keypoints: Keypoint coordinates
            vis_kpt: Boolean array of visible keypoints
            link_idx: Index of skeleton link to draw
            line_width: Width of the line
            
        Returns:
            Image with drawn link
        """
        ske_info = self.skeleton_info[link_idx]
        link = ske_info['link']
        link_color = tuple(ske_info['color'])
        
        pt0 = self.name2id[link[0]]
        pt1 = self.name2id[link[1]]
        
        # Skip certain links in non-COCO mode
        if not self.mode_coco:
            # Skip neck to shoulder links
            if (pt0 == 3 and pt1 == 5) or (pt0 == 4 and pt1 == 6):
                return img
            # Skip shoulder to hip links
            if (pt0 == 5 and pt1 == 11) or (pt0 == 6 and pt1 == 12):
                return img
        
        # Draw line if both keypoints are visible
        if vis_kpt[pt0] and vis_kpt[pt1]:
            kpt0 = keypoints[pt0]
            kpt1 = keypoints[pt1]
            
            cv2.line(
                img,
                (int(kpt0[0]), int(kpt0[1])),
                (int(kpt1[0]), int(kpt1[1])),
                link_color,
                thickness=line_width
            )
        
        return img