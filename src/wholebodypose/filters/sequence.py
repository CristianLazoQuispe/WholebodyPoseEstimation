"""Keypoint sequence filtering utilities for pose-based applications."""

import numpy as np


class PoseSequenceFilter:
    """
    Filter pose keypoint sequences based on various quality criteria.

    Useful for sign language recognition, gesture detection, and action recognition
    where temporal consistency and hand visibility are important.
    """

    def __init__(self, left_hand_indices: np.ndarray | None = None, right_hand_indices: np.ndarray | None = None):
        """
        Initialize the sequence filter.

        Args:
            left_hand_indices: Indices of left hand keypoints (e.g., [91, 92, ..., 111])
            right_hand_indices: Indices of right hand keypoints (e.g., [112, 113, ..., 132])
        """
        self.left_hand_indices = left_hand_indices
        self.right_hand_indices = right_hand_indices

    def filter_invalid_frames(self, keypoints: np.ndarray, threshold: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove frames with too many invalid (zero-valued) keypoints.

        Args:
            keypoints: Keypoint array of shape [T, K, 2] where:
                      T = number of frames
                      K = number of keypoints
                      2 = (x, y) coordinates
            threshold: Maximum allowed ratio of invalid keypoints per frame (0.0 to 1.0)

        Returns:
            Tuple of:
                - filtered_keypoints: Array with invalid frames removed
                - valid_mask: Boolean mask indicating which frames were kept

        Example:
            >>> keypoints = np.random.rand(100, 133, 2)
            >>> filtered, mask = filter.filter_invalid_frames(keypoints, threshold=0.7)
            >>> print(f"Kept {mask.sum()}/{len(mask)} frames")
        """
        is_invalid = np.all(keypoints <= 0, axis=-1)  # Shape: [T, K]
        invalid_ratio = np.mean(is_invalid, axis=1)  # Shape: [T]

        valid_mask = invalid_ratio < threshold
        filtered_keypoints = keypoints[valid_mask]

        # Ensure at least 2 frames remain
        if len(filtered_keypoints) < 2:
            filtered_keypoints = keypoints[:2]
            valid_mask = np.zeros(len(keypoints), dtype=bool)
            valid_mask[:2] = True

        return filtered_keypoints, valid_mask

    def filter_by_motion(self, keypoints: np.ndarray, motion_threshold: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove frames with insufficient motion between consecutive frames.

        Useful for removing static frames or pauses in gesture/sign sequences.

        Args:
            keypoints: Keypoint array of shape [T, K, 2]
            motion_threshold: Minimum average pixel distance between consecutive frames

        Returns:
            Tuple of:
                - filtered_keypoints: Array with low-motion frames removed
                - valid_mask: Boolean mask indicating which frames were kept
        """
        T = keypoints.shape[0]

        if T < 2:
            return keypoints, np.ones(T, dtype=bool)

        # Calculate frame-to-frame motion
        frame_diffs = keypoints[1:] - keypoints[:-1]  # [T-1, K, 2]
        distances = np.linalg.norm(frame_diffs, axis=-1)  # [T-1, K]
        avg_motion = np.mean(distances, axis=1)  # [T-1]

        # Create mask (first frame always included)
        valid_mask = np.zeros(T, dtype=bool)
        valid_mask[0] = True  # Always keep first frame
        valid_mask[1:] = avg_motion > motion_threshold

        filtered_keypoints = keypoints[valid_mask]

        # Ensure at least 2 frames remain
        if len(filtered_keypoints) < 2:
            filtered_keypoints = keypoints[:2]
            valid_mask = np.zeros(T, dtype=bool)
            valid_mask[:2] = True

        return filtered_keypoints, valid_mask

    def filter_by_hand_visibility(
        self, keypoints: np.ndarray, visibility_threshold: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove frames where both hands are essentially invisible.

        Particularly useful for sign language recognition where hand visibility is critical.

        Args:
            keypoints: Keypoint array of shape [T, K, 2]
            visibility_threshold: Minimum average magnitude per hand to be considered visible

        Returns:
            Tuple of:
                - filtered_keypoints: Array with invisible-hand frames removed
                - valid_mask: Boolean mask indicating which frames were kept

        Raises:
            ValueError: If hand indices were not provided during initialization
        """
        if self.left_hand_indices is None or self.right_hand_indices is None:
            raise ValueError(
                "Hand keypoint indices must be provided during initialization. "
                "Example: SequenceFilter(left_hand_indices=np.arange(91, 112), "
                "right_hand_indices=np.arange(112, 133))"
            )

        T = keypoints.shape[0]

        # Extract hand keypoints
        left_hand = keypoints[:, self.left_hand_indices]  # [T, num_left, 2]
        right_hand = keypoints[:, self.right_hand_indices]  # [T, num_right, 2]

        # Calculate average magnitude for each hand
        left_magnitude = np.abs(left_hand).mean(axis=(1, 2))  # [T]
        right_magnitude = np.abs(right_hand).mean(axis=(1, 2))  # [T]

        # Frame is valid if at least one hand is visible
        left_visible = left_magnitude > visibility_threshold
        right_visible = right_magnitude > visibility_threshold
        valid_mask = left_visible | right_visible

        filtered_keypoints = keypoints[valid_mask]

        # Ensure at least 2 frames remain
        if len(filtered_keypoints) < 2:
            filtered_keypoints = keypoints[:2]
            valid_mask = np.zeros(T, dtype=bool)
            valid_mask[:2] = True

        return filtered_keypoints, valid_mask

    def apply_all_filters(
        self,
        keypoints: np.ndarray,
        invalid_threshold: float = 0.7,
        motion_threshold: float = 1.0,
        visibility_threshold: float = 0.01,
        apply_hand_filter: bool = True,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Apply all filters sequentially to a keypoint sequence.

        Args:
            keypoints: Keypoint array of shape [T, K, 2]
            invalid_threshold: Threshold for invalid frame filter
            motion_threshold: Threshold for motion filter
            visibility_threshold: Threshold for hand visibility filter
            apply_hand_filter: Whether to apply hand visibility filter

        Returns:
            Tuple of:
                - filtered_keypoints: Final filtered keypoint array
                - masks: Dictionary containing masks from each filter stage
        """
        masks = {}

        # Stage 1: Remove invalid frames
        keypoints, masks["invalid"] = self.filter_invalid_frames(keypoints, invalid_threshold)

        # Stage 2: Remove low-motion frames
        keypoints, masks["motion"] = self.filter_by_motion(keypoints, motion_threshold)

        # Stage 3: Remove frames with invisible hands (optional)
        if apply_hand_filter and self.left_hand_indices is not None:
            keypoints, masks["visibility"] = self.filter_by_hand_visibility(keypoints, visibility_threshold)

        return keypoints, masks
