"""
Video to Pose Estimation Module

This module provides functionality for extracting whole-body pose estimations from video files
using various backbone models (RTMPose, VitPose, MediaPipe).

Author: Cristian Lazo Quispe
License: MIT
"""

import os
import sys
import math
import shutil
import subprocess
import json
from typing import Dict, Tuple, Optional, Union
from pathlib import Path
from enum import Enum

import numpy as np
import cv2
import torch

# Add src/ to Python path
current_file = Path(__file__).resolve()
src_path = current_file.parents[2]
sys.path.insert(0, str(src_path))

from wholebodypose.utils.vision import DrawerPose
from wholebodypose.utils.files import create_directory

# Optional video conversion (only if ffmpeg installed)
try:
    from .converters import convert_mp4_to_mov, reencode_for_web
    HAS_VIDEO_CONVERSION = True
except ImportError:
    HAS_VIDEO_CONVERSION = False


# Constants
class BackgroundColor(Enum):
    """Predefined background colors for visualization (BGR format)."""
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)


class RotationAngle(Enum):
    """Video rotation angles."""
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = 270
    ROTATE_NEG_90 = -90
    ROTATE_NEG_180 = -180
    ROTATE_NEG_270 = -270


DEFAULT_FPS = 15
DEFAULT_LINE_WIDTH = 2
DEFAULT_RADIUS = 1
DEFAULT_KPT_THRESHOLD = 0.5
VIDEO_CODEC_MP4 = 'avc1'#'mp4v'
MIN_MODE_COUNT = 4
MODE_TOLERANCE = 1.05
LOWER_THRESHOLD_RATIO = 0.75
UPPER_THRESHOLD_RATIO = 1.33


class VideoMetadataExtractor:
    """Handles extraction of video metadata using ffprobe."""
    
    @staticmethod
    def get_rotation_angle(filepath: str) -> int:
        """
        Extract video rotation angle from metadata using ffprobe.
        
        Args:
            filepath: Path to the video file
            
        Returns:
            Rotation angle in degrees (0, 90, 180, 270), defaults to 0 if not found
        """
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                filepath
            ]
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                check=True
            )
            metadata = json.loads(result.stdout)

            for stream in metadata.get("streams", []):
                if stream.get("codec_type") == "video":
                    side_data_list = stream.get("side_data_list", [])
                    for item in side_data_list:
                        if "rotation" in item:
                            return int(item["rotation"])
            return 0
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            print(f"[WARNING] Failed to extract rotation metadata: {e}")
            return 0
        except Exception as e:
            print(f"[WARNING] Unexpected error reading rotation: {e}")
            return 0


class VideoProcessor:
    """Handles video processing operations."""
    
    @staticmethod
    def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
        """
        Rotate a frame based on the rotation angle.
        
        Args:
            frame: Input frame (numpy array)
            rotation: Rotation angle in degrees
            
        Returns:
            Rotated frame
        """
        if rotation in (RotationAngle.ROTATE_90.value, RotationAngle.ROTATE_NEG_270.value):
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation in (RotationAngle.ROTATE_180.value, RotationAngle.ROTATE_NEG_180.value):
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation in (RotationAngle.ROTATE_270.value, RotationAngle.ROTATE_NEG_90.value):
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame
    
    @staticmethod
    def create_solid_color_frame(color: Tuple[int, int, int], width: int, height: int) -> np.ndarray:
        """
        Create a solid color frame.
        
        Args:
            color: BGR color tuple
            width: Frame width
            height: Frame height
            
        Returns:
            Solid color frame as numpy array
        """
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = color
        return frame
    
    @staticmethod
    def get_background_color(color_name: Optional[str]) -> Optional[Tuple[int, int, int]]:
        """
        Convert color name to BGR tuple.
        
        Args:
            color_name: Name of the color (red, blue, green, white, black)
            
        Returns:
            BGR color tuple or None if color_name is None
        """
        if color_name is None:
            return None
        
        color_map = {
            "red": BackgroundColor.RED.value,
            "blue": BackgroundColor.BLUE.value,
            "green": BackgroundColor.GREEN.value,
            "white": BackgroundColor.WHITE.value,
            "black": BackgroundColor.BLACK.value,
        }
        return color_map.get(color_name.lower(), BackgroundColor.BLACK.value)


class KeypointCleaner:
    """Handles cleaning and filtering of keypoint predictions."""
    
    @staticmethod
    def clean_outliers(keypoints: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Remove outlier keypoints using mode-based filtering.
        
        Args:
            keypoints: Array of shape (N, 2) containing x,y coordinates
            
        Returns:
            Cleaned keypoints as torch tensor
        """
        if isinstance(keypoints, (np.ndarray, np.generic)):
            keypoints = torch.from_numpy(keypoints)
        
        rounded = torch.round(keypoints * 1000) / 1000
        
        unique_x, counts_x = torch.unique(rounded[:, 0], return_counts=True)
        mode_idx_x = torch.argmax(counts_x)
        mode_val_x = unique_x[mode_idx_x].item() * MODE_TOLERANCE
        mode_count_x = counts_x[mode_idx_x].item()
        
        unique_y, counts_y = torch.unique(rounded[:, 1], return_counts=True)
        mode_idx_y = torch.argmax(counts_y)
        mode_val_y = unique_y[mode_idx_y].item() * MODE_TOLERANCE
        mode_count_y = counts_y[mode_idx_y].item()
        
        avg_x = torch.mean(rounded[:, 0])
        avg_y = torch.mean(rounded[:, 1])
        
        if (mode_count_x >= MIN_MODE_COUNT and mode_count_y >= MIN_MODE_COUNT and 
            mode_val_x < avg_x * LOWER_THRESHOLD_RATIO and 
            mode_val_y < avg_y * LOWER_THRESHOLD_RATIO):
            mask = torch.logical_and(
                rounded[:, 0] <= mode_val_x, 
                rounded[:, 1] <= mode_val_y
            )
            keypoints[mask, :] = 0
        
        if (mode_count_x >= MIN_MODE_COUNT and mode_count_y >= MIN_MODE_COUNT and 
            mode_val_x > avg_x * UPPER_THRESHOLD_RATIO and 
            mode_val_y < avg_y * LOWER_THRESHOLD_RATIO):
            mask = torch.logical_and(
                rounded[:, 0] <= mode_val_x, 
                rounded[:, 1] <= mode_val_y
            )
            keypoints[mask, :] = 0
        
        return keypoints


class PersonSelector:
    """Handles selection of the primary person from multi-person detections."""
    
    @staticmethod
    def select_largest_person(keypoints: np.ndarray, scores: np.ndarray) -> int:
        """
        Select the person with the largest bounding box area from detections.
        
        Args:
            keypoints: Array of shape (num_people, num_keypoints, 2)
            scores: Array of shape (num_people, num_keypoints) containing confidence scores
            
        Returns:
            Index of the person with the largest visible area
        """
        areas = []
        
        for i in range(keypoints.shape[0]):
            visible_mask = scores[i] > 0
            
            if visible_mask.sum() == 0:
                areas.append(0)
                continue
            
            x_coords = keypoints[i, visible_mask, 0]
            y_coords = keypoints[i, visible_mask, 1]
            
            area = (x_coords.max() - x_coords.min()) * (y_coords.max() - y_coords.min())
            areas.append(area)
        
        return int(np.argmax(areas))


class Video2Pose:
    """
    Main class for video-to-pose estimation pipeline.
    
    Supports multiple pose estimation backends:
    - RTMPose: Fast real-time pose estimation
    - VitPose: Transformer-based pose estimation
    - MediaPipe: Google's lightweight pose solution
    """
    
    def __init__(self, mode_coco: bool = True, model_name: str = "rtmpose") -> None:
        """
        Initialize the Video2Pose estimator.
        
        Args:
            mode_coco: If True, uses COCO Wholebody format. If False, adds custom
                      points (middle_chest, middle_hip) and links.
            model_name: Backbone model to use ('rtmpose', 'vitpose', 'mediapipe')
        """
        self.mode_coco = mode_coco
        self.model_name = model_name
        self.draw_skeleton = DrawerPose(mode_coco=mode_coco)
        
        self.model = self._initialize_model()
        
        self.metadata_extractor = VideoMetadataExtractor()
        self.video_processor = VideoProcessor()
        self.keypoint_cleaner = KeypointCleaner()
        self.person_selector = PersonSelector()
    
    def _initialize_model(self):
        """
        Initialize the pose estimation model based on model_name.
        
        Returns:
            Initialized pose model instance
        """
        if self.model_name == "mediapipe":
            from wholebodypose.models.mediapipe.model import MediapipeModel
            return MediapipeModel(
                mode_coco=True, 
                use_thresholding=True, 
                kpt_thr=DEFAULT_KPT_THRESHOLD
            )
        
        elif self.model_name == "vitpose":
            from wholebodypose.models.vitpose.model import VITPoseModel
            return VITPoseModel(
                device='cuda',
                model_name='ViTPose+_huge_coco_wholebody',
                use_thresholding=True,
                kpt_thr=DEFAULT_KPT_THRESHOLD
            )
        
        else:
            from wholebodypose.models.rtmpose.model import RTMPoseModel
            return RTMPoseModel(
                mode='performance',
                backend='onnxruntime',
                device='cuda',
                use_thresholding=True,
                filter_noise=True,
                kpt_thr=2.5
            )
    
    def convert_mp4_to_mov(
        self, 
        filepath_mp4: str, 
        folder_results: str, 
        remove_mp4: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Convert MP4 video to MOV format.
        
        Args:
            filepath_mp4: Path to input MP4 file
            folder_results: Folder to save the MOV file
            remove_mp4: If True, removes the original MP4 file
            
        Returns:
            Tuple of (success_status, output_filepath)
        """
        if not HAS_VIDEO_CONVERSION:
            raise ImportError(
                "Video conversion requires ffmpeg. "
                "Install with: pip install wholebodypose[video]"
            )
        
        status, filepath_mov = convert_mp4_to_mov(filepath_mp4, folder_results)
        
        if remove_mp4 and status:
            try:
                os.remove(filepath_mp4)
            except OSError as e:
                print(f"[WARNING] Failed to remove MP4 file: {e}")
        
        return status, filepath_mov
    
    def _copy_input_video(self, filepath: str, folder_results: str) -> str:
        """
        Create a copy of the input video in the results folder.
        
        Args:
            filepath: Original video path
            folder_results: Destination folder
            
        Returns:
            Path to the copied video
        """
        filename = Path(filepath).stem
        copied_path = os.path.join(folder_results, f"{filename}_input.mp4")
        shutil.copy(filepath, copied_path)
        return copied_path
    
    def _initialize_video_capture(self, filepath: str) -> Tuple[cv2.VideoCapture, Dict]:
        """
        Initialize video capture and extract video properties.
        
        Args:
            filepath: Path to video file
            
        Returns:
            Tuple of (VideoCapture object, properties dict)
        """
        vid = cv2.VideoCapture(filepath)
        
        if not vid.isOpened():
            raise ValueError(f"Failed to open video file: {filepath}")
        
        properties = {
            'filepath': filepath,
            'width': int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(vid.get(cv2.CAP_PROP_FPS)),
            'total_frames': int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        
        properties['duration'] = properties['total_frames'] / properties['fps']
        
        return vid, properties
    
    def _get_rotated_frame_dimensions(
        self, 
        vid: cv2.VideoCapture, 
        rotation: int
    ) -> Tuple[int, int]:
        """
        Get frame dimensions after applying rotation.
        
        Args:
            vid: VideoCapture object
            rotation: Rotation angle in degrees
            
        Returns:
            Tuple of (width, height) after rotation
        """
        ret, frame = vid.read(cv2.CAP_DSHOW)
        if not ret or frame is None:
            raise ValueError("Failed to read first frame from video")
        
        frame = self.video_processor.rotate_frame(frame, rotation)
        height, width = frame.shape[:2]
        
        return width, height
    
    def _initialize_video_writer(
        self, 
        filepath: str, 
        folder_results: str, 
        width: int, 
        height: int,
        fps: int = DEFAULT_FPS,

    ) -> Tuple[cv2.VideoWriter, str]:
        """
        Initialize video writer for output.
        
        Args:
            filepath: Original video filepath
            folder_results: Output folder
            width: Frame width
            height: Frame height
            
        Returns:
            Tuple of (VideoWriter object, output filepath)
        """
        filename_base = Path(filepath).stem
        output_path = os.path.join(
            folder_results, 
            f"{filename_base}_{self.model_name}.mp4"
        )
        
        #writer = cv2.VideoWriter(
        #    output_path,
        #    cv2.VideoWriter_fourcc(*VIDEO_CODEC_MP4),
        #    DEFAULT_FPS,
        #    (width, height)
        #)


        # Try web-compatible codecs
        web_codecs = [
            ('avc1', 'H.264 (avc1)'),
            ('H264', 'H.264 (H264)'),
            ('X264', 'H.264 (X264)'),
            ('MJPG', 'Motion JPEG'),  # Web-compatible fallback
        ]
        
        writer = None
        codec_used = None
        
        for codec, name in web_codecs:
            test_writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*codec),
                fps,
                (width, height)
            )
            
            if test_writer.isOpened():
                writer = test_writer
                codec_used = codec
                print(f"[INFO] Using {name} codec")
                break
            else:
                test_writer.release()        

        return writer, output_path
    
    def reencode_video_for_web(
        self, 
        input_path: str, 
        output_folder: str
    ) -> Tuple[bool, str]:
        """
        Re-encode video with web-friendly settings using H.264 codec.
        
        Args:
            input_path: Path to input video
            output_folder: Folder for output video
            
        Returns:
            Tuple of (success_status, output_path)
        """
        if not HAS_VIDEO_CONVERSION:
            raise ImportError(
                "Video conversion requires ffmpeg. "
                "Install with: pip install wholebodypose[video]"
            )
        
        return reencode_for_web(input_path, output_folder)
    
    def predict(
        self,
        filepath: str,
        folder_results: Optional[str] = None,
        show: bool = False,
        line_width: int = DEFAULT_LINE_WIDTH,
        radius: int = DEFAULT_RADIUS,
        save_as_mov: bool = False,
        background_plot: Optional[str] = None,
        multiperson: bool = False,
        video_processing_fps: int = None,
        output_video_fps: int = DEFAULT_FPS
    ) -> Dict:
        """
        Run pose estimation on a video file.
        
        Args:
            filepath: Path to input video file
            folder_results: Output folder for results (if None, no video is saved)
            show: If True, displays video during processing
            line_width: Line thickness for skeleton visualization
            radius: Radius of keypoint circles
            save_as_mov: If True, converts output to MOV format
            background_plot: Background color name ('red', 'blue', 'green', 'white', 'black')
            
        Returns:
            Dictionary containing:
                - filepath: Input video path
                - width, height: Video dimensions
                - fps: Frames per second
                - duration: Video duration in seconds
                - n_frames: Number of processed frames
                - total_keypoints: Array of keypoints (n_frames, n_keypoints, 2)
                - total_scores: Array of confidence scores (n_frames, n_keypoints)
                - filepath_result: Output video path (if folder_results is not None)
        """
        if save_as_mov and not HAS_VIDEO_CONVERSION:
            raise ImportError(
                "Video conversion requires ffmpeg. "
                "Install with: pip install wholebodypose[video]"
            )
        
        if folder_results:
            create_directory(folder_results)
            self._copy_input_video(filepath, folder_results)
        
        rotation = self.metadata_extractor.get_rotation_angle(filepath)
        print(f"[INFO] Detected rotation: {rotation}°")
        
        background_color = self.video_processor.get_background_color(background_plot)
        
        vid, results = self._initialize_video_capture(filepath)
        original_fps = results['fps']
        
        if video_processing_fps is None:
            video_processing_fps = original_fps

        if output_video_fps is None:
            output_video_fps = original_fps


        frame_skip = max(1, math.ceil(original_fps / video_processing_fps))
        print(f"[INFO] Original          FPS: {original_fps}")
        print(f"Processing every {frame_skip} frame(s)")
        print(f"[INFO] Target processing FPS: {video_processing_fps}")
        print(f"[INFO] Output video      FPS: {output_video_fps}")

        width, height = self._get_rotated_frame_dimensions(vid, rotation)
        results['width'] = width
        results['height'] = height
        
        #set fps of video to default fps
        vid.set(cv2.CAP_PROP_FPS, DEFAULT_FPS)
        #
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        writer = None
        if folder_results:
            writer, output_path = self._initialize_video_writer(
                filepath, folder_results, width, height, output_video_fps
            )
            results['filepath_result'] = output_path
        
        if show:
            cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
        
        keypoints_list = []
        scores_list = []
        frame_count = 0
        processed_count = 0

        while True:
            ret, frame = vid.read()
            
            if not ret or frame is None:
                break
            
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue            

            h, w = frame.shape[:2]
            new_w = 640
            new_h = int(h * (new_w / w))

            frame = cv2.resize(frame, (new_w, new_h))            

            frame = self.video_processor.rotate_frame(frame, rotation)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keypoints, scores = self.model.predict(frame_rgb)
            
            if background_color is not None:
                frame = self.video_processor.create_solid_color_frame(
                    background_color, width, height
                )
            
            if not multiperson:
                person_idx = self.person_selector.select_largest_person(keypoints, scores)
                keypoints = keypoints[person_idx, :, :]
                scores = scores[person_idx, :]

            keypoints_list.append(keypoints)
            scores_list.append(scores)
            
            if not multiperson:
                keypoints[:, :] = self.keypoint_cleaner.clean_outliers(keypoints[:, :])
            else:
                for idx in range(keypoints.shape[0]):
                    keypoints[idx, :, :] = self.keypoint_cleaner.clean_outliers(
                        keypoints[idx, :, :])
            
            frame = self.draw_skeleton(
                frame,
                np.array(keypoints) if multiperson else np.array([keypoints]),
                np.array(scores) if multiperson else np.array([scores]),
                kpt_thr=DEFAULT_KPT_THRESHOLD,
                line_width=line_width,
                radius=radius
            )
            
            if show:
                cv2.imshow('Pose Estimation', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if writer is not None:
                writer.write(frame)
            
            frame_count += 1
            processed_count += 1
        
        print(f"[INFO] Processed {processed_count}/{frame_count} frames")
        
        
        vid.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        
        results['n_frames'] = processed_count
        results['total_keypoints'] = keypoints_list
        results['total_scores']    = scores_list
                
        if folder_results and writer is not None:
            success, web_path = True,output_path #self.reencode_video_for_web(output_path, folder_results)
            if success:
                results['filepath_result'] = web_path
            
            if save_as_mov:
                success, mov_path = self.convert_mp4_to_mov(
                    results['filepath_result'], 
                    folder_results, 
                    remove_mp4=True
                )
                if success:
                    results['filepath_result'] = mov_path
                    print(f"[INFO] Saved MOV output: {mov_path}")
        
        return results