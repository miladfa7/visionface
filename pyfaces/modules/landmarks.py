from typing import Any, List, Union
import numpy as np

# Pyfaces modules
from pyfaces.models.LandmarkDetector import DetectedLandmark2D, DetectedLandmark3D
from pyfaces.modules.modeling import build_model
from pyfaces.commons.image_utils import load_images, validate_images


class LandmarkDetection:
    def __init__(self, detector_backbone: str = "mediapipe") -> None:
        """
        Initialize the landmark detection with the specified backbone.
        
        Currently supported backbones:
            - "mediapipe": 3D landmark detection
            - "dlib": 2D landmark detection

        Args:
            detector_backbone: Backbone name for the landmark detector (e.g., "mediapipe", "dlib").
        """
        self.detector_backbone = detector_backbone
        self.landmark_detector = self.build_model()

    def build_model(self) -> Any:
        """
        Builds the landmark detection model based on the specified backbone.

        Returns:
            An initialized landmark detection model.
        """
        return build_model(self.detector_backbone, "landmark_detection")

    def detect_3d_landmarks(
        self,
        images: Union[str, np.ndarray, List[np.ndarray], List[str]],
    ) -> List[List[DetectedLandmark3D]]:
        """
        Detect 3D facial landmarks in one or more images using the specified detection backbone.

        Args:
            images: A single image or a list of images, each can be a file path or a NumPy array.

        Returns:
            A list of lists containing DetectedLandmark3D instances with 3D coordinates.
        """
        loaded_images = load_images(images)
        validated_images = validate_images(loaded_images)
        return self.landmark_detector.detect_landmarks(validated_images)

    def detect_landmarks(
        self,
        images: Union[str, np.ndarray, List[np.ndarray], List[str]],
    ) -> List[List[DetectedLandmark2D]]:
        """
        Detect 2D facial landmarks in one or more images using the specified detection backbone.

        Args:
            images: A single image or a list of images, each can be a file path or a NumPy array.

        Returns:
            A list of lists containing DetectedLandmark2D instances with 2D coordinates.
        """
        loaded_images = load_images(images)
        validated_images = validate_images(loaded_images)
        return self.landmark_detector.detect_landmarks(validated_images)
