from typing import List, Tuple, Union
import numpy as np 

# Pyfaces modules
from pyfaces.models.LandmarkDetector import DetectedLandmark3D
from pyfaces.modules.modeling import build_model
from pyfaces.commons.image_utils import load_image


def detect_landmark_3d(
        image_path: Union[str, np.ndarray],
        detector_backbone: str = "mediapipe"
) -> List[DetectedLandmark3D]:
    
    img = load_image(image_path)
    
    if img is None:
        raise ValueError("Input image is None. Please provide a valid image.")
    height, width, _ = img.shape
    
    # Build landmark detection
    landmark_detector = build_model(detector_backbone, "landmark_detection")
    detected_landmarks = landmark_detector.detect_landmarks(img)
    
    return detected_landmarks
        