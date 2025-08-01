from typing import List, Tuple, Union
import numpy as np 

# Pyfaces modules
from pyfaces.models.LandmarkDetector import DetectedLandmark3D
from pyfaces.modules.modeling import build_model
from pyfaces.commons.image_utils import load_images, validate_images


def detect_3d_landmarks(
        images: Union[str, np.ndarray, List[np.ndarray], List[str]],
        detector_backbone: str = "mediapipe"
) -> List[List[DetectedLandmark3D]]:
    """
        Detect 3D facial landmarks in one or more images using the specified detection backbone.

        Parameters
        ----------
        images : Union[str, np.ndarray, List[str], List[np.ndarray]]
                A single image or a list of images. Each image can be either:
                        - A file path (str) to an image file
                        - A NumPy array representing the image

        detector_backbone : str, optional
                The name of the face landmark detection model to use. 
                Supported options typically include "mediapipe", etc. 
                Default is "mediapipe".

        Returns
        -------
        List[List[DetectedLandmark3D]]
                A list of DetectedLandmark3D instances containing the 3D coordinates (x, y, z)
                for each detected facial landmark, along with optional landmark name and confidence score.
        """
    # Load input images
    loaded_images = load_images(images)
    
    # Validate input images for model processing
    validated_images = validate_images(loaded_images)

    # Build landmark detection
    landmark_detector = build_model(detector_backbone, "landmark_detection")

    # Run landmark detector on input images
    detected_landmarks = landmark_detector.detect_landmarks(validated_images)
    
    return detected_landmarks
        