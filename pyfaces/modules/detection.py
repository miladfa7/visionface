from typing import Union, List
import numpy as np

# pyfaces modules 
from pyfaces.models.Detector import Detector, DetectedFace
from pyfaces.modules.modeling import build_model
from pyfaces.commons.image_utils import load_images, validate_images

def detect_faces(
    images: Union[str, np.ndarray, List[np.ndarray], List[str]],
    detector_backbone: str = "mediapipe",
) -> List[List[DetectedFace]]:
    """
        Detect faces in one or more images using the specified detector backbone.

        Parameters
        ----------
        images : Union[str, np.ndarray, List[str], List[np.ndarray]]
                A single image or a list of images. Each image can be either a file path (str)
                or an image array.

        detector_backbone : str, optional
                Name of the face detection backend to use. Default is "mediapipe".

        Returns
        -------
        List[List[DetectedFace]]: 
                A list where each element is a list of DetectedFace objects for the corresponding input image.
    """
    # Load input images
    loaded_images = load_images(images)

    # Validate input images for model processing
    validated_images = validate_images(loaded_images)
    
    # Build face detector
    face_detector = build_model(detector_backbone, "face_detection")

    # Run face detector on input images
    detections = face_detector.detect_faces(validated_images)

    return detections


def detect_faces_with_prompt(
    image_path: Union[str, np.ndarray],
    promtp: Union[str, List[str]],
    detector_backbone: str = "yoloe-medium",
) -> List:
   
    img = load_images(image_path)
    if img is None:
        raise ValueError("Input image is None. Please provide a valid image.")
    height, width, _ = img.shape
    # Build face detector
    face_detector = build_model(detector_backbone, "face_detection")
    detected_faces = face_detector.detect_faces_with_prompt(img, promtp)
    return detected_faces

