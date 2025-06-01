from typing import Union, List
import numpy as np

# pyfaces modules 
from pyfaces.models.Detector import Detector, DetectedFace
from pyfaces.modules.modeling import build_model
from pyfaces.commons.image_utils import load_image

def detect_faces(
    image_path: Union[str, np.ndarray],
    detector_backbone: str = "mediapipe",
) -> List:
    """
    Detect faces in an image using the specified detector backbone.

    This function takes an image path or numpy array and detects faces
    in the image using the specified detector backend.

    Parameters
    ----------
    image_path : Union[str, np.ndarray]
        Path to the image file as a string or a numpy array containing
        the image data.
    detector_backbone : str, optional
        Name of the face detection backend to use, by default "mediapipe".

    Returns
    -------
    List
        A list of detected faces. The structure depends on the detector backend used.

    Raises
    ------
    ValueError
        If the input image is None or cannot be properly loaded.
    """
    img = load_image(image_path)
    if img is None:
        raise ValueError("Input image is None. Please provide a valid image.")
    height, width, _ = img.shape
    # Build face detector
    face_detector = build_model(detector_backbone, "face_detection")
    detected_faces = face_detector.detect_faces(img)
    return detected_faces


def detect_faces_with_prompt(
    image_path: Union[str, np.ndarray],
    promtp: Union[str, List[str]],
    detector_backbone: str = "yoloeye",
) -> List:
   
    img = load_image(image_path)
    if img is None:
        raise ValueError("Input image is None. Please provide a valid image.")
    height, width, _ = img.shape
    # Build face detector
    face_detector = build_model(detector_backbone, "face_detection")
    detected_faces = face_detector.detect_faces_with_prompt(img, promtp)
    return detected_faces

