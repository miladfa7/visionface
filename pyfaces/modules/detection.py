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
        images: Union[str, np.ndarray, List[np.ndarray], List[str]],
        promtps: Union[str, List[str]],
        detector_backbone: str = "yoloe-medium",
) -> List[List[DetectedFace]]:
    """
    Detect faces in one or more images using a prompt-based detection approach.

    Parameters
    ----------
    images : Union[str, np.ndarray, List[str], List[np.ndarray]]
        A single image or a list of images. Each image can be either a file path (str)
        or an image array.

    promtps : Union[str, List[str]]
        A single prompt or a list of prompts describing the object(s) to detect.
        For example, "face".

    detector_backbone : str, optional
        Name of the detection backend to use. Default is "yoloe-medium".
        Must support prompt-based detection.

    Returns
    -------
    List[List[DetectedFace]]
        A list where each element is a list of DetectedFace objects
        for the corresponding input image. Each detection includes bounding box
        coordinates, confidence score, class name, and optionally a cropped region.
    """
   
    # Load input images
    loaded_images = load_images(images)

    # Validate input images for model processing
    validated_images = validate_images(loaded_images)

    if isinstance(promtps, str):
        promtps = [promtps]

    # if len(validated_images) != len(promtps):
    #     raise ValueError("The number of images and prompts must be the same.")
    
    # Build face detector
    face_detector = build_model(detector_backbone, "face_detection")

    # Run face detector with input images and prompts
    detected_faces = face_detector.detect_faces_with_prompt(validated_images, promtps)

    return detected_faces

