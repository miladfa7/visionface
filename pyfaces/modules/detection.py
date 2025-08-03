from typing import Any, Union, List
import numpy as np

# pyfaces modules 
from pyfaces.models.Detector import Detector, DetectedFace
from pyfaces.modules.modeling import build_model
from pyfaces.commons.image_utils import load_images, validate_images

class FaceDetection:
    """
    detecting faces in images using a specified detection backbone.
    """

    def __init__(self, detector_backbone: str = "mediapipe") -> None:
        """
        Initializes the FaceDetection class with the specified detector backbone.

        Parameters
        ----------
        detector_backbone : str, optional
            Name of the face detection backend to use. Default is "mediapipe".
        """
        self.face_detector = self.build_model(detector_backbone)
    
    def build_model(self, model_name: str) -> Any:
        """
        Builds the face detection model based on the specified model name.

        Parameters
        ----------
        model_name : str
            The name of the face detection model to use.

        Returns
        -------
        Any
            An initialized face detection model.
        """
        return build_model(model_name, "face_detection")

    def detect_faces(
        self, 
        images: Union[str, np.ndarray, List[np.ndarray], List[str]],
        return_cropped_faces: bool = True
    ) -> List[List[DetectedFace]]:
        """
        Detect faces in one or more images using the specified detector backbone.

        Parameters
        ----------
        images : Union[str, np.ndarray, List[str], List[np.ndarray]]
                A single image or a list of images. Each image can be either a file path (str)
                or an image array.
        return_cropped_faces : bool, optional
            Whether to include cropped face images in each DetectedFace object. Default is True.

        Returns
        -------
        List[List[DetectedFace]]: 
                A list where each element is a list of DetectedFace objects for the corresponding input image.
        """
        loaded_images = load_images(images)
        validated_images = validate_images(loaded_images)
        return self.face_detector.detect_faces(validated_images, return_cropped_faces)


    def detect_faces_with_prompt(
            self,
            images: Union[str, np.ndarray, List[np.ndarray], List[str]],
            prompts: Union[str, List[str]],
            return_cropped_faces: bool = True
    ) -> List[List[DetectedFace]]:
        """
        Detect faces in one or more images using a prompt-based detection approach.

        Parameters
        ----------
        images : Union[str, np.ndarray, List[str], List[np.ndarray]]
            A single image or a list of images. Each image can be either a file path (str)
            or an image array.

        prompts : Union[str, List[str]]
            A single prompt or a list of prompts describing the object(s) to detect.
            For example, "face".
        
        return_cropped_faces : bool, optional
            Whether to include cropped face images in each DetectedFace object. Default is True.

        Returns
        -------
        List[List[DetectedFace]]
            A list where each element is a list of DetectedFace objects
            for the corresponding input image. Each detection includes bounding box
            coordinates, confidence score, class name, and optionally a cropped region.
        """
        loaded_images = load_images(images)
        validated_images = validate_images(loaded_images)

        if isinstance(prompts, str):
            prompts = [prompts]

        # Optional: enforce prompt count matching image count
        # if len(validated_images) != len(prompts):
        #     raise ValueError("The number of images and prompts must be the same.")

        return self.face_detector.detect_faces_with_prompt(validated_images, prompts, return_cropped_faces)

