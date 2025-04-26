from abc import ABC, abstractmethod
import numpy as np
from typing import Any
from dataclasses import dataclass

class Detector(ABC):
    """
    Abstract base class for a face detection system.

    This class defines the interface for building a detection model,
    running detection on images, and post-processing the results.
    Subclasses must implement all abstract methods.
    """

    def __init__(self, MODEL_ID: int = 0, MIN_CONFIDENCE: float = 0.5):
        """
        Initialize the base Detector with a confidence threshold.

        Args:
            conf (float): Minimum confidence score to consider a face detection valid. Default 0.25
        """
        self.model_id = MODEL_ID
        self.conf = MIN_CONFIDENCE

    @abstractmethod
    def build_model(self) -> Any:
        """
        Build and return the face detection model.

        This method should load or initialize the face detection model.
        Returns:
            model (Any): The model used for detection.
        """
        pass

    @abstractmethod
    def detect_faces(self, img: np.ndarray):
        """
        Detect faces in the given image.

        Args:
            img (np.ndarray): Input image as a NumPy array (H, W, C).

        Returns:
            detections (Any): Raw output of the detection model.
        """
        pass
    
    @abstractmethod
    def process_faces(self, results): 
        """
        Process the raw detections into a structured format.

        This could include bounding boxes, landmarks, confidence scores, etc.

        Args:
            results (Any): Raw model output from `detect_faces`.

        Returns:
            results (List[Any]): Processed list of face detection results in a consistent format.
        """
        pass


@dataclass
class DetectedFace:
    """
    Represents detected faces in an image.

    Attributes:
        x (int): The x-coordinate of the top-left corner of the face bounding box.
        y (int): The y-coordinate of the top-left corner of the face bounding box.
        w (int): The width of the face bounding box.
        h (int): The height of the face bounding box.
        conf (float): The confidence score of the face detection, typically between 0 and 1.
    """
    x: int
    y: int
    w: int 
    h: int 
    conf: float