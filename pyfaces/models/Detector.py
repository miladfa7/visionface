from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Union, List, Optional
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
    def detect_faces(self, imgs: Union[np.ndarray, List[np.ndarray]]):
        """
        Detect faces in a single image or a list of images.

        Args:
            imgs (Union[np.ndarray, List[np.ndarray]]): 
                - A single image as a NumPy array with shape (H, W, 3), or 
                - A list of such images.

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
        class_name (str): The name of the detected class (e.g., "face").
    """
    xmin: int
    ymin: int
    xmax: int 
    ymax: int 
    conf: float
    class_name: str
    cropped_face:  Optional[np.ndarray] = None

    @property
    def xyxy(self):
        """
        Returns the bounding box coordinates as a tuple (xmin, ymin, xmax, ymax).
        """
        return (self.xmin, self.ymin, self.xmax, self.ymax)
    
    @property
    def xywh(self):
        """
        Returns the bounding box coordinates as a tuple (x, y, w, h).
        """
        width = self.xmax - self.xmin
        height = self.ymax - self.ymin
        return (self.xmin, self.ymin, width, height)

    def to_dict(self):
        return {
            "xywh": self.xywh,
            "xyxy": self.xyxy,
            "conf": self.conf,
            "class_name": self.class_name
        }