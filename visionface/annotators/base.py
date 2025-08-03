from abc import ABC, abstractmethod
from typing import Union, List, Optional, Tuple
import numpy as np
from PIL import Image

# Pyface modules 
from visionface.models.Detector import Detector   
from visionface.models.LandmarkDetector import DetectedLandmark3D, DetectedLandmark2D

RawDetection = List[Union[int, float, str]]
ImageType = Union[str, np.ndarray, Image.Image]

class BaseAnnotator(ABC):
    @abstractmethod
    def annotate(self, img: ImageType, detections: Union[List[Detector], List[RawDetection]]) -> np.ndarray:
        pass

class BaseLandmarkAnnotator:
    @abstractmethod
    def annotate(
        self, 
        img: ImageType, 
        landmarks: Union[
            List[DetectedLandmark3D], 
            List[DetectedLandmark2D], 
        ],
        connections: Optional[List[Tuple[int, int]]] = None,
        is_drawing_landmarks: bool = True
    ) -> np.ndarray:
        pass