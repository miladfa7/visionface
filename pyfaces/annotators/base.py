from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np
from PIL import Image

# Pyface modules 
from pyfaces.models.detector import Detector   

RawDetection = List[Union[int, float, str]]
ImageType = Union[str, np.ndarray, Image.Image]

class BaseAnnotator(ABC):
    @abstractmethod
    def annotate(self, img: ImageType, detections: Union[List[Detector], List[RawDetection]]) -> np.ndarray:
        pass