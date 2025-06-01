from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np
from dataclasses import dataclass


class LandmarkDetector(ABC):
    """
    Abstract base class for a face landmark system
    """

    def __init__(self):
        pass

    @abstractmethod
    def build_model(self) -> Any:
        pass

    @abstractmethod
    def detect_landmarks(self, img: np.ndarray):
        pass

    @abstractmethod
    def process_landmarks(self, results):
        pass



@dataclass 
class DetectedLandmark2D:
    x: float
    y : float
    conf: Optional[float] = None

@dataclass 
class DetectedLandmark3D:
    x: float
    y: float
    z: float 
    name: Optional[str] = None
    conf: Optional[float] = None
