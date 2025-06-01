from typing import Any

# face detection models
from pyfaces.models.face_detection.MediaPipe import MediaPipeDetector
from pyfaces.models.face_detection.YOLOEye import (
    YOLOEyeSmallDetector,
    YOLOEyeMediumDetector,
    YOLOEyeLargeDetector
)

# Landmarks detection models
from pyfaces.models.landmark_detection.MediaPipe import MediaPipeFaceMeshDetector


def build_model(model_name: str, task: str) -> Any:
    """
    Build and return a model instance based on the specified task and model name.
    
    This function creates and returns an appropriate model instance
    for the requested task using the specified model implementation.
    
    Parameters
    ----------
    model_name : str
        The name of the model implementation to use (e.g., "mediapipe").
    task : str
        The task category for which to build a model (e.g., "face_detection").
        
    Returns
    -------
    Any
        A buit model class for the specified task.
        
    Raises
    ------
    ValueError
        If the requested task is not implemented in the model registry
    """
    models = {
        "face_detection": {
            "mediapipe": MediaPipeDetector,
            "yoloe-small": YOLOEyeSmallDetector,
            "yoloe-medium": YOLOEyeMediumDetector,
            "yoloe-large": YOLOEyeLargeDetector,
            
        },
        "landmark_detection": {
            "mediapipe": MediaPipeFaceMeshDetector
        }
    }
    
    if models.get(task) is None:
        raise ValueError(f"Unimplemented task: {task}")
    
    model = models[task].get(model_name)
    if model is None:
        raise ValueError(f"Invalid model_name passed - {task}/{model_name}")
    return model()
