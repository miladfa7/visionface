from typing import Any

# face detection models
from pyfaces.models.face_detection import  (
    MediaPipe,
    MTCNN,
    YOLO,
    YOLOEye,
    YOLOWolrd,
    OpenCV
)
from pyfaces.models.face_embedding import (
    FaceNet
)

# Landmarks detection models
from pyfaces.models.landmark_detection import (
    MediaPipe,
    Dlib
)


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
            "mediapipe": MediaPipe.MediaPipeDetector,
            "mtcnn": MTCNN.MTCNNDetector,
            "yoloe-small": YOLOEye.YOLOEyeSmallDetector,
            "yoloe-medium": YOLOEye.YOLOEyeMediumDetector,
            "yoloe-large": YOLOEye.YOLOEyeLargeDetector,
            "yolo-nano": YOLO.YOLONanoDetector,
            "yolo-small": YOLO.YOLOSmallDetector,
            "yolo-medium": YOLO.YOLOMediumDetector,
            "yolo-large": YOLO.YOLOLargeDetector,
            "yolow-small": YOLOWolrd.YOLOWorldSmallDetector,
            "yolow-medium": YOLOWolrd.YOLOWorldMediumDetector,
            "yolow-large": YOLOWolrd.YOLOWorldLargeDetector,
            "yolow-xlarge": YOLOWolrd.YOLOWorldXLargeDetector,
            "opencv": OpenCV.OpenCVDetector
        },
        "landmark_detection": {
            "mediapipe": MediaPipe.MediaPipeFaceMeshDetector,
            "dlib": Dlib.DlibFaceLandmarkDetector
        },
        "face_embedding": {
            "FaceNet-VGG": FaceNet.FaceNetVGG,
            "FaceNet-CASIA": FaceNet.FaceNetCASIA
        }
    }
    
    if models.get(task) is None:
        raise ValueError(f"Unimplemented task: {task}")
    
    model = models[task].get(model_name)
    if model is None:
        raise ValueError(f"Invalid model_name passed - {task}/{model_name}")
    return model()
