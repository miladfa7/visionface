import numpy as np
import logging
from typing import List, Any, Union
from enum import Enum

from torch import device

# Pyfaces modules
from pyfaces.models.Detector import Detector, DetectedFace
from pyfaces.commons.utils import xywh2xyxy
from pyfaces.commons.image_utils import get_cropped_face
from pyfaces.commons.download_files import download_model_weights

logging.basicConfig(level=logging.INFO)

class YOLOModel(Enum):
    """Enum for YOLO model types."""
    NANO = 0
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

WEIGHT_NAMES = [
    "yolov12n-face.pt",
    "yolov12s-face.pt",
    "yolov12m-face.pt",
    "yolov12l-face.pt",
]

WEIGHT_URLS = [
    "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov12n-face.pt",
    "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov12s-face.pt",
    "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov12m-face.pt",
    "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov12l-face.pt",
]


class YOLODetector(Detector):
    """
    References:
        YOLO Face Detection: https://github.com/akanametov/yolo-face
    """
    def __init__(self, model: YOLOModel = YOLOModel.SMALL):
        """
        Initialize the YOLO Detector.
        """
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.build_model(model)

    def build_model(self, model: YOLOModel):
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as error:
            raise ImportError(
                "The 'ultralytics' library is not installed. "
                "It is required for YOLOEyeDetector to work. "
                "Please install it using: pip install ultralytics"
            ) from error
        
        # Get the weight file (and download if necessary)
        model_id = model.value
        model_name = WEIGHT_NAMES[model_id]
        weight_url = WEIGHT_URLS[model_id]
        model_path = download_model_weights(
            filename=model_name,
            download_url=weight_url
        )
        # Load the YOLO face model
        return YOLO(model_path)

    def detect_faces(self, imgs: List[np.ndarray]) -> List[List[DetectedFace]]:
        """
        Detect faces in one or more input images using the MediaPipe model.

        Parameters:
            imgs (List[np.ndarray]): 
                A single image or a list of images in BGR format.

        Returns:
            List[List[DetectedFace]]: 
                A list where each element is a list of DetectedFace objects corresponding to one input image.
                Each DetectedFace includes the bounding box coordinates, confidence score, class name,
                and the cropped face region.
        """
        results = self.model.predict(
            imgs,
            verbose=False,
            show=False, 
            device=self.device
        )
        detections = self.process_faces(imgs, results)
        return detections

    def process_faces(self, imgs: List[np.ndarray], results: Any) -> List[List[DetectedFace]]:
        """
        Process YOLO detection results and convert them into DetectedFace objects.

        Parameters
        ----------
        imgs : List[np.ndarray]
            A single image or a list of images (NumPy arrays).

        results : List[ultralytics.engine.results.Results]
            A list of YOLO detection results, one for each input image.

        Returns
        -------
        List[List[DetectedFace]]
            A list where each element is a list of DetectedFace objects corresponding to one input image.
            Each DetectedFace includes the bounding box coordinates, confidence score, class name,
            and the cropped face region.
        """

        detections = []

        for idx, result in enumerate(results):
            
            if result.boxes is None:
                continue

            current_detections  = []
            bboxes = result.boxes.xyxy.cpu().numpy().astype(int).tolist()
            confidences = result.boxes.conf.cpu().numpy().tolist()
            img = imgs[idx]

            for bbox, conf in zip(bboxes, confidences):
                cropped_face = get_cropped_face(img, bbox)
                facial_info = DetectedFace(
                    xmin=bbox[0],
                    ymin=bbox[1],
                    xmax=bbox[2],
                    ymax=bbox[3],
                    conf=round(conf, 2),
                    class_name="face",
                    cropped_face=cropped_face
                )
                current_detections.append(facial_info)
            
            logging.info(
                f"[YOLODetector] {len(current_detections)} face(s) detected in image id: {idx}, "
                f"min confidence threshold  0.25."
            )
            
            detections.append(current_detections)

        return detections


class YOLONanoDetector(YOLODetector):
    """YOLO Nano detector implementation"""
    def __init__(self):
        super().__init__(model=YOLOModel.NANO)

class YOLOSmallDetector(YOLODetector):
    """YOLO Small detector implementation"""
    def __init__(self):
        super().__init__(model=YOLOModel.SMALL)

class YOLOMediumDetector(YOLODetector):
    """YOLO Medium detector implementation"""
    def __init__(self):
        super().__init__(model=YOLOModel.MEDIUM)

class YOLOLargeDetector(YOLODetector):
    """YOLO Large detector implementation"""
    def __init__(self):
        super().__init__(model=YOLOModel.LARGE)