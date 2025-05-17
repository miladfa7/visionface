import os
import numpy as np
import logging
from typing import List, Any, Union
import cv2
from enum import Enum

# Pyfaces modules 
from pyfaces.models.detector import Detector, DetectedFace

logger = logging.getLogger(__name__)

class YOLOEModel(Enum):
    """Enum for YOLOE model types."""
    SMALL = 0
    MEDIUM = 1
    LARGE = 2

#Text/Visual Prompt models
WEIGHT_NAMES = ["yoloe-11s-seg.pt",
                "yoloe-11m-seg.pt",
                "yoloe-11l-seg.pt"]

WEIGHT_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11m-seg.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11l-seg.pt"
]

class YOLOEyeDetector(Detector):
    """
    Reference: https://github.com/THU-MIG/yoloe
    """
    def __init__(self, model: YOLOEModel = YOLOEModel.MEDIUM):
        """
        Initialize the YOLOEyeDetector.
        """
        self.model = self.build_model(model)

    def build_model(self, model: YOLOEModel):
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as error:
            raise ImportError(
                "The 'ultralytics' library is not installed. "
                "It is required for YOLOEyeDetector to work. "
                "Please install it using: pip install ultralytics"
            ) from error
        
        # Get the weight file (and download if necessary)
        weight_file = self._get_weight_file(model)
        
        # Load the YOLO model
        return YOLO(weight_file)
    
    def _get_weight_file(self, model: YOLOEModel) -> str:
        """
        Get the weight file for the specified model and download if necessary
        
        Args:
            model (YOLOEModel): The model enum
            
        Returns:
            str: Path to the weight file
        """
        weight_name = WEIGHT_NAMES[model.value]
        
        weights_dir = "weights"
        os.makedirs(weights_dir, exist_ok=True)
        
        weight_path = os.path.join(weights_dir, weight_name)
        
        if not os.path.isfile(weight_path):
            logger.info(f"Downloading {weight_name} from {WEIGHT_URLS[model.value]}")
        
        return weight_path

    def detect_faces(self, img: np.ndarray) -> List[DetectedFace]:
        """
        Detect faces in the given image.        
        Args:
            img (np.ndarray): Input image as a NumPy array (H, W, C).
            
        Returns:
            List[DetectedFace]: A list of detected faces.
        """
        # By default, use a generic "face" prompt for detection
        return self.detect_faces_with_prompt(img, prompt="face")
    
    def _set_text_prompt(self, prompt: str):
        """
        Set the text prompt for the YOLOEye model.
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        self.model.set_classes(prompt, self.model.get_text_pe(prompt))

    def detect_faces_with_prompt(
            self, img: np.ndarray, prompt: Union[str, List[str]]
    ) -> List[DetectedFace]:
        """
            Detect faces in the given image based on text prompt guidance.
            
            Args:
                img (np.ndarray): Input image as a NumPy array (H, W, C).
                prompt (Union[str, List[str]]): Either a single text prompt or a list of text prompts
                                                describing the faces to detect.
                
            Returns:
                List[DetectedFace]: A list of detected faces that match the prompt(s).
        """
        # Set text prompt to detect faces
        self._set_text_prompt(prompt)
        
        # Run detection on the given image
        results = self.model.predict(img)
        
        # Check if no faces detected
        if len(results[0]) == 0:
            return []
        
        # Process the results from Ultralytics YOLOE model
        detections = self.process_faces(results[0])
        return detections

    def detect_faces_with_visual(self, img: np.ndarray, ) -> List[DetectedFace]:
        pass
    
    def process_faces(self, results) -> List[DetectedFace]:
        """
        Process the raw detections into a structured format.
        """
        detections = []

        class_id = results.boxes.cls.cpu().numpy().astype(int)
        class_names = np.array([results.names[i] for i in class_id])
        bboxes = results.boxes.xyxy.cpu().numpy().astype(int)
        confidence = results.boxes.conf.cpu().numpy()
        
        for bbox, conf, class_name in zip(bboxes, confidence, class_names):
            facial_info = DetectedFace(
                xmin=bbox[0], 
                ymin=bbox[1], 
                xmax=bbox[2], 
                ymax=bbox[3], 
                conf=round(conf, 2),
                class_name=class_name
            )
            detections.append(facial_info)
        
        logging.info(
            f"[YOLOEyeDetector] {len(detections)} face(s) detected "
        )
        return detections


class YOLOEyeSmallDetector(YOLOEyeDetector):
    """YOLOEye Small detector implementation"""
    def __init__(self):
        super().__init__(model=YOLOEModel.SMALL)

class YOLOEyeMediumDetector(YOLOEyeDetector):
    """YOLOEye Medium detector implementation"""
    def __init__(self):
        super().__init__(model=YOLOEModel.MEDIUM)

class YOLOEyeLargeDetector(YOLOEyeDetector):
    """YOLOEye Large detector implementation"""
    def __init__(self):
        super().__init__(model=YOLOEModel.LARGE)

        