import numpy as np
import logging
from typing import List, Any, Union
from enum import Enum

# Pyfaces modules
from pyfaces.models.Detector import Detector, DetectedFace
from pyfaces.commons.image_utils import get_cropped_face
from pyfaces.commons.download_files import download_model_weights

logging.basicConfig(level=logging.INFO)

class YOLOModel(Enum):
    """Enum for YOLO World model types."""
    SMALL = 0
    MEDIUM = 1
    LARGE = 2
    XLARGE = 3

WEIGHT_NAMES = [
    "yolov8s-world.pt",
    "yolov8m-world.pt",
    "yolov8l-world.pt",
    "yolov8x-world.pt",
]

WEIGHT_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-world.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-world.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-world.pt",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-world.pt",
]


class YOLOWolrdDetector(Detector):
    def __init__(self, model: YOLOModel = YOLOModel.MEDIUM):
        """
        Initialize the YOLO Detector.
        """
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.build_model(model)

    def build_model(self, model: YOLOModel):
        try:
            from ultralytics import YOLOWorld
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
        return YOLOWorld(model_path)

    def detect_faces(self, imgs: List[np.ndarray]) -> List[List[DetectedFace]]:
        """
        Detect faces in one or more input images using the YOLO Wolrd model.

        Parameters:
            imgs (List[np.ndarray]): 
                A single image or a list of images in BGR format.

        Returns:
            List[List[DetectedFace]]: 
                A list where each element is a list of DetectedFace objects corresponding to one input image.
                Each DetectedFace includes the bounding box coordinates, confidence score, class name,
        """
        # By default, use a generic "face" prompt for detection
        return self.detect_faces_with_prompt(imgs, prompts="face")

    def _set_text_prompt(self, prompts: List[str]) -> None:
        """
        Set the text prompt for the YOLO World model.
        """
        self.model.set_classes(prompts)

    def detect_faces_with_prompt(
        self, 
        imgs: List[np.ndarray],
        prompts: List[str],
    ) -> List[List[DetectedFace]]:
        """
        Detect faces in the given image based on text prompt guidance.
        
        Args:
            img (np.ndarray): Input image as a NumPy array (H, W, C).
            prompts (Union[str, List[str]]): Either a single text prompt or a list of text prompts
                                            describing the faces to detect.
            
        Returns:
            List[DetectedFace]: A list of detected faces that match the prompt(s).
        """
        # Set text prompt to detect faces
        self._set_text_prompt(prompts)
        
        results = self.model.predict(
            imgs,
            verbose=False,
            show=False, 
            device=self.device
        )

        # Process the results from Ultralytics YOLO World model
        detections = self.process_faces(imgs, results)

        return detections

    def detect_faces_with_visual(self, imgs: List[np.ndarray]) -> List[DetectedFace]:
        pass
    
    def process_faces(self, imgs: List[np.ndarray], results: List[Any]) -> List[List[DetectedFace]]:
        """
        Process the raw detections into a structured format.
        """

        detections = []

        for idx, result in enumerate(results):

            current_detections = []
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            class_names = np.array([result.names[i] for i in class_id])
            bboxes = result.boxes.xyxy.cpu().numpy().astype(int)
            confidence = result.boxes.conf.cpu().numpy()
            img = imgs[idx]

            if not len(bboxes):
                detections.append(DetectedFace(xmin=0, ymin=0, xmax=0, ymax=0, conf=0))
                continue

            for bbox, conf, class_name in zip(bboxes, confidence, class_names):
                cropped_face = get_cropped_face(img, bbox)
                facial_info = DetectedFace(
                    xmin=bbox[0], 
                    ymin=bbox[1], 
                    xmax=bbox[2], 
                    ymax=bbox[3], 
                    conf=round(conf, 2),
                    class_name=class_name,
                    cropped_face=cropped_face
                )
                current_detections.append(facial_info)
        
            logging.info(
                f"{len(current_detections)} face(s) detected in image id: {idx},"
            )

            detections.append(current_detections)

        return detections


class YOLOWorldSmallDetector(YOLOWolrdDetector):
    """YOLO Small detector implementation"""
    def __init__(self):
        super().__init__(model=YOLOModel.SMALL)

class YOLOWorldMediumDetector(YOLOWolrdDetector):
    """YOLO Medium detector implementation"""
    def __init__(self):
        super().__init__(model=YOLOModel.MEDIUM)

class YOLOWorldLargeDetector(YOLOWolrdDetector):
    """YOLO Large detector implementation"""
    def __init__(self):
        super().__init__(model=YOLOModel.LARGE)

class YOLOWorldXLargeDetector(YOLOWolrdDetector):
    """YOLO XLarge detector implementation"""
    def __init__(self):
        super().__init__(model=YOLOModel.XLARGE)