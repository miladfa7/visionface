import numpy as np
import logging
from typing import List, Any

# FaceTech modules
from facetech.models.detector import Detector, DetectedFace

logging.basicConfig(level=logging.INFO)


class MediaPipeDetector(Detector):
    """
    References:
        MediaPipe Face Detection: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_detection.md
    """
    def __init__(self, MODEL_ID: int = 1, MIN_CONFIDENCE: float = 0.5):
        """
        Initialize the MediaPipeDetector.

        Parameters:
            model_id: int, default=1
                The MediaPipe face detection model to use:
                - 0: Short-range model (optimized for faces within 2 meters)
                - 1: Full-range model (optimized for faces within 5 meters)
            
            min_confidence: float, default=0.5
                Minimum confidence threshold (0.0 to 1.0) for face detection.
                Detections below this threshold will be filtered out.
        """
        if MODEL_ID not in (0, 1):
            raise ValueError(f"Invalid MODEL_ID: {MODEL_ID}. MediaPipe only 0 (short-range) or 1 (full-range) are supported.")
        
        super().__init__(MODEL_ID, MIN_CONFIDENCE)
        self.model = self.build_model()

    def build_model(self) -> Any:
        """
        Build and initialize the MediaPipe face detection model.
        
        Returns:
            An instance of MediaPipe's FaceDetection model.
        
        Raises:
            ImportError: If the 'mediapipe' library is not installed.
        """
        try:
            import mediapipe as mp
        except ModuleNotFoundError as error:
            raise ImportError(
                "The 'mediapipe' library is not installed. "
                "It is required for MediaPipeDetector to work. "
                "Please install it using: pip install mediapipe"
            ) from error
        
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            min_detection_confidence=self.conf,
            model_selection=self.model_id
        )
        return face_detection
    
    def detect_faces(self, img: np.ndarray) -> List[DetectedFace]:
        """
        Detect faces in an input image using the MediaPipe model.
        
        Parameters:
            img: np.ndarray
                Input image as a numpy array (BGR or RGB format).
                
        Returns:
            List[DetectedFace]
                A list of DetectedFace objects containing bounding box coordinates
                and confidence scores for each detected face. Empty list if no faces detected.
        """
        img_height, img_width = img.shape[:2]
        results = self.model.process(img)
        
        # Check if no faces detected
        if results.detections is None:
            return []
        
        detections = self.process_faces(results, img_width, img_height)
        return detections

    def process_faces(self, results: Any, img_width: int, img_height: int) -> List[DetectedFace]:
        """
        Process the raw detection results from MediaPipe into DetectedFace objects.
        
        Parameters:
            results: Any
                Detection results from the MediaPipe model's process.
            img_width: int
                Width of the image in pixels.
            img_height: int
                Height of the image in pixels.
                
        Returns:
            List[DetectedFace]
                A list of DetectedFace objects with face coordinates
                and confidence scores for each detected face.
        """
        detections = []
        for detection in results.detections:
            (confidence,) = detection.score
            bounding_box = detection.location_data.relative_bounding_box
            
            # Convert relative coordinates to absolute pixel coordinates
            x = int(bounding_box.xmin * img_width)
            w = int(bounding_box.width * img_width)
            y = int(bounding_box.ymin * img_height)
            h = int(bounding_box.height * img_height)
            
            facial_info = DetectedFace(
                x=x,
                y=y,
                w=w,
                h=h,
                conf=round(confidence, 2)
            )
            detections.append(facial_info)
            
        logging.info(
            f"[MediaPipeDetector] {len(detections)} face(s) detected "
            f"with min confidence threshold {self.conf:.2f}."
        )
        return detections