import os
import numpy as np
from typing import List, Any
import cv2

from pyfaces.models.LandmarkDetector import LandmarkDetector, DetectedLandmark2D
from pyfaces.commons.download_files import download_model_weights
from pyfaces.models.landmark_detection.utils import dlib_landmarks_names


DLIB_PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
DEFAULT_PREDICTOR_NAME = "shape_predictor_68_face_landmarks.dat"
EXPECTED_LANDMARK_COUNT = 68

class DlibFaceLandmarkDetector(LandmarkDetector):
    """
        A robust facial landmark detector using dlib's 68-point face landmark predictor.
        
        Attributes:
            detector (dlib.get_frontal_face_detector): Dlib face detector instance
            predictor (dlib.shape_predictor): Dlib landmark predictor instance
            dlib_landmarks_names (dict): Mapping of landmark indices to names
        
        Example:
            >>> detector = DlibFaceLandmarkDetector()
            >>> image = cv2.imread("face_image.jpg")
            >>> landmarks = detector.detect_landmarks(image)
            >>> print(f"Detected landmarks: {landmarks}")
    """
    def __init__(self):
        """Initialize the DlibFaceLandmarkDetector."""
        self.detector, self.predictor = self.build_model()
        self.dlib_landmarks_names = dlib_landmarks_names()
        self.dlib_landmarks = EXPECTED_LANDMARK_COUNT

    def build_model(self, predictor_name: str = DEFAULT_PREDICTOR_NAME) -> Any:
        """
        Build and initialize the dlib face detector and landmark predictor.
        
        Args:
            predictor_name: Name of the predictor model file
            
        Returns:
            Tuple containing (detector, predictor) instances
        """
        try:
            import dlib 
        except ImportError as e:
            raise ImportError(
                "dlib library is required but not installed. "
                "Install it using: pip install dlib"
            ) from e
        
        # Get the predictor file path
        predictor_path = self._get_weight_file(predictor_name)

        # Initialize dlib components
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(str(predictor_path))

        return detector, predictor
    
    def _get_weight_file(self, predictor_name: str) -> str:
        """
        Download or locate the dlib predictor weight file.
        
        Args:
            predictor_name: Name of the predictor file
            
        Returns:
            Path to the predictor file
        """

        predictor_path = download_model_weights(
            filename="shape_predictor_68_face_landmarks.dat",
            download_url=DLIB_PREDICTOR_URL,
            compression_format="bz2",
            model_name="landmark_detection"
        )
        
        return predictor_path
    
    def detect_landmarks(self, img: np.ndarray) -> List[DetectedLandmark2D]:
        """
        Detects facial landmarks in the input image using dlib's face detector and shape predictor.

        Args:
            img (np.ndarray): The input image in BGR format

        Returns:
            List[DetectedLandmark2D]: A list of detected 2D facial landmarks with coordinates and names.

        Raises:
            ValueError: If the input image is not a valid NumPy array.
        
        Example:
            >>> detector = DlibFaceLandmarkDetector()
            >>> image = cv2.imread("face_image.jpg")
            >>> landmarks = detector.detect_landmarks(image)
            >>> for lm in landmarks:
            ...     print(lm.name, lm.x, lm.y)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        all_landmarks = []
        for i, face in enumerate(faces):
            landmarks = self.predictor(gray, face)
            all_landmarks.append(landmarks)

        landmarks_proceed = self.process_landmarks(all_landmarks)

        return landmarks_proceed

    def process_landmarks(self, results: List) -> List[DetectedLandmark2D]:
        """
        Converts raw dlib shape objects into a list of DetectedLandmark2D objects with landmark names.

        Args:
            results (List[dlib.full_object_detection]): List of raw landmark detection results from dlib.

        Returns:
            List[DetectedLandmark2D]: A list of 2D landmarks for all detected faces.
        """
        landmarks = []
        for landmark in results:
            for idx in range(self.dlib_landmarks):
                landmark_name = self.dlib_landmarks_names.get(idx, f"unknown_{idx}")
                facial_landmark = DetectedLandmark2D(
                    x=landmark.part(idx).x,
                    y=landmark.part(idx).y,
                    name=landmark_name
                )
                landmarks.append(facial_landmark)
        return landmarks
            