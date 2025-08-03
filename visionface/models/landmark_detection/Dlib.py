import os
import numpy as np
from typing import List, Any
import cv2

from visionface.models.LandmarkDetector import LandmarkDetector, DetectedLandmark2D
from visionface.commons.download_files import download_model_weights
from visionface.models.landmark_detection.utils import dlib_landmarks_names


DLIB_PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
DEFAULT_PREDICTOR_NAME = "shape_predictor_68_face_landmarks.dat"
EXPECTED_LANDMARK_COUNT = 68

class DlibFaceLandmarkDetector(LandmarkDetector):
    """
    Landmark detector using dlib's 68-point facial shape predictor.

    Attributes
    ----------
    detector : dlib.fhog_object_detector
        Dlib's frontal face detector.

    predictor : dlib.shape_predictor
        Dlib's facial landmark shape predictor.

    dlib_landmarks_names : dict
        Mapping of landmark indices to semantic names.

    dlib_landmarks : int
        Expected number of facial landmarks (default: 68).
    """
    def __init__(self):
        """Initialize the DlibFaceLandmarkDetector."""
        self.detector, self.predictor = self.build_model()
        self.dlib_landmarks_names = dlib_landmarks_names()
        self.dlib_landmarks = EXPECTED_LANDMARK_COUNT

    def build_model(self) -> Any:
        """
        Load the dlib face detector and shape predictor.

        Parameters
        ----------
        predictor_name : str, optional
            Filename of the dlib predictor (default is shape_predictor_68_face_landmarks.dat)

        Returns
        -------
        Tuple[dlib.fhog_object_detector, dlib.shape_predictor]
            Dlib face detector and shape predictor.
        """
        try:
            import dlib 
        except ImportError as e:
            raise ImportError(
                "dlib library is required but not installed. "
                "Install it using: pip install dlib or from source https://github.com/davisking/dlib"
            ) from e
        
        # Get the predictor file path
        predictor_path = download_model_weights(
            filename="shape_predictor_68_face_landmarks.dat",
            download_url=DLIB_PREDICTOR_URL,
            compression_format="bz2",
        )
        # Initialize dlib components
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(str(predictor_path))

        return detector, predictor
    
    def _detect_one(self, img: np.ndarray) -> List[DetectedLandmark2D]:
        """
        Detect facial landmarks in a single image.

        Parameters
        ----------
        img : np.ndarray
            The input image in BGR format.

        Returns
        -------
        List[DetectedLandmark2D]
            List of 2D landmarks detected for all faces in the image.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        all_landmarks = [self.predictor(gray, face) for face in faces]
        return self.process_landmarks(all_landmarks)


    def detect_landmarks(self, imgs: List[np.ndarray]) -> List[List[DetectedLandmark2D]]:
        """
        Detects facial landmarks in a list of images using dlib's face detector and shape predictor.

        Parameters
        ----------
        imgs : List[np.ndarray]
            List of images (each as a NumPy array in BGR format).

        Returns:
            List[List[DetectedLandmark2D]]: A list of detected 2D facial landmarks with coordinates and names.
        
        """
        return [self._detect_one(img) for img in imgs]
        

    def process_landmarks(self, results: List) -> List[DetectedLandmark2D]:
        """
        Convert raw dlib detection results into structured landmark data.

        Parameters
        ----------
        results : List[dlib.full_object_detection]
            Raw landmark predictions from dlib.

        Returns
        -------
        List[DetectedLandmark2D]
            List of structured 2D facial landmarks with names and coordinates.
        """
        landmarks = []
        for face_landmarks in results:
            for idx in range(self.dlib_landmarks):
                name = self.dlib_landmarks_names.get(idx, f"unknown_{idx}")
                part = face_landmarks.part(idx)
                landmarks.append(DetectedLandmark2D(x=part.x, y=part.y, name=name))
        return landmarks
            