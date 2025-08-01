import numpy as np
from typing import List
import cv2

# pyfaces modules
from pyfaces.models.LandmarkDetector import LandmarkDetector, DetectedLandmark3D
from pyfaces.models.landmark_detection.utils import medipipe_mesh_landmark_names

class MediaPipeFaceMeshDetector(LandmarkDetector):
    """
    Landmark detector that uses MediaPipe Face Mesh to extract 3D facial landmarks.
    """
    def __init__(self):
        """
        Initialize the MediaPipe face mesh model and load landmark names.
        """
        self.mesh_landmark_names = medipipe_mesh_landmark_names()
        self.model = self.build_model()

    def build_model(self):
        """
        Load the MediaPipe FaceMesh model.

        Returns
        -------
        model : mediapipe.solutions.face_mesh.FaceMesh
            An instance of the MediaPipe FaceMesh model.
        """
        try:
            import mediapipe as mp
        except ModuleNotFoundError as error:
            raise ImportError(
                "The 'mediapipe' library is not installed. "
                "It is required for MediaPipeFaceMeshDetector to work. "
                "Please install it using: pip install mediapipe"
            ) from error

        mp_face_mesh = mp.solutions.face_mesh
        landmark_detection = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        return landmark_detection
    
    def _detect_one(self, img: np.ndarray) -> List[DetectedLandmark3D]:
        """
        Detect facial landmarks in a single image.

        Parameters
        ----------
        img : np.ndarray
            The input image in BGR format.

        Returns
        -------
        landmarks : List[DetectedLandmark3D]
            List of detected 3D landmarks for the face.
        """
        results = self.model.process(img)
        if results.multi_face_landmarks:
            return self.process_landmarks(results)
        else:
            return []

    def detect_landmarks(self, imgs: List[np.ndarray]) -> List[List[DetectedLandmark3D]]:
        """
        Detect facial landmarks in a list of images.

        Parameters
        ----------
        imgs : List[np.ndarray]
            List of images (each as a NumPy array in BGR format).

        Returns
        -------
        List[List[DetectedLandmark3D]]
            A list where each element contains the detected landmarks for an image.
        """
        return [self._detect_one(img) for img in imgs]
    
    def process_landmarks(self, results) -> List[DetectedLandmark3D]:
        """
        Convert MediaPipe landmark results into DetectedLandmark3D objects.

        Parameters
        ----------
        results : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
            The raw landmark output from the MediaPipe model.

        Returns
        -------
        landmarks : List[DetectedLandmark3D]
            List of 3D landmarks with optional names.
        """
        landmarks = []
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                landmark_name = self.mesh_landmark_names.get(idx, f"unknown_{idx}")
                x, y, z = lm.x, lm.y, lm.z
                facial_landmarks = DetectedLandmark3D(x=x, y=y, z=z, name=landmark_name)
                landmarks.append(facial_landmarks)
        return landmarks