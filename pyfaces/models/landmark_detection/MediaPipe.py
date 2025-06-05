import numpy as np
from typing import List
import cv2

# pyfaces modules
from pyfaces.models.LandmarkDetector import LandmarkDetector, DetectedLandmark3D
from pyfaces.models.landmark_detection.utils import medipipe_mesh_landmark_names

class MediaPipeFaceMeshDetector(LandmarkDetector):
    def __init__(self):
        self.mesh_landmark_names = medipipe_mesh_landmark_names()
        self.model = self.build_model()

    def build_model(self):
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
    
    def detect_landmarks(self, img: np.ndarray) -> List[DetectedLandmark3D]:

        if img is None:
            raise ValueError("Input image is None. Please provide a valid image.")
        
        # Convert BGR to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img_rgb.shape[:2]
        
        # Run Mediapipe landmark model
        results = self.model.process(img_rgb)
        
        # Process landmarks 
        landmarks = self.process_landmarks(results)

        return landmarks
    
    def process_landmarks(self, results):
        landmarks = []
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                landmark_name = self.mesh_landmark_names.get(idx, f"unknown_{idx}")
                x, y, z = lm.x, lm.y, lm.z
                facial_landmark = DetectedLandmark3D(
                    x=x, 
                    y=y, 
                    z=z,
                    name=landmark_name
                )
                landmarks.append(facial_landmark)
        return landmarks