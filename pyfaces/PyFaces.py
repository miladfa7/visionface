import os 
import numpy as np
from typing import List, Tuple, Union


# pyfaces moduels 
from pyfaces.modules import detection, embedding, landmarks
from pyfaces.models.Detector import DetectedFace
from pyfaces.models.LandmarkDetector import DetectedLandmark3D
from pyfaces.models.FaceEmbedding import FaceEmbedding

def detect_faces(
        image_path: Union[str, np.ndarray],
        detector_backbone: str = "mediapipe",
) -> List[DetectedFace]:
        """
        Detect faces in an image using the specified detector backbone.

        This function takes an image path or numpy array and detects faces
        in the image using the specified detector backend.

        Parameters
        ----------
        image_path : Union[str, np.ndarray]
                Path to the image file as a string or a numpy array containing
                the image data.
        detector_backbone : str, optional
                Name of the face detection backend to use, by default "mediapipe".

        Returns
        -------
        List
                A list of detected faces. The structure depends on the detector backend used.

        Raises
        ------
        ValueError
                If the input image is None or cannot be properly loaded.
        """
    
        return detection.detect_faces(
                image_path=image_path, 
                detector_backbone=detector_backbone
        )


def detect_faces_with_prompt(
        image_path: Union[str, np.ndarray],
        promtp: Union[str, List[str]],
        detector_backbone: str = "yoloe-medium",
) -> List[DetectedFace]:
        
        return detection.detect_faces_with_prompt(
                image_path=image_path, 
                promtp=promtp,
                detector_backbone=detector_backbone
        ) 

def embed_faces(
        face_imgs: Union[str, np.ndarray, List[np.ndarray], List[str]],
        model_name: str = "FaceNet-VGG",
        normalize_embeddings: bool = True
) -> FaceEmbedding:
        """
        Compute face embeddings for one or more face images.

        Args:
                face_imgs: A single face image or a list of images as file paths (str) or numpy arrays.
                model_name: Name of the embedding model to use.
                normalize_embeddings: Whether to apply L2 normalization to embeddings.

        Returns:
                FaceEmbedding: Embedding objects for each face.
        """
        return embedding.embed_faces(
                face_imgs=face_imgs,
                model_name=model_name,
                normalize_embeddings=normalize_embeddings
        )

def detect_landmark_3d(
        image_path: Union[str, np.ndarray],
        detector_backbone: str = "mediapipe"
) -> List[DetectedLandmark3D]:
        
        return landmarks.detect_landmark_3d(
                image_path=image_path,
                detector_backbone=detector_backbone
        )