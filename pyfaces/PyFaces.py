import os 
import numpy as np
from typing import List, Tuple, Union


# pyfaces moduels 
from pyfaces.modules import detection, embedding, landmarks
from pyfaces.models.Detector import DetectedFace
from pyfaces.models.LandmarkDetector import DetectedLandmark3D
from pyfaces.models.FaceEmbedding import FaceEmbedding

def detect_faces(
        images: Union[str, np.ndarray, List[np.ndarray], List[str]],
        detector_backbone: str = "mediapipe",
) -> List[List[DetectedFace]]:
        """
        Detect faces in one or more images using the specified detector backbone.

        Parameters
        ----------
        images : Union[str, np.ndarray, List[str], List[np.ndarray]]
                A single image or a list of images. Each image can be either a file path (str)
                or an image array.

        detector_backbone : str, optional
                Name of the face detection backend to use. Default is "mediapipe".

        Returns
        -------
        List[List[DetectedFace]]: 
                A list where each element is a list of DetectedFace objects for the corresponding input image.
        """
        return detection.detect_faces(
                images=images, 
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