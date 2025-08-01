import os 
import numpy as np
from typing import List, Tuple, Union


# pyfaces moduels 
from pyfaces.modules import detection, embedding, landmarks
from pyfaces.models.Detector import DetectedFace
from pyfaces.models.LandmarkDetector import DetectedLandmark2D, DetectedLandmark3D
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
        images: Union[str, np.ndarray, List[np.ndarray], List[str]],
        promtps: Union[str, List[str]],
        detector_backbone: str = "yoloe-medium",
) -> List[List[DetectedFace]]:
    """
    Detect faces in one or more images using a prompt-based detection approach.

    Parameters
    ----------
    images : Union[str, np.ndarray, List[str], List[np.ndarray]]
        A single image or a list of images. Each image can be either a file path (str)
        or an image array.

    promtps : Union[str, List[str]]
        A single prompt or a list of prompts describing the object(s) to detect.
        For example, "face".

    detector_backbone : str, optional
        Name of the detection backend to use. Default is "yoloe-medium".
        Must support prompt-based detection.

    Returns
    -------
    List[List[DetectedFace]]
        A list where each element is a list of DetectedFace objects
        for the corresponding input image. Each detection includes bounding box
        coordinates, confidence score, class name, and optionally a cropped region.
    """
    return detection.detect_faces_with_prompt(
        images=images, 
        promtps=promtps,
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

def detect_3d_landmarks(
        images: Union[str, np.ndarray, List[np.ndarray], List[str]],
        detector_backbone: str = "mediapipe"
) -> List[List[DetectedLandmark3D]]:
        """
        Detect 3D facial landmarks in one or more images using the specified detection backbone.

        Parameters
        ----------
        images : Union[str, np.ndarray, List[str], List[np.ndarray]]
                A single image or a list of images. Each image can be either:
                        - A file path (str) to an image file
                        - A NumPy array representing the image

        detector_backbone : str, optional
                The name of the face landmark detection model to use. 
                Supported options typically include "mediapipe", etc. 
                Default is "mediapipe".

        Returns
        -------
        List[List[DetectedLandmark3D]]
                A list of DetectedLandmark3D instances containing the 3D coordinates (x, y, z)
                for each detected facial landmark, along with optional landmark name and confidence score.
        """
        return landmarks.detect_3d_landmarks(
                images=images,
                detector_backbone=detector_backbone
        )
        
def detect_landmarks(
        images: Union[str, np.ndarray, List[np.ndarray], List[str]],
        detector_backbone: str = "dlib"
) -> List[List[DetectedLandmark2D]]:
        """
        Detect 2D facial landmarks in one or more images using the specified detection backbone.

        Parameters
        ----------
        images : Union[str, np.ndarray, List[str], List[np.ndarray]]
                A single image or a list of images. Each image can be either:
                        - A file path (str) to an image file
                        - A NumPy array representing the image

        detector_backbone : str, optional
                The name of the face landmark detection model to use. 
                Supported options typically include "dlib", etc. 
                Default is "dlib".

        Returns
        -------
        List[List[DetectedLandmark2D]]
                A list of DetectedLandmark2D instances containing the 2D coordinates (x, y)
                for each detected facial landmark, along with optional landmark name and confidence score.
        """
        return landmarks.detect_landmarks(
                images=images,
                detector_backbone=detector_backbone
        )