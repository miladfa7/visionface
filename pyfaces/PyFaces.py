import os 
import numpy as np
from typing import List, Tuple, Union


# pyfaces moduels 
from pyfaces.modules import detection


def detect_faces(
        image_path: Union[str, np.ndarray],
        detector_backbone: str = "mediapipe",
) -> List:
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
    
        return detection.detect_faces(image_path=image_path, detector_backbone=detector_backbone)

