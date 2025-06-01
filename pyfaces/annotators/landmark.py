import cv2
import numpy as np
from typing import List, Union, Tuple, Optional, Mapping

# Pyfaces modules
from pyfaces.annotators.base import BaseLandmarkAnnotator
from pyfaces.models.LandmarkDetector import DetectedLandmark3D, DetectedLandmark2D
from pyfaces.annotators.utils import denormalize_landmark
from pyfaces.annotators.helper.landmark_connections import (
    FACEMESH_TESSELATION,
    FACEMESH_CONTOURS,
    FACEMESH_IRISES
)
from pyfaces.annotators.helper.landmark_styles import (
    FaceMeshStyle,
    FaceMeshContoursStyle,
    FaceMeshIrisStyle
)

MEDIAPIPE_FACEMESH_CONNECTIONS = [
    FACEMESH_TESSELATION,
    FACEMESH_CONTOURS,
    FACEMESH_IRISES
]
MEDIAPIPE_FACEMESH_STYLE = [
    FaceMeshStyle(),
    FaceMeshContoursStyle(),
    FaceMeshIrisStyle()

]

class MediaPipeFaceMeshAnnotator(BaseLandmarkAnnotator):
    def __init__(
            self, 
            color: Tuple[int, int, int] = (255, 255, 255),
            thickness: int = 1,
            circle_radius: int = 2
    ):
        self.color = color
        self.thickness = thickness
        self.circle_radius = circle_radius
        
    def annotate(
            self, 
            img: np.ndarray,
            landmarks: Union[
                List[DetectedLandmark3D], 
                List[DetectedLandmark2D], 
            ],
            connections: List[List[Tuple[int, int]]] = MEDIAPIPE_FACEMESH_CONNECTIONS,
            is_drawing_landmarks: bool = True
    ) -> np.ndarray:
        
        image_rows, image_cols, _ = img.shape
        idx_to_coordinates = {}

        for idx, lm in enumerate(landmarks):
            landmark_px = denormalize_landmark(
                normalized_x=lm.x, 
                normalized_y=lm.y,
                image_width=image_cols,
                image_height=image_rows
            )

            if landmark_px:
                idx_to_coordinates[idx] = landmark_px
        
        if connections:
            num_landmarks = len(landmarks)
            for cidx, connection_list in enumerate(connections):
                for connection in connection_list:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                        raise ValueError(f'Landmark index is out of range. Invalid connection '
                                        f'from landmark #{start_idx} to landmark #{end_idx}.')
                    if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                        drawing_spec = MEDIAPIPE_FACEMESH_STYLE[cidx][connection] if isinstance(
                            MEDIAPIPE_FACEMESH_STYLE[cidx], Mapping) else MEDIAPIPE_FACEMESH_STYLE[cidx]
                        cv2.line(img, idx_to_coordinates[start_idx],
                                idx_to_coordinates[end_idx], self.color,
                                self.thickness)

        if is_drawing_landmarks:
            for idx, landmark_px in idx_to_coordinates.items():
                circle_border_radius = max(self.circle_radius + 1, int(self.circle_radius * 1.2))
                cv2.circle(img, landmark_px, circle_border_radius, self.color, self.thickness)
                # Fill color into the circle
                cv2.circle(img, landmark_px, self.circle_radius, self.color, self.thickness)

        return img
