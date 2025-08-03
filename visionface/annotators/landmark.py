import cv2
import numpy as np
from typing import List, Union, Tuple, Optional, Mapping

# VisionFace modules
from visionface.annotators.base import BaseLandmarkAnnotator
from visionface.models.LandmarkDetector import DetectedLandmark3D, DetectedLandmark2D
from visionface.annotators.utils import denormalize_landmark
from visionface.annotators.helper.landmark_connections import (
    FACEMESH_TESSELATION,
    FACEMESH_CONTOURS,
    FACEMESH_IRISES,
    DLIB_FACE_LANDMARK_CONNECTIONS
    
)
from visionface.annotators.helper.landmark_styles import (
    FaceMeshStyle,
    FaceMeshContoursStyle,
    FaceMeshIrisStyle
)

MEDIAPIPE_FACEMESH_CONNECTIONS = [
    FACEMESH_TESSELATION,
    FACEMESH_CONTOURS,
    FACEMESH_IRISES
]
DLIB_LANDMARK_CONNECTIONS = [

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
            landmarks: List[DetectedLandmark3D], 
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

class FaceLandmarkAnnotator(BaseLandmarkAnnotator):
    """
    A facial landmark annotator that visualizes detected landmarks and their connections.
    
    Attributes:
        line_color (Tuple[int, int, int]): BGR color values for connection lines. Default is (0, 255, 0) - green.
        line_thickness (int): Thickness of connection lines in pixels. Default is 1.
        circle_color (Tuple[int, int, int]): BGR color values for landmark points. Default is (255, 255, 255) - white.
        circle_radius (int): Radius of landmark circles in pixels. Default is 2.
    
    Example:
        >>> from VisionFace.models.landmark_detection.Dlib import DlibFaceLandmarkDetector
        >>> from VisionFace.annotators.landmark import FaceLandmarkAnnotator
        >>> from VisionFace.annotators.helper.landmark_connections import DLIB_FACE_LANDMARK_CONNECTIONS
        >>> 
        >>> detector = DlibFaceLandmarkDetector()
        >>> annotator = FaceLandmarkAnnotator(
        ...     line_color=(0, 255, 0),
        ...     circle_color=(255, 0, 0),
        ...     circle_radius=3
        ... )
        >>> 
        >>> img = cv2.imread("face_image.jpg")
        >>> landmarks = detector.detect_landmarks(img)
        >>> annotated_img = annotator.annotate(
        ...     img=img,
        ...     landmarks=landmarks,
        ...     connections=DLIB_FACE_LANDMARK_CONNECTIONS
        ... )
    """
    
    def __init__(
            self, 
            line_color: Tuple[int, int, int] = (0, 255, 0),
            line_thickness: int = 1,
            circle_color: Tuple[int, int, int] = (255, 255, 255),
            circle_radius: int = 2
    ):
        """
        Initialize the FaceLandmarkAnnotator with visualization parameters.
        
        Args:
            line_color (Tuple[int, int, int], optional): BGR color tuple for connection lines. Defaults to (0, 255, 0) - green.
            line_thickness (int, optional): Thickness of connection lines in pixels. Defaults to 1.
            circle_color (Tuple[int, int, int], optional): BGR color for landmark circles. Defaults to (255, 255, 255) - white.
            circle_radius (int, optional): Radius of landmark circles in pixels. Defaults to 2.
        """
        self.line_color = line_color
        self.line_thickness = line_thickness
        self.circle_color = circle_color
        self.circle_radius = circle_radius
    
    def annotate(
            self, 
            img: np.ndarray,
            landmarks: List[DetectedLandmark2D], 
            connections: List[Tuple[int, int]] = "",
            is_drawing_landmarks: bool = True
    ) -> np.ndarray:
        """
        Annotate an image with facial landmarks and their connections.
        
        Args:
            img (np.ndarray): Input image as a numpy array
            landmarks (List[DetectedLandmark2D]): List of detected facial landmarks.
                Each landmark should have 'x' and 'y' attributes representing pixel coordinates.
            connections (List[Tuple[int, int]], optional): landmark connections for drawing facial feature outlines.
            is_drawing_landmarks (bool, optional): Whether to draw landmark annotations.
                If False, returns the original image unchanged. Defaults to True.
        
        Returns:
            np.ndarray: The annotated image with landmarks and connections drawn.
        
        """
        if connections and is_drawing_landmarks:
            # Draw connection lines
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = [landmarks[start_idx].x, landmarks[start_idx].y]
                    end_point = [landmarks[end_idx].x, landmarks[end_idx].y]
                    img = cv2.line(img, start_point, end_point, self.line_color, self.line_thickness)
            
            # Draw landmark points
            for point in landmarks:
                landmark = [point.x, point.y]
                cv2.circle(img, landmark, self.circle_radius, self.circle_color, -1)
        
        return img
