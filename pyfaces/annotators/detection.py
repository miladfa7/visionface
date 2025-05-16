import numpy as np
import cv2
from typing import List, Union

# Pyfaces modules 
from pyfaces.annotators.base import BaseAnnotator, ImageType, RawDetection
from pyfaces.models.detector import Detector
from pyfaces.annotators.utils import (
    highlight_face, 
    convert_img_to_numpy,
    get_xyxy
)

class BoxCornerAnnotator(BaseAnnotator):
    """
    A class for drawing box corners on an image using provided detections.
    
    Parameters
    ----------
    color : tuple, optional
        The BGR color tuple for the corner lines, by default (245, 113, 47)
    thickness : int, optional
        The thickness of the corner lines in pixels, by default 4
    corner_length : int, optional
        The length of each corner segment in pixels, by default 15
    
    Attributes
    ----------
    color : tuple
        The BGR color tuple used for drawing corner lines
    thickness : int
        The thickness of the corner lines in pixels
    corner_length : int
        The length of each corner segment in pixels
    
    Examples
    --------
    >>> # Using Face Detector
    >>> from pyfaces import PyFaces, FaceAnnotators
    >>> detected_faces = PyFaces.detect_faces(img)
    >>> annotated_img = FaceAnnotators.box_corner_annotator(img, detected_faces, highlight=True)

    >>> # Using raw detection lists
    >>> raw_detections = [[10, 20, 100, 200, 0.95, 'face'], [30, 40, 120, 220, 0.90, 'face']]
    >>> annotated_img = annotator.annotate(img, raw_detections, highlight=True)
    """

    def __init__(self, color: tuple = (245, 113, 47), thickness: int = 4, corner_length: int = 15):
        self.color = color
        self.thickness = thickness
        self.corner_length = corner_length

    def annotate(
            self, 
            img: ImageType, 
            detections: Union[List[Detector], List[RawDetection]],
            highlight: bool = True,
            highlight_opacity: float = 0.2,
            highlight_color: tuple = (255, 255, 255),
    ) -> ImageType:
        """
        Annotate the image with corner boxes for each detection.
        
        This method draws L-shaped corners at each corner of the bounding boxes
        defined by the provided detections. Optionally, it can also highlight
        the detected regions with a semi-transparent overlay.
        
        Parameters
        ----------
        img : ImageType
            The input image to annotate (can be a file path string, numpy array, or PIL Image)
        detections : Union[List[Detector], List[RawDetection]]
            List of detections, where each detection can be either:
            - Detector object with an xyxy property returning (x1, y1, x2, y2)
            - RawDetection list in format [x1, y1, x2, y2, confidence, class_name]
        highlight : bool, optional
            Whether to highlight the detected regions, by default True
        highlight_opacity : float, optional
            Opacity of the highlight overlay (0.0 to 1.0), by default 0.2
        highlight_color : tuple, optional
            BGR color tuple for the highlight, by default (255, 255, 255)
            
        Returns
        -------
        ImageType
            The annotated image with box corners and optional highlights
        """
        # Convert image to numpy for processing
        img = convert_img_to_numpy(img)

        # Apply highlighting if enabled
        if highlight:
            img = highlight_face(
                img,
                detections,
                highlight_opacity=highlight_opacity,
                highlight_color=highlight_color
            )

        # Draw box corners
        for detection in detections:
            x1, y1, x2, y2 = get_xyxy(detection)
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            for x, y in corners:
                x_end = x + self.corner_length if x == x1 else x - self.corner_length
                cv2.line(img, (x, y), (x_end, y), self.color, thickness=self.thickness)

                y_end = y + self.corner_length if y == y1 else y - self.corner_length
                cv2.line(img, (x, y), (x, y_end), self.color, thickness=self.thickness)
        
        return img