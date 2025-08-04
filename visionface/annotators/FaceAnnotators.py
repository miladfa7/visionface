from typing import List, Union, Tuple


#VisionFace module
from visionface.annotators.base import ImageType, RawDetection
from visionface.models.Detector import Detector
from visionface.annotators.detection import BoxCornerAnnotator, BoxAnnotator


def box_annotator(
        img: ImageType,
        detections: Union[List[Detector], List[RawDetection]],
        color: Tuple = (245, 113, 47),
        thickness: int = 4,
        highlight: bool = True,
        highlight_opacity: float = 0.2,
        highlight_color: tuple = (255, 255, 255),
):
    """
    Annotate an image with bounding boxes around detected face(s).
    
    Parameters
    ----------
    img : ImageType
        The input image on which to draw annotations. Can be either a NumPy array
        or a PIL Image object.
    detections : List[Detector]
        A list of detection face(s) containing bounding box information.
    color : Tuple, optional
        The RGB color for the bounding boxes, default is (245, 113, 47).
    thickness : int, optional
        The thickness of the bounding box lines in pixels, default is 4.
    highlight : bool, optional
        Whether to highlight the detected regions, by default True
    highlight_opacity : float, optional
        Opacity of the highlight overlay (0.0 to 1.0), by default 0.2
    highlight_color : tuple, optional
        BGR color tuple for the highlight, by default (255, 255, 255)
    
    Returns
    -------
    ImageType
        The input image with bounding box annotations added.
    """
    annotator = BoxAnnotator(
        color=color, 
        thickness=thickness,
    )
    return annotator.annotate(
        img=img, 
        detections=detections,
        highlight=highlight,
        highlight_opacity=highlight_opacity,
        highlight_color=highlight_color
    )

def box_corner_annotator(
        img: ImageType,
        detections: Union[List[Detector], List[RawDetection]],
        color: Tuple = (245, 113, 47),
        thickness: int = 4,
        corner_length: int = 15,
        highlight: bool = True,
        highlight_opacity: float = 0.2,
        highlight_color: tuple = (255, 255, 255),
):
    """
    Annotate an image with corner boxes around detected face(s).
    
    Parameters
    ----------
    img : ImageType
        The input image on which to draw annotations. Can be either a NumPy array
        or a PIL Image object.
    detections : List[Detector]
        A list of detection face(s) containing bounding box information.
    color : Tuple, optional
        The RGB color for the corner boxes, default is (245, 113, 47).
    thickness : int, optional
        The thickness of the corner box lines in pixels, default is 4.
    corner_length : int, optional
        The length of each corner in pixels, default is 15.
    
    Returns
    -------
    ImageType
        The input image with corner box annotations added.
    """
    annotator = BoxCornerAnnotator(
        color=color, 
        thickness=thickness, 
        corner_length=corner_length,
    )
    return annotator.annotate(
        img=img, 
        detections=detections,
        highlight=highlight,
        highlight_opacity=highlight_opacity,
        highlight_color=highlight_color
    )