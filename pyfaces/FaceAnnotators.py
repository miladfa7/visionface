from typing import List, Union, Tuple


#PyFaces module
from pyfaces.annotators.base import ImageType, RawDetection
from pyfaces.models.detector import Detector
from pyfaces.annotators.detection import BoxCornerAnnotator

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