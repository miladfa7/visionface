import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Union

# Pyfaces modules
from pyfaces.annotators.base import ImageType, RawDetection
from pyfaces.models.detector import Detector

def get_xyxy(detection: Union[Detector, RawDetection]) -> Tuple[int, int, int, int]:
    """
    Extract bounding box coordinates from detection object or list.
    
    Args:
        detection: Detector or list with [x1, y1, x2, y2]
        
    Returns:
        Tuple of (x1, y1, x2, y2)
        
    Raises:
        TypeError: If detection format is not supported!
    """
    if hasattr(detection, 'xyxy'):
        return detection.xyxy
    elif (isinstance(detection, List) or isinstance(detection, Tuple)) and len(detection) >= 4:
        return int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
    else:
        raise TypeError(f"Unsupported detection type: {type(detection)}")
    

def highlight_face(
        img: ImageType,
        detections: List[Detector],
        highlight_opacity: float = 0.2,
        highlight_color: Tuple[int, int, int] = (255, 255, 255),
) -> ImageType:
    """
    Apply semi-transparent highlight to detected regions in image.
    
    Args:
        img: Input image
        detections: List of detections to highlight
        highlight_opacity: Opacity of highlight (0.0-1.0)
        highlight_color: BGR color tuple for highlight
        
    Returns:
        Image with highlighted regions
    """
    overlay = img.copy()
    for detection in detections:
        x1, y1, x2, y2 = get_xyxy(detection)
        cv2.rectangle(
            overlay, 
            (x1, y1), 
            (x2, y2), 
            highlight_color, 
            -1
        )
        cv2.addWeighted(overlay, highlight_opacity, img, 1 - highlight_opacity, 0, img)
    return img

def convert_img_to_numpy(img: ImageType) -> np.ndarray:
    """
    Convert different image formats to numpy array for processing.
    
    Args:
        img: Image as file path, numpy array, or PIL Image
        
    Returns:
        Image as numpy array in BGR format
        
    Raises:
        FileNotFoundError: If image file does not exist
        ValueError: If image file cannot be loaded
        TypeError: If image format is not supported
    """
    if isinstance(img, str):
        if not os.path.exists(img):
            raise FileNotFoundError(f"Image file not found: {img}")
        img_np = cv2.imread(img)
        if img_np is None:
            raise ValueError(f"Failed to load image: {img}")
        return img_np
    
    elif isinstance(img, np.ndarray):
        return img.copy()
    
    elif isinstance(img, Image.Image):
        img_np = np.array(img)
        # Convert RGB to BGR (OpenCV format)
        if img_np.shape[-1] == 3: 
            img_np = img_np[..., ::-1].copy()
        return img_np
    
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    

