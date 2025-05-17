from typing import List


def xywh2xyxy(detection: List[int]) -> List[int]: 
    """
    Convert bounding box coordinates from [x, y, width, height] to [x1, y1, x2, y2] format.
    
    Parameters
    ----------
    detection : List[int]
        Bounding box in [x, y, width, height] format where:
        - x, y: coordinates of the top-left corner
        - width, height: dimensions of the bounding box
    
    Returns
    -------
    List[int]
        Bounding box in [x1, y1, x2, y2] format where:
        - x1, y1: coordinates of the top-left corner
        - x2, y2: coordinates of the bottom-right corner
    """   
    return [
        detection[0],
        detection[1],
        detection[0] + detection[2],
        detection[1] + detection[3],
    ]