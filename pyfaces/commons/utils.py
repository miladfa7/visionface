import torch
import numpy as np
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

def get_home_directory():
    return "."

def nms_numpy(boxes, scores, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))

    x1 = boxes[:, 0].copy()
    y1 = boxes[:, 1].copy()
    x2 = boxes[:, 2].copy()
    y2 = boxes[:, 3].copy()
    s = scores
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]

        xx1 = np.maximum(x1[i], x1[idx]).copy()
        yy1 = np.maximum(y1[i], y1[idx]).copy()
        xx2 = np.minimum(x2[i], x2[idx]).copy()
        yy2 = np.minimum(y2[i], y2[idx]).copy()

        w = np.maximum(0.0, xx2 - xx1 + 1).copy()
        h = np.maximum(0.0, yy2 - yy1 + 1).copy()

        inter = w * h
        if method == 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]

    pick = pick[:counter].copy()
    return pick

def batched_nms_numpy(boxes, scores, idxs, threshold, method):
    device = boxes.device
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    boxes_for_nms = boxes_for_nms.detach().numpy()
    scores = scores.detach().numpy()
    keep = nms_numpy(boxes_for_nms, scores, threshold, method)
    return torch.as_tensor(keep, dtype=torch.long, device=device)