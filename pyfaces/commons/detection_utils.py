from typing import List
import torch

from pyfaces.models.Detector import DetectedFace


def convert_to_square_bbox(bboxA):
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    
    l = torch.max(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, :2] + l.repeat(2, 1).permute(1, 0)

    return bboxA


def box_padding(boxes, w, h):
    boxes = boxes.trunc().int().cpu().numpy()
    x = boxes[:, 0]
    y = boxes[:, 1]
    ex = boxes[:, 2]
    ey = boxes[:, 3]

    x[x < 1] = 1
    y[y < 1] = 1
    ex[ex > w] = w
    ey[ey > h] = h

    return y, ey, x, ex

def apply_bbox_regression(boundingbox, reg):
    if reg.shape[1] == 1:
        reg = torch.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, :4] = torch.stack([b1, b2, b3, b4]).permute(1, 0)

    return boundingbox

def select_max_conf_faces(
    face_detections: List[List[DetectedFace]]
) -> List[DetectedFace]:
    """
    Selects the DetectedFace with the highest confidence from each list of detections.

    Parameters
    ----------
    face_detections : List[List[DetectedFace]]
        A list of detection lists. Each inner list contains DetectedFace objects for one image.

    Returns
    -------
    List[DetectedFace]
        A list containing the DetectedFace with the highest confidence from each image.
    """
    return [[max(detections, key=lambda face: face.conf) for detections in face_detections if detections]]