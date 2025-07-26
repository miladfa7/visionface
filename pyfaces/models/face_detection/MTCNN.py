import torch
from torch import nn
import numpy as np
import logging
from typing import List, Any, Union
from enum import Enum
from torchvision.ops.boxes import batched_nms

# pyfaces modules
from pyfaces.models.Detector import Detector, DetectedFace
from pyfaces.commons.download_files import download_model_weights
from pyfaces.commons.image_utils import image_resample, get_cropped_face
from pyfaces.commons.utils import batched_nms_numpy
from pyfaces.commons.detection_utils import (
    convert_to_square_bbox,
    box_padding,
    apply_bbox_regression
)

logging.basicConfig(level=logging.INFO)


class MTCCNModel(Enum):
    ONET = 0
    PNET = 1
    RNET = 2

WEIGHT_NAMES = [
    "mtccn-onet-face.pt",
    "mtccn-pnet-face.pt",
    "mtccn-rnet-face.pt",
]

WEIGHT_URLS = [
    "https://raw.githubusercontent.com/timesler/facenet-pytorch/master/data/onet.pt",
    "https://raw.githubusercontent.com/timesler/facenet-pytorch/master/data/pnet.pt",
    "https://raw.githubusercontent.com/timesler/facenet-pytorch/master/data/rnet.pt",
]


class MTCNNDetector(Detector):
    """MTCNN face detection module.

        This class loads pretrained P-, R-, and O-nets and returns bounding boxes for detected faces
        
        Keyword Arguments:
            min_face_size {int} -- Minimum face size to search for. (default: {20})
            thresholds {list} -- MTCNN face detection thresholds (default: {[0.6, 0.7, 0.7]})
            factor {float} -- Factor used to create a scaling pyramid of face sizes. (default: {0.709})
            post_process {bool} -- Whether or not to post process images tensors before returning.
                (default: {True})
            select_largest {bool} -- If True, if multiple faces are detected, the largest is returned.
                If False, the face with the highest detection probability is returned.
                (default: {True})
            selection_method {string} -- Which heuristic to use for selection. Default None. If
                specified, will override select_largest:
                        "probability": highest probability selected
                        "largest": largest box selected
                        "largest_over_threshold": largest box over a certain probability selected
                        "center_weighted_size": box size minus weighted squared offset from image center
                    (default: {None})
            keep_all {bool} -- If True, all detected faces are returned, in the order dictated by the
                select_largest parameter. (default: {False})
            device {torch.device} -- The device on which to run neural net passes. (default: {None})
    """

    def __init__(
        self, 
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], 
        factor=0.709, 
        post_process=True,
        select_largest=True, 
        selection_method=None, 
        keep_all=True, 
        device=None
    ):
        super().__init__()
        
        # MTCNN specific parameters
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.select_largest = select_largest
        self.keep_all = keep_all
        self.selection_method = selection_method
        
        # Initialize model components
        self.pnet = None
        self.rnet = None  
        self.onet = None
        
        # Build the models
        self.model = self.build_model()
        
        if not self.selection_method:
            self.selection_method = 'largest' if self.select_largest else 'probability'

    def build_model(self) -> Any:
        """
        Build and return the MTCNN face detection model.
        This method loads the P-Net, R-Net, and O-Net components.
        
        Returns:
            dict: Dictionary containing the mtcnn network components
        """
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        
        self.pnet.to(self.device)
        self.rnet.to(self.device)
        self.onet.to(self.device)
        
        return {
            'pnet': self.pnet,
            'rnet': self.rnet,
            'onet': self.onet
        }

    def detect_faces(self, imgs: List[np.ndarray]) -> List[List[DetectedFace]]:
        """
        Detect faces in one or more input images using the MTCNN model.

        Parameters:
            imgs (List[np.ndarray]): 
                A single image or a list of images in BGR format.
        
        Args:
            imgs (Union[np.ndarray, List[np.ndarray]]):
                - A single image as a NumPy array with shape (H, W, 3), or
                - A list of such images.
        
        Returns:
            List[List[DetectedFace]]: 
                A list where each element is a list of DetectedFace objects corresponding to one input image.
                Each DetectedFace object contains:
                    - Bounding box coordinates (xmin, ymin, xmax, ymax)
                    - Confidence score (conf)
                    - Class name ("face")
                    - The cropped face region (cropped_face), extracted from the original image.
        """
        # Preprocess input images
        processed_imgs = self._preprocess_images(imgs)
        
        # Run MTCNN face detection on the preprocessed images
        batch_boxes = self._run_mtcnn_pipeline(processed_imgs)

        # Convert raw detection results into DetectedFace objects
        detections = self.process_faces(imgs, batch_boxes)

        return detections

    def process_faces(self, imgs: List[np.ndarray], results: np.ndarray) -> List[List[DetectedFace]]:
        """
        Process MTCNN detection results and convert them into DetectedFace objects.

        Args:
            imgs (List[np.ndarray]): 
                A list of input images (as NumPy arrays).
            
            results (np.ndarray): 
                A NumPy array of shape (batch_size, num_faces, 5), where each detected face is represented by 
                [x1, y1, x2, y2, confidence_score]. Each sub-array corresponds to detections for a single image.

        Returns:
            List[List[DetectedFace]]: 
                A list where each element is a list of DetectedFace objects corresponding to one input image.
                Each DetectedFace object contains:
                    - Bounding box coordinates (xmin, ymin, xmax, ymax)
                    - Confidence score (conf)
                    - Class name ("face")
                    - The cropped face region (cropped_face), extracted from the original image.
        """

        detections = []
        
        for idx, bboxes in enumerate(results):
            img = imgs[idx]
            current_detections = []
            face_no = 0
            for bbox in bboxes:
                cropped_face = get_cropped_face(img, bbox[:-1])
                class_name = "face" if bbox[2] != 0 and bbox[3] != 0 else None
                facial_info = DetectedFace(
                    xmin=bbox[0],
                    ymin=bbox[1],
                    xmax=bbox[2],
                    ymax=bbox[3],
                    conf=round(bbox[4], 2),
                    class_name = class_name,
                    cropped_face=cropped_face
                )
                current_detections.append(facial_info)
                face_no = face_no+1 if class_name is not None else face_no

            logging.info(
                f"[MTCNNDetector] {face_no} face(s) detected in image id: {idx}, "
                f"min confidence threshold  0.25."
            )
            
            detections.append(current_detections)

        return detections

    def _preprocess_images(self, imgs: List[np.ndarray]) -> torch.Tensor:
        """Preprocess input images for MTCNN."""
        
        if any(img.size != imgs[0].size for img in imgs):
            raise Exception("MTCNN batch processing only compatible with equal-dimension images.")
        
        imgs = np.stack([np.uint8(img) for img in imgs])
        imgs = torch.as_tensor(imgs.copy(), device=self.device)
        return imgs

    def _run_mtcnn_pipeline(self, imgs: torch.Tensor) -> List[np.ndarray]:
        """
        Run the complete MTCNN detection pipeline.
        """
        model_dtype = next(self.pnet.parameters()).dtype
        imgs = imgs.permute(0, 3, 1, 2).type(model_dtype)

        batch_size = len(imgs)
        h, w = imgs.shape[2:4]
        minsize = self.min_face_size
        threshold = self.thresholds
        factor = self.factor
        
        m = 12.0 / minsize
        minl = min(h, w)
        minl = minl * m

        # Create scale pyramid
        scale_i = m
        scales = []
        while minl >= 12:
            scales.append(scale_i)
            scale_i = scale_i * factor
            minl = minl * factor

        # First stage
        boxes = []
        image_inds = []
        scale_picks = []
        all_i = 0
        offset = 0
        
        for scale in scales:
            im_data = image_resample(imgs, (int(h * scale + 1), int(w * scale + 1)))
            im_data = (im_data - 127.5) * 0.0078125
            reg, probs = self.pnet(im_data)
        
            boxes_scale, image_inds_scale = generate_bounding_box(reg, probs[:, 1], scale, threshold[0])
            boxes.append(boxes_scale)
            image_inds.append(image_inds_scale)

            pick = batched_nms(boxes_scale[:, :4], boxes_scale[:, 4], image_inds_scale, 0.5)
            scale_picks.append(pick + offset)
            offset += boxes_scale.shape[0]

        boxes = torch.cat(boxes, dim=0)
        image_inds = torch.cat(image_inds, dim=0)
        scale_picks = torch.cat(scale_picks, dim=0)

        # NMS within each scale + image
        boxes, image_inds = boxes[scale_picks], image_inds[scale_picks]

        # NMS within each image
        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        boxes, image_inds = boxes[pick], image_inds[pick]

        regw = boxes[:, 2] - boxes[:, 0]
        regh = boxes[:, 3] - boxes[:, 1]
        qq1 = boxes[:, 0] + boxes[:, 5] * regw
        qq2 = boxes[:, 1] + boxes[:, 6] * regh
        qq3 = boxes[:, 2] + boxes[:, 7] * regw
        qq4 = boxes[:, 3] + boxes[:, 8] * regh
        boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
        boxes = convert_to_square_bbox(boxes)
        y, ey, x, ex = box_padding(boxes, w, h)
        
        # Second stage
        if len(boxes) > 0:
            im_data = []
            for k in range(len(y)):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                    im_data.append(image_resample(img_k, (24, 24)))
            im_data = torch.cat(im_data, dim=0)
            im_data = (im_data - 127.5) * 0.0078125

            # This is equivalent to out = rnet(im_data) to avoid GPU out of memory.
            out = fixed_batch_process(im_data, self.rnet)

            out0 = out[0].permute(1, 0)
            out1 = out[1].permute(1, 0)
            score = out1[1, :]
            ipass = score > threshold[1]
            boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
            image_inds = image_inds[ipass]
            mv = out0[:, ipass].permute(1, 0)

            # NMS within each image
            pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
            boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
            boxes = apply_bbox_regression(boxes, mv)
            boxes = convert_to_square_bbox(boxes)

        # Third stage
        if len(boxes) > 0:
            y, ey, x, ex = box_padding(boxes, w, h)
            im_data = []
            for k in range(len(y)):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                    im_data.append(image_resample(img_k, (48, 48)))
            im_data = torch.cat(im_data, dim=0)
            im_data = (im_data - 127.5) * 0.0078125
            
            # This is equivalent to out = onet(im_data) to avoid GPU out of memory.
            out = fixed_batch_process(im_data, self.onet)

            out0 = out[0].permute(1, 0)
            out1 = out[1].permute(1, 0)
            out2 = out[2].permute(1, 0)
            score = out2[1, :]
            ipass = score > threshold[2]
            boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
            image_inds = image_inds[ipass]
            mv = out0[:, ipass].permute(1, 0)

            boxes = apply_bbox_regression(boxes, mv)

            # NMS within each image using "Min" strategy
            pick = batched_nms_numpy(boxes[:, :4], boxes[:, 4], image_inds, 0.7, 'Min')
            boxes, image_inds = boxes[pick], image_inds[pick]

        boxes = boxes.detach().numpy()
        image_inds = image_inds.cpu()

        # Group boxes by image
        batch_boxes = []
        for b_i in range(batch_size):
            b_i_inds = np.where(image_inds == b_i)
            batch_boxes.append(boxes[b_i_inds].copy())

        # Post-process boxes and probabilities
        boxes, probs = [], []
        for box in batch_boxes:
            box = np.array(box)
            if len(box) == 0:
                boxes.append(None)
                probs.append([None])
            elif self.select_largest:
                box_order = np.argsort((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))[::-1]
                box = box[box_order]
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
            else:
                boxes.append(box[:, :4])
                probs.append(box[:, 4])

        boxes = np.array(boxes, dtype=object)
        probs = np.array(probs, dtype=object)

        return self._combine_boxes_and_probs(boxes, probs)

    def _combine_boxes_and_probs(
        self, 
        boxes: List[Union[np.ndarray, None]],
        probs: List[Union[np.ndarray, None]]
    ) -> np.ndarray:
        combined = []
        for b, p in zip(boxes, probs):
            if b is None or p is None:
                combined.append(np.array([[0, 0, 0, 0, 0]]))
            else:
                p = np.expand_dims(p, axis=1)  # shape (N, 1)
                combined.append(np.concatenate((b.astype(np.int32), p), axis=1))  # shape (N, 5)
        return combined

class PNet(nn.Module):
    """MTCNN PNet.
    
    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

        self.training = False

        if pretrained:
            model_id = MTCCNModel.PNET.value
            model_name = WEIGHT_NAMES[model_id]
            weight_url = WEIGHT_URLS[model_id]
            model_path = download_model_weights(
                filename=model_name,
                download_url=weight_url
            )
            state_dict = torch.load(model_path, weights_only=False)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        a = self.conv4_1(x)
        a = self.softmax4_1(a)
        b = self.conv4_2(x)
        return b, a

class RNet(nn.Module):
    """MTCNN RNet.
    
    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)

        self.training = False

        if pretrained:
            model_id = MTCCNModel.RNET.value
            model_name = WEIGHT_NAMES[model_id]
            weight_url = WEIGHT_URLS[model_id]
            model_path = download_model_weights(
                filename=model_name,
                download_url=weight_url
            )
            state_dict = torch.load(model_path, weights_only=False)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)
        a = self.dense5_1(x)
        a = self.softmax5_1(a)
        b = self.dense5_2(x)
        return b, a
    

class ONet(nn.Module):
    """MTCNN ONet.
    
    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

        self.training = False

        if pretrained:
            model_id = MTCCNModel.ONET.value
            model_name = WEIGHT_NAMES[model_id]
            weight_url = WEIGHT_URLS[model_id]
            model_path = download_model_weights(
                filename=model_name,
                download_url=weight_url
            )
            state_dict = torch.load(model_path, weights_only=False)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)
        a = self.dense6_1(x)
        a = self.softmax6_1(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b, c, a


def fixed_batch_process(im_data, model):
    batch_size = 512
    out = []
    for i in range(0, len(im_data), batch_size):
        batch = im_data[i:(i+batch_size)]
        out.append(model(batch))

    return tuple(torch.cat(v, dim=0) for v in zip(*out))

def generate_bounding_box(reg, probs, scale, thresh):
    stride = 2
    cellsize = 12

    reg = reg.permute(1, 0, 2, 3)

    mask = probs >= thresh
    mask_inds = mask.nonzero()
    image_inds = mask_inds[:, 0]
    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)
    bb = mask_inds[:, 1:].type(reg.dtype).flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    return boundingbox, image_inds