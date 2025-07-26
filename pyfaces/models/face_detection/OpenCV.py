import cv2
import numpy as np
import logging
from typing import List

# pyfaces modules
from pyfaces.models.Detector import Detector, DetectedFace
from pyfaces.commons.download_files import download_model_weights
from pyfaces.commons.image_utils import get_cropped_face

logging.basicConfig(level=logging.INFO)


FILE_NAMES = [
    "opencv_deploy.prototxt",
    "opencv_res10_300x300_ssd_iter_140000.caffemodel",
]

FILE_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
]


class OpenCVDetector(Detector):
    def __init__(self):
        """
        Initializes the OpenCV face detector using a pre-trained Caffe model.
        """
        super().__init__()
        self.input_size = (300, 300)  # Standard input size for the model    
        self.model = self.build_model()

    def build_model(self) -> cv2.dnn_Net:
        """
        Downloads model files and loads the OpenCV DNN face detector.

        Returns:
            cv2.dnn_Net: The loaded OpenCV DNN model.
        """
        prototxt_name = FILE_NAMES[0]
        prototxt_url = FILE_URLS[0]
        weights_name = FILE_NAMES[1]
        weights_url = FILE_URLS[1]

        prototxt_path = download_model_weights(
            filename=prototxt_name,
            download_url=prototxt_url
        )
        weights_path = download_model_weights(
            filename=weights_name,
            download_url=weights_url
        )
        # Load OpenCV DNN model
        model = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

        # Set backend and target for better performance
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        return model

    def _detect_one(self, img: np.ndarray) -> np.ndarray:
        """
        Detects faces in a single image using the loaded OpenCV DNN model.

        Args:
            img (np.ndarray): Input image in BGR format.

        Returns:
            np.ndarray: Raw detection output from the model.
        """
        blob = cv2.dnn.blobFromImage(
            img, 
            scalefactor=1.0,
            size=self.input_size,
            mean=(104.0, 177.0, 123.0)
            )
        self.model.setInput(blob)
        detections = self.model.forward()
        return detections
    
    def detect_faces(self, imgs: List[np.ndarray]) -> List[List[DetectedFace]]:
        """
        Detect faces in one or more input images using the Opencv model.

        Parameters:
            imgs (List[np.ndarray]): 
                A single image or a list of images in BGR format.

        Returns:
            List[List[DetectedFace]]: 
                A list where each element is a list of DetectedFace objects corresponding to one input image.
                Each DetectedFace includes the bounding box coordinates, confidence score, class name,
                and the cropped face region.
        """        

        results = [self._detect_one(img) for img in imgs]
        
        all_detections = self.process_faces(imgs, results)

        return all_detections

    def process_faces(self, imgs: List[np.ndarray], results: List[np.ndarray]) -> List[List[DetectedFace]]:
        """
        Converts raw model outputs into structured DetectedFace objects.

        Args:
            imgs (List[np.ndarray]): List of original images.
            results (List[np.ndarray]): List of raw model outputs per image.

        Returns:
            List[List[DetectedFace]]: List of detections for each image.
        """

        detections = [] 

        for idx, result in enumerate(results):
            img = imgs[idx]
            h, w = img.shape[:2]
            current_detections = []
            face_no = 0
            for i in range(result.shape[2]):
                confidence = result[0, 0, i, 2]
                if confidence > self.conf:
                    # Get bounding box coordinates
                    box = result[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1),  max(0, y1)
                    x2, y2 = min(w, x2),  min(h, y2)
                    cropped_face = get_cropped_face(img, [x1, y1, x2, y2])
                    
                    facial_info = DetectedFace(
                        xmin=x1,
                        ymin=y1,
                        xmax=x2,
                        ymax=y2,
                        conf=round(confidence, 2),
                        class_name="face",
                        cropped_face=cropped_face
                    )
                    current_detections.append(facial_info)
                    face_no +=1

            if not len(current_detections):
                current_detections = DetectedFace(
                        xmin=0,
                        ymin=0,
                        xmax=0,
                        ymax=0,
                        conf=0,
                        class_name=None,
                        cropped_face=None
                    )
            logging.info(
                f"[OpenCVDetector] {face_no} face(s) detected in image id: {idx}, "
                f"min confidence threshold  0.25."
            )

            detections.append(current_detections)

        return detections