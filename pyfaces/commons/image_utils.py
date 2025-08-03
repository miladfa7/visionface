# Part of this module is adapted from the DeepFace library
# Source: https://github.com/serengil/deepface/blob/master/deepface/commons/image_utils.py
# Original author: Alireza Makhzani and contributors

import os
from typing import Union, Tuple, IO, List
import numpy as np
import cv2
from pathlib import Path
import io
import base64
from PIL import Image
import requests
from torch.nn.functional import interpolate

from pyfaces.models.Detector import DetectedFace


def load_images(
    inputs: Union[str, np.ndarray, IO[bytes], List[Union[str, np.ndarray, IO[bytes]]]]
) -> List[Tuple[np.ndarray, str]]:
    """
    Load one or more images from various sources.

    Args:
        inputs: A single image or a list of images. Each image can be:
            - A file path (str)
            - A URL (str)
            - A base64-encoded string (str)
            - A numpy array (np.ndarray)
            - A file-like object (IO[bytes])

    Returns:
        List[np.ndarray]: A list of loaded images in BGR format
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    loaded_images = []
    for item in inputs:
        if isinstance(item, list):
            for i in item:
                if isinstance(i, DetectedFace):
                    loaded_images.append(i.cropped_face)
            continue
        elif isinstance(item, np.ndarray):
            loaded_images.append(item)
        elif hasattr(item, 'read') and callable(item.read):
            if isinstance(item, io.StringIO):
                raise ValueError("Image requires bytes, not io.StringIO.")
            img_arr = load_image_from_io_object(item)
            loaded_images.append(img_arr)
        elif isinstance(item, Path):
            img_arr = _load_from_str(str(item))
            loaded_images.append(img_arr)
        elif isinstance(item, str):
            img_arr = _load_from_str(item)
            loaded_images.append(img_arr)
        else:
            raise ValueError(f"Unsupported input type: {type(item)}")
    return loaded_images


def _load_from_str(img: str) -> np.ndarray:
    if img.startswith("data:image/"):
        return load_image_from_base64(img)
    elif img.lower().startswith(("http://", "https://")):
        return load_image_from_web(url=img)
    elif not os.path.isfile(img):
        raise ValueError(f"{img} is not exists")
    elif not img.isascii():
        raise ValueError(f"Input image must not have non-English characters - {img}")
    else:
        img_obj_bgr = cv2.imread(img)
        return img_obj_bgr


def load_image_from_io_object(obj: IO[bytes]) -> np.ndarray:
    """
    Load image from an object that supports being read
    Args:
        obj: a file like object.
    Returns:
        img (np.ndarray): The decoded image as a numpy array (OpenCV format).
    """
    try:
        _ = obj.seek(0)
    except (AttributeError, TypeError, io.UnsupportedOperation):
        seekable = False
        obj = io.BytesIO(obj.read())
    else:
        seekable = True
    try:
        nparr = np.frombuffer(obj.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img
    finally:
        if not seekable:
            obj.close()


def load_image_from_io_object(obj: IO[bytes]) -> np.ndarray:
    """
    Load image from an object that supports being read
    Args:
        obj: a file like object.
    Returns:
        img (np.ndarray): The decoded image as a numpy array (OpenCV format).
    """
    try:
        _ = obj.seek(0)
    except (AttributeError, TypeError, io.UnsupportedOperation):
        seekable = False
        obj = io.BytesIO(obj.read())
    else:
        seekable = True
    try:
        nparr = np.frombuffer(obj.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img
    finally:
        if not seekable:
            obj.close()


def load_image_from_base64(uri: str) -> np.ndarray:
    """
    Load image from base64 string.
    Args:
        uri: a base64 string.
    Returns:
        numpy array: the loaded image.
    """

    encoded_data_parts = uri.split(",")

    if len(encoded_data_parts) < 2:
        raise ValueError("format error in base64 encoded string")

    encoded_data = encoded_data_parts[1]
    decoded_bytes = base64.b64decode(encoded_data)

    # similar to find functionality, we are just considering these extensions
    # content type is safer option than file extension
    with Image.open(io.BytesIO(decoded_bytes)) as img:
        file_type = img.format.lower()
        if file_type not in {"jpeg", "png"}:
            raise ValueError(f"Input image can be jpg or png, but it is {file_type}")

    nparr = np.frombuffer(decoded_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr



def load_image_from_web(url: str) -> np.ndarray:
    """
    Loading an image from web
    Args:
        url: link for the image
    Returns:
        img (np.ndarray): equivalent to pre-loaded image from opencv (BGR format)
    """
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    image_array = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return img


def validate_images(imgs: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
    """
    Validates and standardizes image input for model processing.

    Args:
        imgs (Union[np.ndarray, List[np.ndarray]]): 
            A single image with shape (H, W, 3) or a list of such images as NumPy arrays.

    Returns:
        List[np.ndarray]: A list of validated images, each with shape (H, W, 3).
    """
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    elif not isinstance(imgs, list):
        raise ValueError(f"Expected input to be a numpy array or list, but got {type(imgs)}")

    if not imgs:
        raise ValueError("Empty image list provided for face processing!")

    for i, img in enumerate(imgs):
        if not isinstance(img, np.ndarray):
            raise ValueError(f"Image {i} is not a numpy array. Got {type(img)} instead.")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Image {i} must have shape (H, W, 3), got {img.shape}")
        
    return imgs


def get_cropped_face(img: np.ndarray, bbox: List[int]) -> np.ndarray:
    """
    Crop a face region from the input image using the detected bounding box.

    Parameters:
        img (np.ndarray): The input image in BGR or RGB format.
        bbox (List[int]): Bounding box coordinates in [x1, y1, x2, y2] format.

    Returns:
        np.ndarray: 
            The cropped face image as a NumPy array. If the bounding box is 
            partially out of bounds, it will be clipped to fit within the image dimensions.
    """
    h, w = img.shape[:2]
    x1 = max(0, bbox[0])
    y1 = max(0, bbox[1])
    x2 = min(w, bbox[2])
    y2 = min(h, bbox[3])
    cropped_face = img[y1:y2, x1:x2] 
    return cropped_face


def image_resample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data
