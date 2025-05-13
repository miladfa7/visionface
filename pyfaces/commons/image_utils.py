import os
from typing import Union, Tuple, IO
import numpy as np
import cv2
from pathlib import Path
import io
import base64
from PIL import Image
import requests



def load_image(img: Union[str, np.ndarray, IO[bytes]]) -> Tuple[np.ndarray, str]:
    """
    Load image from path, url, file object, base64 or numpy array.
    Args:
        img: a path, url, file object, base64 or numpy array.
    Returns:
        image (numpy array): the loaded image in BGR format
        image name (str): image name itself
    """
    if isinstance(img, np.ndarray):
        return img

    if hasattr(img, 'read') and callable(img.read):
        if isinstance(img, io.StringIO):
            raise ValueError(
                'img requires bytes and cannot be an io.StringIO object.'
            )
        return load_image_from_io_object(img), 'io object'

    if isinstance(img, Path):
        img = str(img)

    if not isinstance(img, str):
        raise ValueError(f"img must be numpy array or str but it is {type(img)}")

    # The image is a base64 string
    if img.startswith("data:image/"):
        return load_image_from_base64(img), "base64 encoded string"

    # The image is a url
    if img.lower().startswith(("http://", "https://")):
        return load_image_from_web(url=img), img

    # The image is a path
    if not os.path.isfile(img):
        raise ValueError(f"Confirm that {img} exists")

    # image must be a file on the system then

    # image name must have english characters
    if not img.isascii():
        raise ValueError(f"Input image must not have non-english characters - {img}")

    img_obj_bgr = cv2.imread(img)
    # img_obj_rgb = cv2.cvtColor(img_obj_bgr, cv2.COLOR_BGR2RGB)
    return img_obj_bgr, img


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