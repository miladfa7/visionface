"""
VisionFace: Modern Face Detection & Recognition Framework
"""

__version__ = "1.0.0"
__author__ = "VisionFace Team"
__email__ = "visio.face2025@gmail.com"


try:
    from .modules.recognition import FaceRecognition
    from .modules.detection import FaceDetection
    from .modules.embedding import FaceEmbedder
    from .modules.landmarks import LandmarkDetection
    from .annotators import FaceAnnotators

    __all__ = [
        "FaceDetection", 
        "FaceEmbedder", 
        "FaceRecognition", 
        "LandmarkDetection", 
        "FaceAnnotators"
    ]
except ImportError as e:
    print(f"Warning: Some modules could not be imported: {e}")
    __all__ = []
