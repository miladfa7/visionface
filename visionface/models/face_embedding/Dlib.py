from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# VisionFace modules
from visionface.commons.download_files import download_model_weights
from visionface.models.FaceEmbedding import FaceEmbedder


DLIB_WEIGHTS = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
DLIB_WEIGHT_FILENAME = "dlib_resnet_v1.dat"

class DlibFaceEmbedder(FaceEmbedder):
    """
    Dlib-based face embedding model implementation.
    """
    def __init__(self):
        super().__init__()
        self.model = DlibResNetModel()
        self.model_name = "Dlib"
        self.input_shape = (150, 150)
        self.output_shape = 128

class DlibResNetModel(nn.Module):
    """
    Dlib face recognition ResNet model.
    """
    
    def __init__(self):
        self._dlib_model = self._load_dlib_model()
        
    def _load_dlib_model(self):
        """
        Load the Dlib face recognition model.
        
        Returns:
            dlib.face_recognition_model_v1: Loaded Dlib face recognition model.
        """
        try:
            import dlib
        except ModuleNotFoundError as e:
            raise ImportError(
                "Dlib is an optional dependency. Please install it using 'pip install dlib' "
                "to use the Dlib face embedder."
            ) from e
            
        # Download model weights if necessary
        weight_file_path = download_model_weights(
            filename=DLIB_WEIGHT_FILENAME,
            download_url=DLIB_WEIGHTS,
            compression_format="bz2"
        )
        return dlib.face_recognition_model_v1(str(weight_file_path))
    
    def forward(self, imgs: List[np.ndarray], normalize_embeddings: bool = True) -> List[List[float]]:
        """
        Compute face embeddings for a batch of images.

        Args:
            imgs (List[np.ndarray]): List of face images.
            normalize_embeddings (bool): Whether to apply L2 normalization to embeddings.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, 128) with face embeddings.
        """
        
        embeddings = []

        for img in imgs:
            face_descriptor = self._dlib_model.compute_face_descriptor(img)
            embedding_vector = np.array(face_descriptor, dtype=np.float32)
            embeddings.append(embedding_vector)
        
        # Convert list of arrays to tensor
        embeddings_tensor = torch.tensor(embeddings)

        if normalize_embeddings:
            embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1)

        return embeddings_tensor