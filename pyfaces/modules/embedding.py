from typing import Any, List, Union
import numpy as np

from pyfaces.models.Detector import DetectedFace
from pyfaces.models.FaceEmbedding import FaceEmbedding
from pyfaces.modules.modeling import build_model
from pyfaces.commons.image_utils import load_images, validate_images


class FaceEmbedder:
    """
    A class for generating embeddings from face images
    using a specified face embedding model.
    """
    def __init__(self, embedding_backbone: str = "FaceNet-VGG") -> None:
        """
        Initializes the FaceEmbedder with the given embedding model.

        Parameters
        ----------
        embedding_backbone : str, optional
            The name of the face embedding model to use. Default is "FaceNet-VGG".
        """
        self.face_embedder = self.build_model(embedding_backbone)
        self.vector_size = self.face_embedder.output_shape

    def build_model(self, embedding_backbone) -> Any:
        """
        Builds and returns the face embedding model.

        Parameters
        ----------
        embedding_backbone : str
            The name of the model to load.

        Returns
        -------
        Any
            An initialized face embedding model instance.
        """
        return build_model(embedding_backbone, "face_embedding")
    
    def embed_faces(
        self,
        face_imgs: Union[str, np.ndarray, List[np.ndarray], List[str], List[DetectedFace]],
        normalize_embeddings: bool = True
    ) -> FaceEmbedding:
        """
        Computes face embeddings for one or more face images.

        Parameters
        ----------
        face_imgs : Union[str, np.ndarray, List[np.ndarray], List[str], List[DetectedFace]]
            A single face image or a list of face images. Each image can be a file path (str),
            a NumPy array, or a DetectedFace object.

        normalize_embeddings : bool, optional
            Whether to apply L2 normalization to the output embeddings. Default is True.

        Returns
        -------
        FaceEmbedding
            An object containing embedding vectors for each face.
        """
        face_images = load_images(face_imgs)
        validated_images = validate_images(face_images)
        return self.face_embedder.embed(validated_images, normalize_embeddings)

