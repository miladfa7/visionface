from typing import List, Union
import numpy as np

from pyfaces.models.FaceEmbedding import FaceEmbedding
from pyfaces.modules.modeling import build_model
from pyfaces.commons.image_utils import load_images

def embed_faces(
        face_imgs: Union[str, np.ndarray, List[np.ndarray], List[str]],
        model_name: str = "FaceNet-VGG",
        normalize_embeddings: bool = True
) -> FaceEmbedding:
        """
        Compute face embeddings for one or more face images.

        Args:
                face_imgs: A single face image or a list of images as file paths (str) or numpy arrays.
                model_name: Name of the embedding model to use.
                normalize_embeddings: Whether to apply L2 normalization to embeddings.

        Returns:
                FaceEmbedding: Embedding objects for each face.
        """
        face_images = load_images(face_imgs)
        face_embedder = build_model(model_name, "face_embedding")
        embeded_faces = face_embedder.embed(face_images, normalize_embeddings)
        return embeded_faces
