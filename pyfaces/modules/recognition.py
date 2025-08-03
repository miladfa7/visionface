


from typing import Any, Dict, List, Optional, Union

import numpy as np
from torch import embedding

from pyfaces.commons.detection_utils import select_max_conf_faces
from pyfaces.models.Detector import DetectedFace
from pyfaces.modules.detection import FaceDetection
from pyfaces.modules.embedding import FaceEmbedder
from pyfaces.commons.image_utils import load_images, validate_images

from pyfaces.modules.modeling import build_model


class FaceRecognition:
    """    
    FaceRecognition pipeline for face detection, embedding, storage, and search.
    """
    def __init__(
        self, 
        detector_backbone: str = "yolo-small",
        embedding_backbone: str = "FaceNet-VGG",
        db_backend: str = "qdrant",
        db_config: Optional[Dict] = None
    ) -> None:
        """
        Initialize the face recognition system with specified components.

        Args:
            detector_backbone: 
                Backbone name for the face detector (e.g., "yolo-small", "mtcnn").
            embedding_backbone: 
                Backbone name for the face embedder (e.g., "FaceNet-VGG", "ArcFace").
            db_backend: 
                Database backend name. Supported values include:
                - 'qdrant'
                - 'milvus'
                - 'file'
            db_config: 
                Optional dictionary for configuring the vector database connection.
                This is primarily used when `db_backend='qdrant'`. Supported keys include:

                - host (str): Hostname of the Qdrant server. Default is `"localhost"`.
                - port (int): Port number of the Qdrant server. Default is `6333`.
                - url (str, optional): Full URL (overrides host and port if provided).
                - api_key (str, optional): API key for secure Qdrant access.
                - https (bool): Whether to use HTTPS instead of HTTP. Default is `False`.
                - timeout (float): Timeout duration in seconds for requests. Default is `5.0`.
        """
        self.face_detector = FaceDetection(detector_backbone=detector_backbone)
        self.face_embedder = FaceEmbedder(embedding_backbone=embedding_backbone)
        self.db = self._init_db_backend(db_backend, db_config or {})
        
    def _init_db_backend(self, db_backend: str, db_config: Dict) -> Any:
        """
        Initializes the vector database backend.

        Args:
            db_backend: The name of the backend (e.g., 'qdrant').
            db_config: Configuration parameters for the backend.

        Returns:
            A vector database client instance.
        """
        if db_backend == "qdrant":
            from pyfaces.db.qdrant_client import QdrantVectorDB
            return QdrantVectorDB(**db_config)
        elif db_backend == "milvus":
            pass
        elif db_backend == "file":
            pass
        else:
            raise ValueError(f"Unsupported DB backend: {db_backend}")
    
    def _compute_embeddings(
        self, 
        images: Union[str, np.ndarray, List[np.ndarray], List[str]], 
        normalize_embeddings: bool = True
    ) -> List[List[float]]:
        """
        Detects and embeds the most confident face in each image.

        Args:
            images: Image(s) as file path(s) or NumPy array(s).
            normalize: Whether to normalize the embedding vectors.

        Returns:
            List of face embedding vectors.
        """
        detections = self.face_detector.detect_faces(images, return_cropped_faces=True)
        top_faces = select_max_conf_faces(detections)
        embeddings = self.face_embedder.embed_faces(top_faces, normalize_embeddings=normalize_embeddings)
        return embeddings.to_list()
    
    def upsert_faces(
        self,
        images: Union[str, np.ndarray, List[np.ndarray], List[str]],  
        labels: Union[str, List[str]],
        collection_name: str,
        batch_size: int = 10,
        normalize_embeddings: bool = True
    ) -> None:
        """
        Detect, embed, and store faces in a collection with automatic face selection and upserting.

        Parameters:
        ----------
            images (Union[str, np.ndarray, List[np.ndarray], List[str]]): 
            Input image(s) containing faces to process and store. Can be:
            - str: Path to a single image file
            - np.ndarray: Single image as a numpy array (H, W, C format expected)
            - List[np.ndarray]: Multiple images as numpy arrays
            - List[str]: Multiple image file paths
            
        labels (Union[str, List[str]]): 
            Label(s) to associate with the detected faces. 
            
        collection_name (str): 
            Name of the face collection where embeddings will be stored. If the collection
            doesn't exist, it will be created automatically.
            
        batch_size (int, optional): 
            Number of images to process simultaneously in each batch. Larger batch sizes
            can improve processing speed but require more memory. Defaults to 10.
            
        normalize_embeddings (bool, optional): 
            Whether to L2-normalize the computed face embeddings before storage. Defaults to True.

        Returns:
        ----------
            None: This method doesn't return a value but modifies the collection state.
        """
        vector_size = self.face_embedder.vector_size
        self.db.create_collection(collection_name, vector_size=vector_size)

        embeddings = self._compute_embeddings(images, normalize_embeddings)
        payloads = [{"face_name": label} for label in labels]

        self.db.insert_embeddings(
            collection_name=collection_name,
            embeddings=embeddings,
            payloads=payloads,
            batch_size=batch_size
        )
        
        
    def search_faces(
        self,
        images: Union[str, np.ndarray, List[np.ndarray], List[str]], 
        collection_name: str,
        score_threshold: Optional[float] = None,
        top_k: int = 5,
        ) -> List[Dict]:
        """
        Search for similar faces in a specified collection using facial recognition embeddings.
        
        Parameters
        ----------
        images : Union[str, np.ndarray, List[str], List[np.ndarray]]
                A single image or a list of images. Each image can be either a file path (str)
                or an image array.
        collection_name (str): 
            Name of the face collection to search within. The collection must exist
            and contain pre-indexed face embeddings.

        score_threshold (Optional[float], optional): 
            Minimum similarity score threshold for returned matches. Only faces with
            similarity scores above this threshold will be included in results.
            If None, no filtering is applied. Range typically [0.0, 1.0] where
            higher values indicate greater similarity. Defaults to None.
        
        top_k (int, optional): 
            Maximum number of most similar faces to return per input image.
            Results are ordered by similarity score in descending order.
            Defaults to 5.

        Returns:
        ----------
            List[Dict]: 
                List of search results, one dictionary per input image. Each dictionary
                contains the top-k most similar faces found in the collection.
        """
        embeddings = self._compute_embeddings(images)
        return self.db.search_embeddings(
            collection_name=collection_name,
            query_vectors=embeddings,
            score_threshold=score_threshold,
            top_k=top_k
        )



        
