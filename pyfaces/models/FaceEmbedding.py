from abc import ABC
from typing import List, Union, Any, Tuple
from dataclasses import dataclass
import numpy as np 
import cv2
import torch
from torchvision.transforms import functional as F

# pyfaces module 
from pyfaces.commons.image_utils import validate_images

class FaceEmbedder(ABC):
    model: Any
    model_name: str
    input_shape: Tuple[int, int]
    output_shape: int

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _img_preprocess(self, imgs: Union[np.ndarray, List[np.ndarray]]) -> Union[torch.Tensor, np.ndarray]:
        """
        Preprocess input images based on the model type.

        For PyTorch models:
            - Converts images to normalized float tensors [0, 1]
            - Returns a batched tensor of shape (N, 3, H, W)

        For Dlib models:
            - Ensures each image is resized to the target input shape
            - Returns a list of RGB images as NumPy arrays

        Args:
            imgs (List[np.ndarray]): List of images in BGR format (OpenCV)

        Returns:
            Union[torch.Tensor, List[np.ndarray]]: Preprocessed inputs ready for embedding
        """
        target_h, target_w = self.input_shape
        batch_size = len(imgs)

        if self.model_name=="Dlib":
            batch_inputs = []
            for img in imgs:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.shape[:2] != (target_h, target_w):
                    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                batch_inputs.append(img)
            return batch_inputs
        else:
            batch_tensor = torch.empty(batch_size, 3, target_h, target_w, dtype=torch.float32, device=self.device)
            for i, img in enumerate(imgs):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.shape[:2] != (target_h, target_w):
                    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                batch_tensor[i] = img_tensor
            return batch_tensor
        
    @torch.no_grad()
    def embed(self, imgs: Union[np.ndarray, List[np.ndarray]], normalize_embeddings: bool = True) -> 'FaceEmbedding':
        """
        Generate face embeddings from one or more face images.

        Args:
            imgs (Union[np.ndarray, List[np.ndarray]]): 
                A single image with shape (H, W, 3) or a list of such images in BGR format.
            normalize_embeddings (bool, optional): 
                If True, applies L2 normalization to the output embeddings. Default is True.

        Returns:
            FaceEmbedding: 
                An object containing the computed embedding tensor(s) with shape (N, D), 
                where N is the number of input images and D is the embedding dimension (e.g., 128 or 512).
        """
        
        # Validate input images
        imgs = validate_images(imgs)

        # Preprocess images depending on model type
        batch_inputs = self._img_preprocess(imgs) 

        # Compute embeddings using the model's forward
        embeddings = self.model.forward(batch_inputs, normalize_embeddings)

        return FaceEmbedding(embeddings)
        

@dataclass
class FaceEmbedding:
    embeddings: torch.Tensor 

    def __getitem__(self, idx):
        """Get embedding vector(s) at index idx (supports int or slice)."""
        return self.embeddings[idx]

    def batch_size(self) -> int:
        """Returns the batch size (number of embeddings)."""
        return self.embeddings.size(0)

    def to(self, device: torch.device):
        """Returns a new FaceEmbedding on the given device."""
        return FaceEmbedding(self.embeddings.to(device))

    def cpu(self):
        """Move embeddings to CPU."""
        return self.to(torch.device('cpu'))

    def cuda(self):
        """Move embeddings to CUDA device."""
        return self.to(torch.device('cuda'))

    def as_numpy(self):
        """Return embeddings as a NumPy array (on CPU)."""
        return self.embeddings.detach().cpu().numpy()
    
    def to_list(self):
        """Return embeddings as a list of lists (compatible with Qdrant)."""
        return self.embeddings.detach().cpu().numpy().tolist()