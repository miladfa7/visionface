from typing import Optional, List, Dict, Union
import logging

import numpy as np 
from pyfaces.db.qdrant.config import CollectionConfig, ConnectionConfig, SearchConfig
from pyfaces.db.qdrant.data_manager import DataManager
from pyfaces.db.qdrant.search_manager import SearchManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class QdrantVectorDB:
    """
    Qdrant Vector Database for face vector storage and search operation
    """
    def __init__(self, **kwargs):
        try:
            try:
                from qdrant_client import QdrantClient
                from pyfaces.db.qdrant.collection_manager import CollectionManager
            except ImportError:
                logger.error("Please install qdrant-client: pip install qdrant-client")
                raise
            
            host = kwargs.get("host", "localhost")
            port = kwargs.get("port", 6333)
            url = kwargs.get("url", None)
            api_key = kwargs.get("api_key", None)
            https = kwargs.get("https", False)
            timeout = kwargs.get("timeout", 5.0)

            # Create connection config
            self.config = ConnectionConfig(
                host=host, port=port, url=url, 
                api_key=api_key, https=https, timeout=timeout
            )
            self.config.validate()
            
            # Initialize Qdrant client
            if url:
                self.client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
            else:
                self.client = QdrantClient(
                    host=host, 
                    port=port, 
                    https=https,
                    api_key=api_key,
                    timeout=timeout
                )
            
            # Initialize managers
            self.collections = CollectionManager(self.client)
            self.search = SearchManager(self.client)
            self.data = DataManager(self.client)
            
            logger.info(f"Connected to Qdrant at {url or f'{host}:{port}'}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Connection failed: {e}")
        
    def create_collection(
        self, 
        collection_name: str, 
        vector_size: int
    ) -> bool:
        """Create a new collection"""
        config = CollectionConfig(
            name=collection_name,
            vector_size=vector_size,
        )
        if not self.collection_exists(collection_name):
            self.collections.create_collection(config)
        else:
            logger.info(f"Collection [{collection_name}] already exists! ✅")

    def list_collections(self) -> List[str]:
        """List all collections"""
        return self.collections.list_collections()
        
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        return self.collections.delete_collection(collection_name)
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """Get collection information"""
        return self.collections.get_collection_info(collection_name)
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        return self.collections.collection_exists(collection_name)
    
    def insert_embeddings(self, 
                         collection_name: str,
                         embeddings: List[List[float]],
                         payloads: Optional[List[Dict]] = None,
                         ids: Optional[List[Union[str, int]]] = None,
                         batch_size: int = 100) -> bool:
        """Insert embeddings with optional payloads"""
        return self.data.insert_embeddings(
            collection_name, embeddings, payloads, ids, batch_size
        )
    
    def search_embeddings(self,
                         collection_name: str,
                         query_vectors: List[np.ndarray],
                         score_threshold: Optional[float] = None,
                         top_k: int = 5) -> List[Dict]:
        """Search embeddings using various methods"""
        config: SearchConfig = SearchConfig()
        config.limit = top_k
        config.score_threshold = score_threshold
        return self.search.search_embeddings(
            collection_name, query_vectors, config
        )