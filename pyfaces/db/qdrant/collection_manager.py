import logging 

from typing import Any, Dict, List, Optional
from qdrant_client.http.models import (
    VectorParams, HnswConfigDiff, OptimizersConfigDiff,
    ScalarQuantization, ProductQuantization, BinaryQuantization,
)
from qdrant_client.http.models import Distance
from pyfaces.db.qdrant.config import CollectionConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CollectionManager:
    """Manages collection operations for Qdrant"""
    
    def __init__(self, client):
        self.client = client

    def create_collection(self, config: CollectionConfig) -> bool:
        """
        Create a new collection with configuration
        
        Args:
            config: Collection configuration
            
        Returns:
            bool: Success status
        """
        try:
            # Validate configuration
            config.validate()

            # Build vector params
            vectors_config = VectorParams(
                size=config.vector_size,
                distance=Distance.COSINE
            )
            
            # Build HNSW config
            hnsw_config = None
            if config.hnsw_config:
                hnsw_config = HnswConfigDiff(**config.hnsw_config)
            
            # Build optimizer config
            optimizer_config = None
            if config.optimizer_config:
                optimizer_config = OptimizersConfigDiff(**config.optimizer_config)
            
            # Build quantization config
            quantization_config = self._build_quantization_config(config.quantization_config)
            
            # Create collection
            self.client.create_collection(
                collection_name=config.name,
                vectors_config=vectors_config,
                hnsw_config=hnsw_config,
                optimizers_config=optimizer_config,
                quantization_config=quantization_config,
                replication_factor=config.replication_factor,
                write_consistency_factor=config.write_consistency_factor
            )
            
            logger.info(f"Collection '{config.name}' created successfully ✅")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection '{config.name}': {e}")
            if "Connection refused" in str(e):
                logger.error(
                    "Qdrant connection was refused. Make sure the Qdrant server is running.\n"
                    "To start it with Docker, run Qdrant server locally with docker:\n"
                    "docker run -d -p 6333:6333 qdrant/qdrant:latest",
                    "See more launch options in, https://github.com/qdrant/qdrant#usage"
                )

            raise ValueError(f"Collection creation failed: {e}")
    
    def _build_quantization_config(self, quantization_config: Optional[Dict]) -> Optional[Any]:
        """Build quantization configuration"""
        if not quantization_config:
            return None
        
        quant_type = quantization_config.get("type", "scalar")
        
        if quant_type == "scalar":
            return ScalarQuantization(scalar=quantization_config)
        elif quant_type == "product":
            return ProductQuantization(product=quantization_config)
        elif quant_type == "binary":
            return BinaryQuantization(binary=quantization_config)
        else:
            raise ValueError(f"Unknown quantization type: {quant_type}")
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """Get detailed collection information"""
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "status": info.status,
                "optimizer_status": info.optimizer_status,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "config": {
                    "params": info.config.params.__dict__ if info.config.params else None,
                    "hnsw_config": info.config.hnsw_config.__dict__ if info.config.hnsw_config else None,
                    "optimizer_config": info.config.optimizer_config.__dict__ if info.config.optimizer_config else None,
                    "quantization_config": str(info.config.quantization_config) if info.config.quantization_config else None
                },
                "payload_schema": info.payload_schema
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise ValueError(f"Collection '{collection_name}' not found")
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            collections = self.client.get_collections()
            return [collection.name for collection in collections.collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Collection '{collection_name}' deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        try:
            self.client.get_collection(collection_name)
            return True
        except:
            return False
    
    def refresh_collection(self, collection_name: str) -> bool:
        """Refresh collection (optimize indexes)"""
        try:
            self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=20000
                )
            )
            logger.info(f"Collection '{collection_name}' refresh initiated")
            return True
        except Exception as e:
            logger.error(f"Failed to refresh collection '{collection_name}': {e}")
            return False
