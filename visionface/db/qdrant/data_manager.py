import logging
import uuid
from typing import List, Dict, Union, Optional
import numpy as np
from qdrant_client.http.models import PointStruct, models

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataManager:
    """Manages data operations for Qdrant"""

    def __init__(self, client):
        self.client = client
    
    def insert_embeddings(self, 
                         collection_name: str,
                         embeddings: List[List[float]],
                         payloads: Optional[List[Dict]] = None,
                         ids: Optional[List[Union[str, int]]] = None,
                         batch_size: int = 100) -> bool:
        """
        Insert embeddings with optional payloads
        
        Args:
            collection_name: Target collection
            embeddings: List of embedding vectors
            payloads: Optional metadata for each embedding
            ids: Optional custom IDs (auto-generated if None)
            batch_size: Batch size for insertion
            
        Returns:
            bool: Success status
        """
        try:
            if not embeddings:
                logger.warning("No embeddings provided")
                return False
            
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in embeddings]
            
            # Ensure payloads list matches embeddings length
            if payloads is None:
                payloads = [{}] * len(embeddings)
            elif len(payloads) != len(embeddings):
                raise ValueError("Payloads length must match embeddings length")
            
            # Process in batches
            total_inserted = 0
            batch_size = len(embeddings) if len(embeddings)<=batch_size else batch_size

            for i in range(0, len(embeddings), batch_size):
                batch_embeddings = embeddings[i:i+batch_size]
                batch_payloads = payloads[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                points = [
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                    for point_id, embedding, payload in zip(batch_ids, batch_embeddings, batch_payloads)
                ]
                
                self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
                total_inserted += len(points)            
            logger.info(f"Successfully inserted {total_inserted} embeddings into '{collection_name}' ✅")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert embeddings: {e}")
            raise ValueError(f"Insertion failed: {e}")
    
    def delete_embeddings(self,
                         collection_name: str,
                         ids: Optional[List[Union[str, int]]] = None) -> bool:
        """
        Delete embeddings by IDs or filter conditions
        
        Args:
            collection_name: Target collection
            ids: Specific IDs to delete
            filter_conditions: Filter conditions for deletion
            
        Returns:
            bool: Success status
        """
        try:
            if ids:
                # Delete by IDs
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=ids)
                )
                logger.info(f"Deleted {len(ids)} points by ID")
            else:
                raise ValueError("Either ids must be provided for removing embeddings")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            raise ValueError(f"Deletion failed: {e}")
    

    def get_points(self, 
                   collection_name: str,
                   ids: List[Union[str, int]],
                   with_payload: bool = True,
                   with_vectors: bool = False) -> List[Dict]:
        """Retrieve specific points by ID"""
        try:
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=ids,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            
            return [
                {
                    "id": point.id,
                    "payload": point.payload if with_payload else None,
                    "vector": point.vector if with_vectors else None
                }
                for point in points
            ]
        except Exception as e:
            logger.error(f"Failed to retrieve points: {e}")
            return []