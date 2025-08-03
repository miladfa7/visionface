import logging
from typing import List, Dict, Union, Optional
import numpy as np

from pyfaces.db.qdrant.config import SearchConfig, SearchMethod


logger = logging.getLogger(__name__)


class SearchManager:
    """Manages search operations for Qdrant"""
    
    def __init__(self, client):
        self.client = client
    
    def search_embeddings(self,
                         collection_name: str,
                         query_vectors: Union[np.ndarray, List[float]],
                         config: SearchConfig = SearchConfig()) -> List[Dict]:
        """
        Search embeddings using various methods
        
        Args:
            collection_name: Target collection
            query_vectors: Query embedding vector
            config: Search configuration

        Returns:
            List[Dict]: Search results
        """
        try:
            # Validate config
            config.validate()
            results = []            
            if config.method == SearchMethod.SIMILARITY:
                for query_vector in query_vectors:
                    results.extend(
                        self._similarity_search(collection_name, query_vector, config)
                    )
            else:
                raise ValueError(f"Unsupported search method: {config.method}")
            
            formatted_results = self._format_results(results, config)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise ValueError(f"Search operation failed: {e}")
    
    def _similarity_search(self, collection_name: str, query_vector: List[float], 
                          config: SearchConfig) -> List:
        """Perform similarity search"""
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=config.limit,
            offset=config.offset,
            with_payload=config.with_payload,
            with_vectors=config.with_vectors,
            score_threshold=config.score_threshold
        )
    
    def _format_results(self, results: List, config: SearchConfig) -> List[Dict]:
        """Format search results"""
        formatted_results = []
        for result in results:
            formatted_result = {
                "id": result.id,
                "face_name": result.payload["face_name"] if config.with_payload else None,
                "score": getattr(result, 'score', None),
                "vector": result.vector if config.with_vectors else None
            }
            formatted_results.append(formatted_result)
        return formatted_results