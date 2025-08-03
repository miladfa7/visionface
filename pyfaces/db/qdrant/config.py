
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional
from qdrant_client.http.models import Distance


class IndexType(Enum):
    """Available index types for payload fields"""
    TEXT = "text"
    INTEGER = "integer" 
    FLOAT = "float"
    BOOL = "bool"
    GEO = "geo"
    DATETIME = "datetime"
    
@dataclass
class ConnectionConfig:
    """Configuration for Qdrant connection"""
    host: str = "localhost"
    port: int = 6333
    url: Optional[str] = None
    api_key: Optional[str] = None
    https: bool = False
    timeout: float = 5.0
    
    def validate(self) -> bool:
        """Validate connection configuration"""
        if not self.url and not self.host:
            raise ValueError(f"Either URL or host must be provided, url: {self.url}, host: {self.host}")
        if self.port <= 0:
            raise ValueError(f"Port must be positive, You set port to {self.port}")
        if self.timeout <= 0:
            raise ValueError(f"Timeout must be positive, {self.timeout}")
        return True

@dataclass
class CollectionConfig:
    """Configuration for collection creation"""
    name: str
    vector_size: int
    hnsw_config: Optional[Dict] = None
    optimizer_config: Optional[Dict] = None
    quantization_config: Optional[Dict] = None
    payload_indexes: Optional[Dict[str, IndexType]] = None
    replication_factor: int = 1
    write_consistency_factor: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CollectionConfig':
        """Create config from dictionary"""
        return cls(**data)

    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.vector_size <= 0:
            raise ValueError(f"Vector size must be positive, {self.vector_size}")
        if self.replication_factor <= 0:
            raise ValueError(f"Replication factor must be positive, {self.replication_factor}")
        if self.write_consistency_factor <= 0:
            raise ValueError(f"Write consistency factor must be positive, {self.write_consistency_factor}")
        return True

class SearchMethod(Enum):
    """Available search methods"""
    SIMILARITY = "similarity"

@dataclass
class SearchConfig:
    """Configuration for search operations"""
    method: SearchMethod = SearchMethod.SIMILARITY
    limit: int = 10
    offset: int = 0
    with_payload: bool = True
    with_vectors: bool = False
    score_threshold: Optional[float] = None
    exact: bool = False
    hnsw_ef: Optional[int] = None
    quantization_rescore: Optional[bool] = None

    def validate(self) -> bool:
        """Validate search configuration"""
        if self.limit <= 0:
            raise ValueError("Limit must be positive")
        if self.offset < 0:
            raise ValueError("Offset must be non-negative")
        if self.score_threshold is not None and (self.score_threshold < 0 or self.score_threshold > 1):
            raise ValueError("Score threshold must be between 0 and 1")
        return True