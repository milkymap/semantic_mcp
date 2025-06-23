from pydantic_settings import BaseSettings
from pydantic import Field

class VectorStoreConfig(BaseSettings):
    embedding_model_name: str = Field(..., description="Embedding model", validation_alias="EMBEDDING_MODEL_NAME")
    embedding_dimensions: int = Field(..., description="Embedding dimensions", validation_alias="EMBEDDING_DIMENSIONS")
    cache_folder: str = Field(..., description="HF home for cache folder", validation_alias="HF_HOME")
    device: str = Field(..., description="Device to use for embedding", validation_alias="DEVICE")
    nb_workers: int = Field(..., description="Number of workers to use for embedding", validation_alias="EMBEDDING_NB_WORKERS")
    qdrant_url: str = Field(..., description="Qdrant URL", validation_alias="QDRANT_URL")
    qdrant_api_key: str = Field(..., description="Qdrant API key", validation_alias="QDRANT_API_KEY")
    qdrant_services_collection_name: str = Field(..., description="Qdrant collection name for mcp services", validation_alias="QDRANT_SERVICES_COLLECTION_NAME")
    qdrant_tools_collection_name: str = Field(..., description="Qdrant collection name for mcp tools", validation_alias="QDRANT_TOOLS_COLLECTION_NAME")
    
 