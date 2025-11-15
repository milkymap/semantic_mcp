from pydantic_settings import BaseSettings
from pydantic import Field


class ApiKeysSettings(BaseSettings):
    OPENAI_API_KEY: str = Field(validation_alias="OPENAI_API_KEY")
    COHERE_API_KEY: str = Field(validation_alias="COHERE_API_KEY")
    ENCRYPTION_KEY: str = Field(validation_alias="ENCRYPTION_KEY")
    API_KEY: str = Field(validation_alias="API_KEY")
    MCP_DESCRIPTOR_MODEL_NAME:str = Field("gpt-4.1-mini", validation_alias="MCP_DESCRIPTOR_MODEL_NAME")
    EMBEDDING_MODEL_NAME:str = Field("embed-v4.0", validation_alias="EMBEDDING_MODEL_NAME")
    DIMENSIONS:int = Field(1024, validation_alias="DIMENSIONS")
    INDEX_NAME: str = Field("semantic-mcp-index", validation_alias="INDEX_NAME")
    QDRANT_STORAGE_PATH: str = Field(validation_alias="QDRANT_STORAGE_PATH")
    THREAD_POOL_MAX_WORKERS:int = Field(32, validation_alias="THREAD_POOL_MAX_WORKERS")


class ApiServerSettings(BaseSettings):
    HOST:str = Field("0.0.0.0", validation_alias="API_SERVER_HOST")
    PORT:int = Field(8000, validation_alias="API_SERVER_PORT")
    WORKERS:int = Field(1, validation_alias="API_SERVER_WORKERS")