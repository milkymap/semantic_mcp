from typing import Self 
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, AsyncExitStack
from .settings import ApiKeysSettings

from .services.embedding import EmbeddingService
from .services.mcp_descriptor import MCPDescriptorService
from .services.vector_store import VectorStoreService
from .services.encryption import EncryptionService

from .middleware.auth_middleware import AuthMiddleware

from .log import logger

class APIEngine:
    def __init__(self, api_keys_settings:ApiKeysSettings):
        self.api_keys_settings = api_keys_settings
        self.auth_middleware = AuthMiddleware(api_key=self.api_keys_settings.API_KEY)

    
    async def __aenter__(self) -> Self:
        self.threads_pool = ThreadPoolExecutor(
            max_workers=self.api_keys_settings.THREAD_POOL_MAX_WORKERS
        )
        self.async_exit_stack = AsyncExitStack()

        self.vector_store_service = await self.async_exit_stack.enter_async_context(
            VectorStoreService(
                index_name=self.api_keys_settings.INDEX_NAME,
                dimensions=self.api_keys_settings.DIMENSIONS,
                qdrant_storage_path=self.api_keys_settings.QDRANT_STORAGE_PATH
            )
        )
        self.embedding_service = await self.async_exit_stack.enter_async_context(
            EmbeddingService(
                api_key=self.api_keys_settings.COHERE_API_KEY,
                embedding_model_name=self.api_keys_settings.EMBEDDING_MODEL_NAME,
                dimension=self.api_keys_settings.DIMENSIONS
            )
        )
        self.mcp_descriptor_service = await self.async_exit_stack.enter_async_context(
            MCPDescriptorService(
                openai_api_key=self.api_keys_settings.OPENAI_API_KEY,
                openai_model_name=self.api_keys_settings.MCP_DESCRIPTOR_MODEL_NAME
            )
        )
        self.encryption_service = await self.async_exit_stack.enter_async_context(
            EncryptionService(encryption_key=self.api_keys_settings.ENCRYPTION_KEY)
        )
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(f"Exception in APIEngine context manager: {exc_value}")
            logger.exception(traceback)
        await self.async_exit_stack.aclose()
        self.threads_pool.shutdown()