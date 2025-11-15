import numpy as np 
from numpy.typing import NDArray

from openai import AsyncOpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.embedding import Embedding

from cohere.client_v2 import AsyncClientV2 
from cohere.types.embed_response import EmbedResponse

from typing import List, Self
from ..log import logger

class EmbeddingService:
    def __init__(self, api_key: str, embedding_model_name:str, dimension:int):
        self.api_key = api_key
        self.embedding_model_name = embedding_model_name
        self.dimension = dimension

    async def __aenter__(self) -> Self:
        self.client = AsyncClientV2(api_key=self.api_key)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(f"Exception in EmbeddingService context manager: {exc_value}")
            logger.exception(traceback)

    async def create_embedding(self, texts: list[str], input_type="search_document") -> List[List[float]]:
        embedding_response:EmbedResponse = await self.client.embed(
            model=self.embedding_model_name,
            texts=texts,
            input_type=input_type,
            output_dimension=self.dimension,
            embedding_types=["float"]
        )
        embeddings = embedding_response.embeddings.float_
        return embeddings

        # response:CreateEmbeddingResponse = await self.client.embeddings.create(
        #     input=texts,
        #     model=self.embedding_model_name,
        #     dimensions=self.dimension
        # )
        # embeddings = [item.embedding for item in response.data]
        # return embeddings
    
    def weighted_embedding(self, base_embedding:List[float], corpus_embeddings:List[List[float]], alpha:float=0.1) -> List[List[float]]:
        base_embedding_array = np.array(base_embedding)
        corpus_embeddings_array = np.array(corpus_embeddings)
        weighted_corpus_embeddings = alpha * base_embedding_array + (1 - alpha) * corpus_embeddings_array
        return weighted_corpus_embeddings.tolist()