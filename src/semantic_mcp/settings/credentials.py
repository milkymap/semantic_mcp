from pydantic_settings import BaseSettings
from pydantic import Field

class CredentialsConfig(BaseSettings):
    openai_api_key: str = Field(..., description="OpenAI API key", validation_alias="OPENAI_API_KEY")