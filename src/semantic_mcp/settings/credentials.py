from pydantic_settings import BaseSettings
from pydantic import Field

class CredentialsConfig(BaseSettings):
    openai_api_key: str = Field(..., description="OpenAI API key", validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(..., description="Anthropic API key", validation_alias="ANTHROPIC_API_KEY")
    gemini_api_key: str = Field(..., description="Google API key", validation_alias="GEMINI_API_KEY")
    