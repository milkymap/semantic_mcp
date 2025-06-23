from pydantic_settings import BaseSettings
from pydantic import Field

class ApiServerConfig(BaseSettings):
    host: str = Field(..., description="API server host", validation_alias="API_SERVER_HOST")
    port: int = Field(..., description="API server port", validation_alias="API_SERVER_PORT")
    