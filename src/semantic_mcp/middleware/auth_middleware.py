from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..log import logger


class AuthMiddleware:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.security = HTTPBearer()

        if not self.api_key:
            raise ValueError("API_KEY must be provided")

    def verify_api_key(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        if credentials.credentials != self.api_key:
            logger.warning("Invalid API key attempt")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key",
                headers={"WWW-Authenticate": "Bearer"}
            )
        return credentials.credentials