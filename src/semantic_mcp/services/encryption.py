import json
from typing import Dict, Any, Self
from cryptography.fernet import Fernet

from ..log import logger


class EncryptionService:
    def __init__(self, encryption_key: str):
        self.encryption_key = encryption_key
        self.fernet = None

    async def __aenter__(self) -> Self:
        if not self.encryption_key:
            raise ValueError("Encryption key not provided")

        try:
            if isinstance(self.encryption_key, str):
                encryption_key_bytes = self.encryption_key.encode('utf-8')
            else:
                encryption_key_bytes = self.encryption_key
            self.fernet = Fernet(encryption_key_bytes)
        except Exception as e:
            raise ValueError(f"Invalid encryption key format: {str(e)}")

        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(f"Exception in EncryptionService context manager: {exc_value}")
            logger.exception(traceback)
        self.fernet = None
        self.encryption_key = None

    def encrypt(self, data: Dict[str, Any]) -> str:
        if self.fernet is None:
            raise ValueError("EncryptionService not properly initialized. Use async context manager.")

        try:
            json_str = json.dumps(data, sort_keys=True)
            data_bytes = json_str.encode('utf-8')
            encrypted_data = self.fernet.encrypt(data_bytes)
            return encrypted_data.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to encrypt data: {str(e)}")

    def decrypt(self, encrypted_str: str) -> Dict[str, Any]:
        if self.fernet is None:
            raise ValueError("EncryptionService not properly initialized. Use async context manager.")

        try:
            encrypted_bytes = encrypted_str.encode('utf-8')
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            json_str = decrypted_bytes.decode('utf-8')
            data = json.loads(json_str)
            return data
        except Exception as e:
            raise ValueError(f"Failed to decrypt data: {str(e)}")