import asyncio 
import click 

from dotenv import load_dotenv

from .api_server import APIServer
from .settings import ApiKeysSettings, ApiServerSettings

@click.command()
def main() -> None:
    load_dotenv()
    api_keys_settings = ApiKeysSettings()
    api_server_settings = ApiServerSettings()
    async def run_server():
        api_server = APIServer(
            api_keys_settings=api_keys_settings,
            api_server_settings=api_server_settings
        )
        await api_server.listen()
    
    asyncio.run(run_server())