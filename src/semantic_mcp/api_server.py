import asyncio 

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status 
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from uvicorn import Server, Config 

from .api_engine import APIEngine
from .settings import ApiServerSettings, ApiKeysSettings
from .api_server_constants import TITLE, VERSION, DESCRIPTION

from .api_routers.mcp_handler import MCPhandler

class APIServer:
    def __init__(self, api_keys_settings:ApiKeysSettings, api_server_settings:ApiServerSettings):
        self.api_server_settings = api_server_settings
        self.api_keys_settings = api_keys_settings 
        self.app = FastAPI(
            title=TITLE,
            version=VERSION,
            description=DESCRIPTION,
            lifespan=self.lifespan
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins="*",
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    @asynccontextmanager
    async def lifespan(self, app:FastAPI):
        async with APIEngine(api_keys_settings=self.api_keys_settings) as api_engine:
            self.api_engine = api_engine
            await self.define_routes(app=app, api_engine=api_engine)
            yield 

    async def define_routes(self, app:FastAPI, api_engine:APIEngine):
        @app.get("/health")
        async def health_check():
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": "ok",
                    "message": "API server is healthy"
                }
            )

        mcp_handler = MCPhandler(api_engine=api_engine)
        app.include_router(mcp_handler, prefix="/api")
    
    async def listen(self):
        config = Config(
            app=self.app,
            host=self.api_server_settings.HOST,
            port=self.api_server_settings.PORT,
            workers=self.api_server_settings.WORKERS
        )
        server = Server(config)
        await server.serve()

