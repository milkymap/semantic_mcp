import click 
import asyncio 
import json 

from dotenv import load_dotenv

from src.semantic_mcp.settings import CredentialsConfig, VectorStoreConfig, ApiServerConfig
from src.semantic_mcp.backend.mcp_router import MCPRouter
from src.semantic_mcp.backend.mcp_types import McpServers, McpConfig
from src.semantic_mcp.backend.mq_embedding import MQEmbedding
from src.semantic_mcp.backend.mcp_api_server import start_mcp_server


# python -m src.semantic_mcp launch-mcp-router -j mcp-servers.json

@click.group()
@click.pass_context
def handler(ctx:click.Context):
    ctx.ensure_object(dict)
    ctx.obj["credentials"] = CredentialsConfig()
    ctx.obj["vector_store"] = VectorStoreConfig()
    ctx.obj["api_server"] = ApiServerConfig()


@handler.command()
@click.pass_context
def launch_embedding_service(ctx:click.Context):
    mq_embedding = MQEmbedding(
        embedding_model_name=ctx.obj["vector_store"].embedding_model_name,
        cache_folder=ctx.obj["vector_store"].cache_folder,
        device=ctx.obj["vector_store"].device,
        nb_workers=ctx.obj["vector_store"].nb_workers
    )
    mq_embedding.run_loop()


@handler.command()
@click.option("--path2json_file_config", "-j" ,type=click.Path(exists=True), help="Path to json file config")
@click.pass_context
def launch_mcp_router(ctx:click.Context, path2json_file_config:str):
    with open(path2json_file_config, "r") as f:
        mcp_servers = json.load(f)
    mcp_servers = McpServers(**mcp_servers)

    async def main():
        mcp_router = MCPRouter(
            mcp_servers=mcp_servers,
            vector_store_config=ctx.obj["vector_store"],
            credentials=ctx.obj["credentials"]
        )
        async with mcp_router as mcp_router:
            await mcp_router.init_mcp_servers()
            await mcp_router.export_tools("tools.txt")
            

    asyncio.run(main())


@handler.command()
@click.option("--path2json_file_config", "-j" ,type=click.Path(exists=True), help="Path to json file config")
@click.pass_context
def launch_api_server(ctx:click.Context, path2json_file_config:str):
    with open(path2json_file_config, "r") as f:
        mcp_servers = json.load(f)
    mcp_servers = McpServers(**mcp_servers)

    start_mcp_server(
        mcp_servers=mcp_servers,
        vector_store_config=ctx.obj["vector_store"],
        credentials=ctx.obj["credentials"]
    )


if __name__ == "__main__":
    load_dotenv()
    handler(
        obj={}
    )