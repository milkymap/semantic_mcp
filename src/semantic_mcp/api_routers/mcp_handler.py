from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import asyncio
import time
from uuid import uuid5, UUID, NAMESPACE_DNS

import yaml

from fastapi import APIRouter, HTTPException, status, Path, Query, Depends
from fastapi.responses import JSONResponse
from qdrant_client import models

from numpy.typing import NDArray
import numpy as np

from ..api_engine import APIEngine
from ..services.types import McpServerDescription, McpStartupConfig, DescribeMcpServerResponse
from ..log import logger
from .mcp_handler_schema import (
    AddMCPServerRequest,
    UpdateMCPServerRequest,
    MCPServerResponse,
    MCPServerListResponse,
    MCPServerInfo,
    MCPStatisticsResponse,
    MCPToolResponse,
    MCPToolInfo,
    MCPToolListResponse,
    MCPCommandResponse,
    SearchServersRequest,
    SearchToolsRequest,
    SearchServersResponse,
    SearchToolsResponse,
    SearchResultServer,
    SearchResultTool
)

"""
MCP Handler - Provides REST API endpoints for managing MCP servers and tools.

Available endpoints (all prefixed with /api):
- POST /api/mcp/servers - Add new MCP server
- GET /api/mcp/servers - List all servers (paginated)
- GET /api/mcp/servers/{server_name} - Get server details
- GET /api/mcp/servers/{server_name}/tools - List all tools for a server (paginated)
- GET /api/mcp/servers/{server_name}/tools/{tool_name} - Get specific tool details
- GET /api/mcp/servers/{server_name}/command - Get decrypted startup command
- PUT /api/mcp/servers/{server_name} - Update server configuration
- DELETE /api/mcp/servers/{server_name} - Remove server and tools
- GET /api/mcp/statistics - Get server and tool count statistics
- POST /api/mcp/servers/search - Semantic search for servers
- POST /api/mcp/tools/search - Semantic search for tools
"""

class MCPhandler(APIRouter):
    def __init__(self, api_engine: APIEngine):
        super(MCPhandler, self).__init__(tags=["MCP Servers"])
        self.api_engine = api_engine
        self.define_routes()

    def _build_startup_config(self, command: str, args: List[str], env: Dict[str, str]) -> McpStartupConfig:
        return McpStartupConfig(
            command=command,
            args=args,
            env=env
        )

    async def _describe_and_embed_server(self, server_name: str, startup_config: McpStartupConfig, timeout: int, alpha: float):
        describe_response = await self.api_engine.mcp_descriptor_service.describe_mcp_server(
            server_name=server_name,
            mcp_startup_config=startup_config,
            timeout=timeout
        )

        if not describe_response:
            raise Exception(f"Failed to connect or analyze MCP server '{server_name}'")

        server_description = describe_response.server_description
        nb_tools = len(describe_response.tools.tools)

        server_text = yaml.dump(server_description, sort_keys=False)
        server_embeddings = await self.api_engine.embedding_service.create_embedding([server_text])
        server_embedding = server_embeddings[0]

        enhanced_tool_descriptions = []
        tool_embeddings = []

        if describe_response.tools and describe_response.tools.tools:
            enhance_tool_tasks = []
            for tool in describe_response.tools.tools:
                enhance_tool_tasks.append(
                    self.api_engine.mcp_descriptor_service.enhance_tool(
                        server_name=server_name,
                        tool_name=tool.name,
                        tool_description=tool.description,
                        tool_schema=tool.inputSchema
                    )
                )

            enhanced_tool_descriptions = await asyncio.gather(*enhance_tool_tasks, return_exceptions=True)
            if any(isinstance(result, Exception) for result in enhanced_tool_descriptions):
                raise Exception("Error enhancing tool descriptions")

            tool_embedding_inputs = [desc for desc in enhanced_tool_descriptions]
            tool_embeddings_raw = await self.api_engine.embedding_service.create_embedding(tool_embedding_inputs)
            tool_embeddings = self.api_engine.embedding_service.weighted_embedding(
                base_embedding=server_embedding,
                corpus_embeddings=tool_embeddings_raw,
                alpha=alpha
            )

        startup_config_data = {
            "command": startup_config.command,
            "args": startup_config.args,
            "env": startup_config.env
        }
        encrypted_startup_config = self.api_engine.encryption_service.encrypt(startup_config_data)

        return {
            "server_description": server_description,
            "server_embedding": server_embedding,
            "nb_tools": nb_tools,
            "tools": describe_response.tools.tools if describe_response.tools else [],
            "tool_embeddings": tool_embeddings,
            "encrypted_startup_config": encrypted_startup_config
        }

    async def _upsert_tools(self, server_name: str, tools: List, tool_embeddings: List):
        upsert_tasks = []
        for tool, embedding_result in zip(tools, tool_embeddings):
            upsert_tasks.append(
                self.api_engine.vector_store_service.add_tool(
                    server_name=server_name,
                    tool_name=tool.name,
                    tool_description=tool.description,
                    tool_schema=tool.inputSchema,
                    embedding=embedding_result
                )
            )

        if upsert_tasks:
            gather_results = await asyncio.gather(*upsert_tasks, return_exceptions=True)
            for result in gather_results:
                if isinstance(result, Exception):
                    logger.error(f"Error during tool upsert operation: {str(result)}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to upsert tools: {str(result)}"
                    )

    async def _upsert_server(self, server_name: str, server_description: McpServerDescription, server_embedding: List[float], nb_tools: int, encrypted_startup_config: str):
        await self.api_engine.vector_store_service.add_server(
            server_name=server_name,
            mcp_server_description=server_description,
            embedding=server_embedding,
            nb_tools=nb_tools,
            startup_config=encrypted_startup_config
        )

    def define_routes(self):

        @self.post("/mcp/servers", response_model=MCPServerResponse, status_code=status.HTTP_201_CREATED)
        async def add_mcp_server(request: AddMCPServerRequest, api_key: str = Depends(self.api_engine.auth_middleware.verify_api_key)):
            try:
                startup_config = self._build_startup_config(
                    command=request.command,
                    args=request.args,
                    env=request.env
                )

                bundle = await self._describe_and_embed_server(
                    server_name=request.server_name,
                    startup_config=startup_config,
                    timeout=request.timeout,
                    alpha=request.alpha
                )

                await self._upsert_tools(
                    server_name=request.server_name,
                    tools=bundle["tools"],
                    tool_embeddings=bundle["tool_embeddings"]
                )

                await self._upsert_server(
                    server_name=request.server_name,
                    server_description=bundle["server_description"],
                    server_embedding=bundle["server_embedding"],
                    nb_tools=bundle["nb_tools"],
                    encrypted_startup_config=bundle["encrypted_startup_config"]
                )

                server_info = MCPServerInfo(
                    server_name=request.server_name,
                    title=bundle["server_description"].title,
                    summary=bundle["server_description"].summary,
                    capabilities=bundle["server_description"].capabilities,
                    limitations=bundle["server_description"].limitations,
                    nb_tools=bundle["nb_tools"],
                    startup_config=bundle["encrypted_startup_config"]
                )

                return MCPServerResponse(
                    server=server_info,
                    message=f"MCP server '{request.server_name}' successfully analyzed and indexed"
                )

            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to process MCP server: {str(e)}"
                )

        @self.get("/mcp/servers/{server_name}", response_model=MCPServerResponse)
        async def get_mcp_server(server_name: str = Path(..., description="MCP server name"), api_key: str = Depends(self.api_engine.auth_middleware.verify_api_key)):
            try:
               server_info = await self.api_engine.vector_store_service.get_server(server_name)
               return MCPServerResponse(
                    server=MCPServerInfo(**server_info),
                    message=f"MCP server '{server_name}' retrieved successfully"
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to retrieve server: {str(e)}"
                )

        @self.get("/mcp/servers/{server_name}/tools/{tool_name}", response_model=MCPToolResponse)
        async def get_mcp_tool(
            server_name: str = Path(..., description="MCP server name"),
            tool_name: str = Path(..., description="Tool name"),
            api_key: str = Depends(self.api_engine.auth_middleware.verify_api_key)
        ):
            try:
                tool_info = await self.api_engine.vector_store_service.get_tool(server_name, tool_name)
                return MCPToolResponse(
                    tool_name=tool_info["tool_name"],
                    tool_description=tool_info["tool_description"],
                    tool_schema=tool_info["tool_schema"],
                    server_name=tool_info["server_name"],
                    message=f"Tool '{tool_name}' from server '{server_name}' retrieved successfully"
                )
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(e)
                )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to retrieve tool: {str(e)}"
                )

        @self.get("/mcp/servers/{server_name}/command", response_model=MCPCommandResponse)
        async def get_mcp_server_command(server_name: str = Path(..., description="MCP server name"), api_key: str = Depends(self.api_engine.auth_middleware.verify_api_key)):
            try:
                server_info = await self.api_engine.vector_store_service.get_server(server_name)
                encrypted_startup_config = server_info.get("startup_config", "")

                if not encrypted_startup_config:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"No startup config found for server '{server_name}'"
                    )

                try:
                    startup_config_data = self.api_engine.encryption_service.decrypt(encrypted_startup_config)
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to decrypt command: {str(e)}"
                    )

                return MCPCommandResponse(
                    server_name=server_name,
                    command=startup_config_data.get("command", ""),
                    args=startup_config_data.get("args", []),
                    env=startup_config_data.get("env", {}),
                    message=f"Command for server '{server_name}' retrieved successfully"
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to retrieve command: {str(e)}"
                )

        @self.put("/mcp/servers/{server_name}", response_model=MCPServerResponse)
        async def update_mcp_server(
            request: UpdateMCPServerRequest,
            server_name: str = Path(..., description="MCP server name"),
            api_key: str = Depends(self.api_engine.auth_middleware.verify_api_key)
        ):
            try:
                # Check if server exists and get current info
                current_server_info = await self.api_engine.vector_store_service.get_server(server_name)

                # Get current encrypted startup config and decrypt it
                current_encrypted_config = current_server_info.get("startup_config", "")
                if not current_encrypted_config:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Server '{server_name}' has no startup configuration to update"
                    )

                current_config = self.api_engine.encryption_service.decrypt(current_encrypted_config)

                # Build new config using provided values or keeping current ones
                new_command = request.command if request.command is not None else current_config.get("command")
                new_args = request.args if request.args is not None else current_config.get("args", [])
                new_env = request.env if request.env is not None else current_config.get("env", {})

                # Check if startup config actually changed
                startup_changed = (
                    new_command != current_config.get("command") or
                    new_args != current_config.get("args", []) or
                    new_env != current_config.get("env", {})
                )

                if startup_changed:
                    # Re-analyze server with new startup config
                    startup_config = self._build_startup_config(
                        command=new_command,
                        args=new_args,
                        env=new_env
                    )

                    # Use default values for timeout and alpha if not provided in original request
                    bundle = await self._describe_and_embed_server(
                        server_name=server_name,
                        startup_config=startup_config,
                        timeout=50,  # Default timeout
                        alpha=0.1    # Default alpha
                    )

                    # Delete existing tools and server, then re-upsert with new data
                    await self.api_engine.vector_store_service.delete_server(server_name)

                    await self._upsert_tools(
                        server_name=server_name,
                        tools=bundle["tools"],
                        tool_embeddings=bundle["tool_embeddings"]
                    )

                    await self._upsert_server(
                        server_name=server_name,
                        server_description=bundle["server_description"],
                        server_embedding=bundle["server_embedding"],
                        nb_tools=bundle["nb_tools"],
                        encrypted_startup_config=bundle["encrypted_startup_config"]
                    )

                    server_info = MCPServerInfo(
                        server_name=server_name,
                        title=bundle["server_description"].title,
                        summary=bundle["server_description"].summary,
                        capabilities=bundle["server_description"].capabilities,
                        limitations=bundle["server_description"].limitations,
                        nb_tools=bundle["nb_tools"],
                        startup_config=bundle["encrypted_startup_config"]
                    )

                    return MCPServerResponse(
                        server=server_info,
                        message=f"MCP server '{server_name}' successfully updated and re-analyzed"
                    )
                else:
                    # No changes to startup config, return current info
                    server_info = MCPServerInfo(**current_server_info)
                    return MCPServerResponse(
                        server=server_info,
                        message=f"MCP server '{server_name}' - no changes detected"
                    )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to update server '{server_name}': {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to update server: {str(e)}"
                )

        @self.delete("/mcp/servers/{server_name}")
        async def delete_mcp_server(server_name: str = Path(..., description="MCP server name"), api_key: str = Depends(self.api_engine.auth_middleware.verify_api_key)):
            try:
                server_info = await self.api_engine.vector_store_service.delete_server(server_name)
                return MCPServerResponse(
                    server=MCPServerInfo(**server_info),
                    message=f"MCP server '{server_name}' successfully deleted"
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to delete server: {str(e)}"
                )

        @self.get("/mcp/servers", response_model=MCPServerListResponse)
        async def list_mcp_servers(
            limit: int = Query(50, ge=1, le=1000, description="Maximum results to return"),
            offset: Optional[str] = Query(None, description="Pagination offset"),
            api_key: str = Depends(self.api_engine.auth_middleware.verify_api_key)
        ):
            try:
                servers, offset = await self.api_engine.vector_store_service.list_servers(
                    limit=limit,
                    offset=offset
                )
                return MCPServerListResponse(
                    servers=[MCPServerInfo(**server) for server in servers],
                    limit=limit,
                    offset=offset
                )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to list servers: {str(e)}"
                )

        @self.get("/mcp/servers/{server_name}/tools", response_model=MCPToolListResponse)
        async def list_mcp_tools(
            server_name: str = Path(..., description="MCP server name"),
            limit: int = Query(50, ge=1, le=1000, description="Maximum results to return"),
            offset: Optional[str] = Query(None, description="Pagination offset"),
            api_key: str = Depends(self.api_engine.auth_middleware.verify_api_key)
        ):
            try:
                tools, next_offset = await self.api_engine.vector_store_service.list_tools(
                    server_name=server_name,
                    limit=limit,
                    offset=offset
                )
                return MCPToolListResponse(
                    tools=[MCPToolInfo(**tool) for tool in tools],
                    server_name=server_name,
                    limit=limit,
                    offset=next_offset
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to list tools: {str(e)}"
                )


        @self.get("/mcp/statistics", response_model=MCPStatisticsResponse)
        async def get_statistics(api_key: str = Depends(self.api_engine.auth_middleware.verify_api_key)):
            try:
                # Get both counts concurrently for better performance
                nb_servers_task = self.api_engine.vector_store_service.nb_servers()
                nb_tools_task = self.api_engine.vector_store_service.nb_tools()

                nb_servers, nb_tools = await asyncio.gather(nb_servers_task, nb_tools_task)

                return MCPStatisticsResponse(
                    total_servers=nb_servers,
                    total_tools=nb_tools
                )
            except Exception as e:
                logger.error(f"Failed to get statistics: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get statistics: {str(e)}"
                )
        
        

        @self.post("/mcp/servers/search", response_model=SearchServersResponse)
        async def search_servers(request: SearchServersRequest, api_key: str = Depends(self.api_engine.auth_middleware.verify_api_key)) -> SearchServersResponse:
            """
            Search for MCP servers using semantic similarity.
            """
            start_time = time.time()

            try:
                # Create embedding for the search query
                query_embedding = await self.api_engine.embedding_service.create_embedding([request.query], input_type="search_query")

                # Search for servers in the vector store
                search_results = await self.api_engine.vector_store_service.search(
                    embedding=query_embedding[0],
                    top_k=request.limit,
                    payload_types=["server"]
                )

                servers = []
                for result in search_results:
                    if result["score"] < (request.min_score or 0.0):
                        continue
                    payload = result["payload"]
                    server_result = SearchResultServer(
                        server_id=str(result["id"]),
                        server_name=payload.get("server_name", ""),
                        title=payload.get("title", ""),
                        summary=payload.get("summary", ""),
                        capabilities=payload.get("capabilities", []),
                        limitations=payload.get("limitations", []),
                        nb_tools=payload.get("nb_tools", 0),
                        startup_config=payload.get("startup_config", ""),
                        score=float(result["score"])
                    )
                    servers.append(server_result)

                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                return SearchServersResponse(
                    servers=servers,
                    total_results=len(servers),
                    query=request.query,
                    message=f"Found {len(servers)} matching servers",
                    processing_time_ms=processing_time
                )

            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Server search failed: {str(e)}"
                )

        @self.post("/mcp/tools/search", response_model=SearchToolsResponse)
        async def search_tools(request: SearchToolsRequest, api_key: str = Depends(self.api_engine.auth_middleware.verify_api_key)) -> SearchToolsResponse:
            """
            Search for MCP tools using semantic similarity.
            """
            start_time = time.time()

            try:
                # enhance query 
                # enhanced_query = await self.api_engine.mcp_descriptor_service.enhance_query_with_llm(request.query)
                # Create embedding for the search query
                query_embedding = await self.api_engine.embedding_service.create_embedding([request.query], input_type="search_query")
                
                # Search for tools in the vector store
                search_results = await self.api_engine.vector_store_service.search(
                    embedding=query_embedding[0],
                    top_k=request.limit,
                    server_names=request.server_names,
                    payload_types=["tool"],
                )

                tools = []
                for result in search_results:
                    if result["score"] < (request.min_score or 0.0):
                        continue
                    payload = result["payload"]
                    tools.append(SearchResultTool(
                        tool_id=str(result["id"]),
                        tool_name=payload.get("tool_name", ""),
                        tool_description=payload.get("tool_description", ""),
                        tool_schema=payload.get("tool_schema", {}),
                        server_name=payload.get("server_name", ""),
                        score=float(result["score"])
                    ))

                processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                return SearchToolsResponse(
                    tools=tools,
                    total_results=len(tools),
                    query=request.query,
                    message=f"Found {len(tools)} matching tools",
                    processing_time_ms=processing_time
                )

            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Tool search failed: {str(e)}"
                )


    