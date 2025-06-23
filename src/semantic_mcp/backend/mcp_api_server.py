"""
MCP Engine module for exposing LLM Reflexion functionality as an MCP server using FastMCP.

This module provides the MCPEngine class which wraps the MCPRouter functionality
and exposes it as a Model Context Protocol server using FastMCP. It allows external MCP clients
to access the same service discovery, tool management, and execution capabilities
that the LLMEngine uses internally.

The server exposes a single tool 'use_services' that provides access to all
the dynamic tool ecosystem functionality through a unified interface.

Classes:
    MCPEngine: Main class for running the MCP server with tool access

Dependencies:
    - asyncio: For asynchronous operations
    - mcp.server.fastmcp: For FastMCP server implementation
    - json: For data serialization
"""

import asyncio
import json
from functools import partial
from typing import Any, Dict, List, Optional, AsyncGenerator, Coroutine
from uuid import uuid4
from fastapi_mcp import FastApiMCP
from uvicorn import run 
from fastapi import FastAPI

from contextlib import asynccontextmanager

from .mcp_router import MCPRouter
from .mcp_types import McpServers
from ..log import logger
from ..settings import CredentialsConfig, VectorStoreConfig

from enum import Enum
from pydantic import BaseModel, Field 

class Action(str, Enum):
    GET_SYSTEM_INSTRUCTIONS = "get_system_instructions"
    STATISTICS = "statistics"
    SEARCH_TOOLS = "search_tools"
    EXECUTE_TOOL = "execute_tool"
    SCROLL_OVER_SERVICES = "scroll_over_services"
    READ_SERVICE_INFORMATION = "read_service_information"
    SEARCH_SERVICES = "search_services"
    READ_TOOL_INFORMATION = "read_tool_information"
    GET_BACKGROUND_TASK_STATUS = "get_background_task_status"

class ExecutionMode(str, Enum):
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"

class Payload(BaseModel):
    action: Action
    query: Optional[str] = None
    service_name: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    offset: Optional[str] = None
    services: Optional[List[str]] = None
    top_n: Optional[int] = None
    task_id: Optional[str] = None
    execution_mode: Optional[ExecutionMode] = ExecutionMode.SYNCHRONOUS

class MCPEngine:
    """
    MCP Server that exposes LLM Reflexion's dynamic tool ecosystem functionality using FastMCP.
    
    This class wraps the MCPRouter and exposes its capabilities through a single
    MCP tool called 'use_services'. External MCP clients can use this to:
    - Discover available services and tools
    - Register and unregister tools dynamically  
    - Execute registered tools through higher_order_apply
    - Get AI assistance for service-specific tasks
    
    The server follows the same higher-order execution pattern as the LLMEngine
    but exposes it through the MCP protocol instead of direct LLM interaction.
    """
    
    def __init__(self, mcp_servers: McpServers, vector_store_config: VectorStoreConfig, credentials: CredentialsConfig):
        self.mcp_servers = mcp_servers
        self.vector_store_config = vector_store_config
        self.credentials = credentials
        self.app = FastAPI(
            title="MCP Server",
            description="MCP Server that exposes LLM Reflexion's dynamic tool ecosystem functionality",
            version="0.1.0",
            lifespan=self.lifespan
        )
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        mcp_router = MCPRouter(
            mcp_servers=self.mcp_servers,
            vector_store_config=self.vector_store_config,
            credentials=self.credentials
        )
        async with mcp_router as mcp_router:
            await mcp_router.start_mcp_servers()
            operations = await self.register_services(mcp_router)
            mcp = FastApiMCP(
                app,
                name="mcp_galaxy",
                description="""
                A comprehensive and dynamic interface for discovering, exploring, and interacting with available services and their associated tools.
                """,
                describe_all_responses=True,
                describe_full_response_schema=True,
                include_operations=operations
            )
            mcp.mount()
            yield
        
    async def register_services(self, mcp_router: MCPRouter) -> None:
        description = """
        🚀 **Comprehensive MCP Services Gateway**

        This unified interface provides intelligent access to a dynamic ecosystem of MCP services and tools through semantic search, discovery, and execution capabilities. The system leverages vector embeddings and AI-powered recommendations to help you find and utilize the right tools for any task.

        ## 🔍 **DISCOVERY & EXPLORATION ACTIONS**

        ### **📄 get_system_instructions**
        Get the system instructions for the agent.
            - **Purpose**: Get the system instructions for the agent
            - **Required**: None
            - **Returns**: System instructions
            - **Use Case**: Very important to understand the agent's behavior and capabilities, it is the first thing to do when starting a new conversation.

        ### **📊 statistics**
        Get statistics about the number of services and tools in the ecosystem.
            - **Purpose**: Get a quick overview of the number of services and tools available
            - **Required**: None
            - **Returns**: Statistics about the number of services and tools
            - **Use Case**: Initial exploration, service catalog browsing

        ### **📋 scroll_over_services**
        Paginate through all available services in the ecosystem.
            - **Purpose**: Browse the complete catalog of services systematically
            - **Required**: `limit` (int) - Number of services per page
            - **Optional**: `offset` (str) - Pagination cursor (null for first page)
            - **Returns**: List of services with pagination metadata
            - **Use Case**: Initial exploration, service catalog browsing

        ### **🔍 search_services** 
        Apply semantic search across all services using natural language queries.
            - **Purpose**: Find services by capability, domain, or functionality
            - **Required**: 
            - `query` (str) - Natural language description of needed capability
            - `top_n` (int) - Number of most relevant services to return
            - **Returns**: Ranked list of services matching the query
            - **Use Case**: "Find services for voice generation", "What can handle file operations?"

        ### **📖 read_service_information**
        Get comprehensive details about a specific service.
            - **Purpose**: Understand a service's capabilities, tools, and purpose
            - **Required**: `service_name` (str) - Exact name of the service
            - **Returns**: Detailed service metadata (title, summary, capabilities, available tools)
            - **Use Case**: Deep dive into a specific service before using its tools

        ## 🛠️ **TOOL MANAGEMENT ACTIONS**

        ### **🔍 search_tools**
        Apply semantic search across tools with optional service filtering.
            - **Purpose**: Find specific tools by functionality across services
            - **Required**: 
            - `query` (str) - Natural language description of needed functionality
            - `top_n` (int) - Number of most relevant tools to return
            - **Optional**: `services` (List[str]) - Filter search to specific services only
            - **Returns**: Ranked list of tools with full schemas
            - **Use Case**: "Find tools for text-to-speech", "What can create agents?"

        ### **📖 read_tool_information**
        Get complete documentation for a specific tool including execution schema.
            - **Purpose**: Understand tool parameters, types, and requirements before execution
            - **Required**: 
            - `service_name` (str) - Parent service name
            - `tool_name` (str) - Exact tool name
            - **Returns**: Full tool specification (description, input schema, parameter details)
            - **Use Case**: Validate parameters before execution, understand tool capabilities

        ## ⚡ **EXECUTION ACTION**

        ### **🚀 execute_tool**
        Execute a specific tool with provided arguments.
            - **Purpose**: Run tools from any service with type-safe parameter passing
            - **Required**: 
            - `service_name` (str) - Parent service name
            - `tool_name` (str) - Exact tool name from describe_tool
            - `tool_arguments` (Dict[str, Any]) - Parameters matching the tool's input schema
            - `execution_mode` (str) - Execution mode: "synchronous" or "asynchronous"
            - **Returns**: Tool execution results (success/error with details)
            - **Use Case**: Actual tool execution after discovery and validation. Use "asynchronous" mode to execute tools that take a long time to complete.
        
        ### **🔍 get_background_task_status**
        Get the status of a background task.
            - **Purpose**: Get the status of a background task
            - **Required**: 
            - `task_id` (str) - Task ID
            - **Returns**: Task status
            - **Use Case**: Get the status of a background task
        """

        self.app.post(
            path=f"/mcp/interact_with_services",
            response_model=str,
            status_code=200,
            description=description,
            operation_id=f"interact_with_services"
        )(self.high_level_use_services(mcp_router=mcp_router))
        
    def high_level_use_services(self, mcp_router: MCPRouter) -> Coroutine:
        async def use_services(payload: Payload) -> str:
            try:
                # Route the action to the appropriate MCPRouter method
                tool_call = {
                    "id": str(uuid4()),
                    "function": {
                        "name": f'interact_with_services',
                        "arguments": json.dumps({
                            "action": payload.action,
                            "query": payload.query,
                            "top_n": payload.top_n,
                            "tool_name": payload.tool_name,
                            "tool_arguments": payload.tool_arguments,
                            "service_name": payload.service_name,
                            "services": payload.services,
                            "limit": payload.limit,
                            "offset": payload.offset,
                            "task_id": payload.task_id,
                            "execution_mode": payload.execution_mode
                        })
                    }   
                }
                tool_call_result_content = await mcp_router.main_runner(tool_call)
                return tool_call_result_content
                
            except Exception as e:
                logger.error(f"Error in use_services: {e}")
                return f"Error: {str(e)}"
        
        return use_services
        

def start_mcp_server(mcp_servers: McpServers, vector_store_config: VectorStoreConfig, credentials: CredentialsConfig) -> None:
    """
    Start the MCP server with the given configuration.
    
    Args:
        mcp_servers: MCP servers configuration
        vector_store_config: Vector store configuration
        credentials: API credentials
    """
    engine = MCPEngine(
        mcp_servers=mcp_servers,
        vector_store_config=vector_store_config,
        credentials=credentials
    )
    run(engine.app, host="0.0.0.0", port=8200)
    