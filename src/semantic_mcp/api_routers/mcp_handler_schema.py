from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID

# Request Schemas
class AddMCPServerRequest(BaseModel):
    server_name: str = Field(..., description="Unique name identifier for the server")
    command: str = Field(..., description="Main command to start the MCP server (e.g., 'python')")
    args: List[str] = Field(default_factory=list, description="Arguments for the command (e.g., ['-m', 'my_mcp_server'])")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables for the server")
    timeout: Optional[int] = Field(50, description="Timeout in seconds for server description")
    alpha: Optional[float] = Field(0.1, description="Weighting factor for embeddings")

class MCPServerConfig(BaseModel):
    command: str = Field(..., description="Command to start the MCP server")
    args: List[str] = Field(default_factory=list, description="Arguments for the command")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables for the server")
    timeout: Optional[int] = Field(50, description="Timeout in seconds for server description")
    alpha: Optional[float] = Field(0.1, description="Weighting factor for embeddings")

class ClaudeCodeMCPConfig(BaseModel):
    mcpServers: Dict[str, MCPServerConfig] = Field(..., description="Dictionary of server configurations keyed by server name")

class BulkAddMCPServersRequest(BaseModel):
    servers: List[AddMCPServerRequest] = Field(..., description="List of MCP server configurations to add")
    
class UpdateMCPServerRequest(BaseModel):
    command: Optional[str] = Field(None, description="Updated main command to start the server")
    args: Optional[List[str]] = Field(None, description="Updated arguments for the command")
    env: Optional[Dict[str, str]] = Field(None, description="Updated environment variables")
    

class MCPServerInfo(BaseModel):
    server_name: str = Field(..., description="Server name")
    title: str = Field(..., description="AI-generated server title")
    summary: str = Field(..., description="AI-generated server summary")
    capabilities: List[str] = Field(..., description="List of server capabilities")
    limitations: List[str] = Field(..., description="List of server limitations")
    nb_tools: int = Field(..., description="Number of tools available")
    startup_config: str = Field(..., description="Encrypted startup configuration")
    
    
class MCPServerListResponse(BaseModel):
    servers: List[MCPServerInfo] = Field(..., description="List of MCP servers")
    limit: int = Field(..., description="Applied limit")
    offset: Optional[str] = Field(None, description="Applied offset")

class MCPToolInfo(BaseModel):
    tool_name: str = Field(..., description="Tool name")
    tool_description: str = Field(..., description="Tool description")
    tool_schema: Dict[str, Any] = Field(..., description="Tool JSON schema")
    server_name: str = Field(..., description="Parent server name")

class MCPToolListResponse(BaseModel):
    tools: List[MCPToolInfo] = Field(..., description="List of MCP tools")
    server_name: str = Field(..., description="Server name")
    limit: int = Field(..., description="Applied limit")
    offset: Optional[str] = Field(None, description="Applied offset")

class MCPServerResponse(BaseModel):
    server: MCPServerInfo = Field(..., description="MCP server information")
    message: str = Field(..., description="Operation result message")

class MCPStatisticsResponse(BaseModel):
    total_servers: int = Field(..., description="Total number of MCP servers")
    total_tools: int = Field(..., description="Total number of MCP tools")

# Search Request Schemas
class SearchServersRequest(BaseModel):
    query: str = Field(..., description="Natural language search query", min_length=1, max_length=1000)
    limit: int = Field(10, description="Maximum number of results to return", ge=1, le=100)
    min_score: Optional[float] = Field(0.7, description="Minimum similarity score (0.0-1.0)", ge=0.0, le=1.0)
    
class SearchToolsRequest(BaseModel):
    query: str = Field(..., description="Natural language search query", min_length=1, max_length=1000)
    limit: int = Field(10, description="Maximum number of results to return", ge=1, le=100)
    min_score: Optional[float] = Field(0.7, description="Minimum similarity score (0.0-1.0)", ge=0.0, le=1.0)
    server_names: Optional[List[str]] = Field(None, description="Filter tools by server name")

# Search Response Schemas
class SearchResultServer(BaseModel):
    server_id: str = Field(..., description="Server UUID")
    server_name: str = Field(..., description="Server name")
    title: str = Field(..., description="AI-generated server title")
    summary: str = Field(..., description="AI-generated server summary")
    capabilities: List[str] = Field(..., description="List of server capabilities")
    limitations: List[str] = Field(..., description="List of server limitations")
    nb_tools: int = Field(..., description="Number of tools available")
    startup_config: str = Field(..., description="Encrypted startup configuration")
    score: float = Field(..., description="Similarity score (0.0-1.0)")
    
class SearchResultTool(BaseModel):
    tool_id: str = Field(..., description="Tool UUID")
    tool_name: str = Field(..., description="Tool name")
    tool_description: str = Field(..., description="Tool description")
    tool_schema: Dict[str, Any] = Field(..., description="Tool JSON schema")
    server_name: str = Field(..., description="Parent server name")
    score: float = Field(..., description="Similarity score (0.0-1.0)")

class SearchServersResponse(BaseModel):
    servers: List[SearchResultServer] = Field(..., description="Matching servers")
    total_results: int = Field(..., description="Total number of results found")
    query: str = Field(..., description="Original search query")
    message: str = Field(..., description="Operation result message")
    processing_time_ms: float = Field(..., description="Query processing time in milliseconds")

class SearchToolsResponse(BaseModel):
    tools: List[SearchResultTool] = Field(..., description="Matching tools")
    total_results: int = Field(..., description="Total number of results found")
    query: str = Field(..., description="Original search query")
    message: str = Field(..., description="Operation result message")
    processing_time_ms: float = Field(..., description="Query processing time in milliseconds")

class MCPToolResponse(BaseModel):
    tool_name: str = Field(..., description="Tool name")
    tool_description: str = Field(..., description="Tool description")
    tool_schema: Dict[str, Any] = Field(..., description="Tool JSON schema")
    server_name: str = Field(..., description="Parent server name")
    message: str = Field(..., description="Operation result message")

class MCPCommandResponse(BaseModel):
    server_name: str = Field(..., description="Server name")
    command: str = Field(..., description="Main command to start the MCP server")
    args: List[str] = Field(..., description="Command arguments")
    env: Dict[str, str] = Field(..., description="Environment variables")
    message: str = Field(..., description="Operation result message")

class BulkAddResult(BaseModel):
    server_name: str = Field(..., description="Server name")
    success: bool = Field(..., description="Whether the server was added successfully")
    message: str = Field(..., description="Success message or error description")
    server_info: Optional[MCPServerInfo] = Field(None, description="Server information if successfully added")

class BulkAddMCPServersResponse(BaseModel):
    results: List[BulkAddResult] = Field(..., description="Results for each server")
    successful_count: int = Field(..., description="Number of servers successfully added")
    failed_count: int = Field(..., description="Number of servers that failed to add")
    total_count: int = Field(..., description="Total number of servers processed")
    message: str = Field(..., description="Overall operation result message")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
