from pydantic import BaseModel
from typing import List, Dict, Optional

class McpConfig(BaseModel):
    command:str
    args:List[str]
    env:Optional[Dict[str, str]]=None

class McpSettings(BaseModel):
    name:str
    mcp_config:McpConfig
    allowed_tools:List[str]=[]
    ignore:bool=False
    startup_timeout:float=30
    index:bool=False

class McpServers(BaseModel):
    mcpServers:List[McpSettings]