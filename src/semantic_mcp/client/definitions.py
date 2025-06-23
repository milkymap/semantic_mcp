from typing import List, Tuple, Dict 
from enum import Enum 

class SystemPromptDefinitions(str, Enum):
    MAIN_AGENT_LOOP:str = '''
    You are a helpful assistant with access to semantic_mcp, an advanced mcp server that expose a large ecosystem of tools.
    '''
   