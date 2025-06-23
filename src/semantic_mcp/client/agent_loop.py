import json 
from anthropic import AsyncAnthropic

from anthropic.types import RawMessageStreamEvent

from .types import Role, ChatMessage, StopReason
from .definitions import SystemPromptDefinitions
from typing import List, Tuple, Dict, Any, Optional, Iterable, AsyncIterable, Self

from operator import itemgetter, attrgetter

from ..log import logger 
from contextlib import suppress, AsyncExitStack


from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

from ..backend.mcp_types import McpConfig

class AgentLoop:
    def __init__(self, anthropic_api_key:str, mcp_config:McpConfig):
        self.anthropic_api_key = anthropic_api_key
        self.mcp_config = mcp_config

    async def __aenter__(self) -> Self:
        server_parameters = StdioServerParameters(
            command=self.mcp_config.command,
            args=self.mcp_config.args,
            env=self.mcp_config.env
        )
        self.resources_manager = AsyncExitStack()
        stdio_transport = await self.resources_manager.enter_async_context(stdio_client(server=server_parameters))
        reader, writer = stdio_transport
        self.mcp_session = await self.resources_manager.enter_async_context(ClientSession(read_stream=reader, write_stream=writer))
        await self.mcp_session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.resources_manager.aclose()

    async def handle_tool(self, tool_name:str, tool_args:str, tool_use_id:str) -> ChatMessage:
        try:
            tool_result = await self.mcp_session.call_tool(tool_name, json.loads(tool_args))
            tool_result_content = '\n'.join([ res.model_dump_json(indent=2) for res in tool_result.content ])
        
            return ChatMessage(
                role=Role.USER, 
                content=[
                    {
                        'type': 'tool_result',
                        'tool_use_id': tool_use_id,
                        'content': tool_result_content
                    }
                ]
            )
        except Exception as e:
            logger.error(f'error: {e}')
            return ChatMessage(
                role=Role.USER,
                content=[
                    {
                        'type': 'tool_result',
                        'tool_use_id': tool_use_id,
                        'is_error': True,
                        'content': str(e)
                    }
                ]
            )

    async def consume_stream(self, stream:AsyncIterable[RawMessageStreamEvent]) -> Tuple[StopReason, List[ChatMessage]]:
        conversation_history:List[ChatMessage] = []
        stop_reason = StopReason.END_TURN
        content:List[Dict] = []
        current_block_type = None
        text, signature, thinking, tool_name, tool_args, tool_use_id = None, None, None, None, None, None
        async for event in stream:
            match event.type:
                case 'message_start':
                    print('')
                case 'message_delta':
                    stop_reason = event.delta.stop_reason
                    conversation_history.append(ChatMessage(role=Role.ASSISTANT, content=content))
                    if stop_reason != StopReason.TOOL_USE:
                        continue
                    print(tool_name)
                    print(tool_args)
                    tool_result = await self.handle_tool(tool_name, tool_args, tool_use_id)
                    conversation_history.append(tool_result)
                case 'content_block_start':
                    current_block_type = event.content_block.type
                    match event.content_block.type:
                        case 'text': 
                            print('<response>')
                            text = event.content_block.text
                        case 'thinking': 
                            print('<thinking>')
                            thinking = event.content_block.thinking
                            signature = event.content_block.signature
                        case 'tool_use':
                            print('<tool_use>')
                            tool_name = event.content_block.name
                            tool_args = ''
                            tool_use_id = event.content_block.id
                case 'content_block_delta':
                    match event.delta.type:
                        case 'text_delta':
                            print(event.delta.text, end='', flush=True)
                            text = text + event.delta.text 
                        case 'thinking_delta':
                            print(event.delta.thinking, end='', flush=True)
                            thinking = thinking + event.delta.thinking
                        case 'signature_delta':
                            signature = signature + event.delta.signature
                        case 'input_json_delta':
                            tool_args = tool_args + event.delta.partial_json 
                case 'content_block_stop':
                    print('')
                    match current_block_type:
                        case 'text':
                            print('</response>')
                            content.append({'type': 'text', 'text': text})  
                        case 'thinking':
                            print('</thinking>')
                            content.append({'type': 'thinking', 'thinking': thinking, 'signature': signature})
                        case 'tool_use':
                            print('</tool_use>')
                            content.append({'type': 'tool_use', 'name': tool_name, 'input': json.loads(tool_args), 'id': tool_use_id})
            # end match event.type 
        # end for event in stream
        return stop_reason, conversation_history
    
    async def handle_conversation(self, conversation_history:List[ChatMessage], system:str, model:str='claude-sonnet-4-20250514', max_tokens:int=8192, thinking={'type': 'enabled', "budget_tokens": 1024}, tools=[]) -> Optional[AsyncIterable[RawMessageStreamEvent]]:
        try:
            anthropic_client = AsyncAnthropic(api_key=self.anthropic_api_key)
            completion_res:AsyncIterable[RawMessageStreamEvent] = await anthropic_client.messages.create(
                model=model, 
                messages=conversation_history,
                system=system,
                max_tokens=max_tokens,
                stream=True,
                thinking=thinking, 
                tools=tools
            )
            return completion_res
        except Exception as e:
            logger.error(f'error: {e}')
            
    async def run(self) -> None:
        stop_reason = StopReason.END_TURN
        conversation_history:List[ChatMessage] = []
        
        list_tools_response = await self.mcp_session.list_tools()
        tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in list_tools_response.tools]

        while True:
            try:
                if stop_reason != StopReason.TOOL_USE:
                    query = input('query: ')
                    user_message = ChatMessage(role=Role.USER, content=query)
                    conversation_history.append(user_message)
                
                completion_res = await self.handle_conversation(
                    conversation_history, 
                    SystemPromptDefinitions.MAIN_AGENT_LOOP, 
                    tools=tools
                )
                if completion_res is None:
                    logger.error('error: completion_res is None')
                    continue
                
                stop_reason_delta, conversation_history_delta = await self.consume_stream(completion_res)
                stop_reason = stop_reason_delta
                conversation_history.extend(conversation_history_delta)
            except KeyboardInterrupt:
                logger.info('exiting...')
                break
            except Exception as e:
                logger.error(f'error: {e}')
                break  
