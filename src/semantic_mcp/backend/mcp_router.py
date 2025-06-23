import multiprocessing as mp
import asyncio 
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from datetime import datetime

import json, pickle, yaml 
import numpy as np 
from numpy.typing import NDArray

from os import path 

import zmq 
import zmq.asyncio as aiozmq 

from uuid import uuid5, NAMESPACE_DNS, uuid4
from hashlib import sha256

from operator import itemgetter, attrgetter

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Self, AsyncGenerator, Any, Tuple, Coroutine, Set
from enum import Enum

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, RecommendQuery, RecommendInput, Filter, RecommendStrategy, FieldCondition, MatchAny

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

from .mcp_types import McpServers, McpSettings, McpConfig
from .mq_embedding import MQEmbedding

from semantic_mcp.log import logger
from semantic_mcp.settings import VectorStoreConfig, CredentialsConfig

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ParsedChatCompletion

from .language_model_types import ChatMessage, StopReason, OpenaiModel, Role
from .system_instructions import SystemInstructions

import base64

class MCPRouter:
    def __init__(self, mcp_servers:McpServers, vector_store_config:VectorStoreConfig, credentials:CredentialsConfig):
       self.mcp_servers = mcp_servers
       self.vector_store_config = vector_store_config
       self.credentials = credentials

       
    async def __aenter__(self) -> Self:
        self.qdrant_client = AsyncQdrantClient(
            url=self.vector_store_config.qdrant_url,
            api_key=self.vector_store_config.qdrant_api_key
        )
        self.openai_client = AsyncOpenAI(api_key=self.credentials.openai_api_key)
        if not await self.qdrant_client.collection_exists(collection_name=self.vector_store_config.qdrant_services_collection_name):
            await self.qdrant_client.create_collection(
                collection_name=self.vector_store_config.qdrant_services_collection_name,
                vectors_config=VectorParams(
                    size=self.vector_store_config.embedding_dimensions,
                    distance=Distance.COSINE
                )
            )
        if not await self.qdrant_client.collection_exists(collection_name=self.vector_store_config.qdrant_tools_collection_name):
            await self.qdrant_client.create_collection(
                collection_name=self.vector_store_config.qdrant_tools_collection_name,
                vectors_config=VectorParams(
                    size=self.vector_store_config.embedding_dimensions,
                    distance=Distance.COSINE
                )
            )
        self.ctx = aiozmq.Context()
        self.resources_manager = AsyncExitStack()
        self.mcp_session_mutex = asyncio.Lock()
        self.service_name2mcp_session:Dict[str, ClientSession] = {}
        self.background_tasks:Dict[str, asyncio.Task] = {}
        self.background_tasks_mutex = asyncio.Lock()
        
        return self 

    
    async def __aexit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(f"MCPRouter received error: {exc_value}")
            logger.exception(traceback)
        
        self.ctx.term()
        async with self.background_tasks_mutex:
            for task_id, task in self.background_tasks.items():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await self.resources_manager.aclose()
    
    @asynccontextmanager
    async def _build_socket(self, ctx:aiozmq.Context, socket_type:str, method:str, addr:str) -> AsyncGenerator[aiozmq.Socket, None]:
        socket = ctx.socket(socket_type=socket_type)
        try:
            attrgetter(method)(socket)(addr=addr)
            yield socket 
        except Exception as e:
            logger.error(e)
        finally:
            socket.close(linger=0)
    
    async def describe_tool(self, tool_data:Dict[str, Any]) -> List[str]:
        stringified_tool_data = yaml.dump(tool_data, sort_keys=False)
        class ToolDescription(BaseModel):
            uttirances:List[str] = Field(description="List of 10/15 uttirances that describe the tool")

        response:ParsedChatCompletion = await self.openai_client.beta.chat.completions.parse(
            model=OpenaiModel.GPT_41_MINI,
            messages=[
                ChatMessage(
                    role=Role.SYSTEM,
                    content=SystemInstructions.TOOL_DESCRIPTION
                ),
                ChatMessage(role=Role.USER, content=stringified_tool_data)
            ],
            max_tokens=1024,
            response_format=ToolDescription
        )

        parsed_data:ToolDescription = response.choices[0].message.parsed
        uttirances = parsed_data.uttirances
        logger.info(f"Tool {tool_data['name']} described with {len(uttirances)} uttirances")
        return uttirances

    async def describe_service(self, tools_data:List[Dict[str, Any]]) -> Dict[str, Any]:
        stringified_tools_data = []
        for tool_data in tools_data:
            stringified_tools_data.append(yaml.dump(tool_data, sort_keys=False))
        stringified_tools_data = "\n###\n".join(stringified_tools_data)
        
        class ServiceDescription(BaseModel):
            title: str = Field(description="Title of the service")
            summary: str = Field(description="Summary of what the service does")
            capabilities: str = Field(description="List of key capabilities of the service")
            
        response:ParsedChatCompletion = await self.openai_client.beta.chat.completions.parse(
            model=OpenaiModel.GPT_41_MINI,
            messages=[
                ChatMessage(
                    role=Role.SYSTEM,
                    content=SystemInstructions.SERVICE_DESCRIPTION
                ),
                ChatMessage(role=Role.USER, content=stringified_tools_data)
            ],
            max_tokens=1024,
            response_format=ServiceDescription
        )

        service_description:ServiceDescription = response.choices[0].message.parsed
        return service_description.model_dump()
            
    async def index_mcp_server(self, mcp_settings:McpSettings):
        if not mcp_settings.index:
            return 

        server_parameters = StdioServerParameters(
            command=mcp_settings.mcp_config.command,
            args=mcp_settings.mcp_config.args,
            env=mcp_settings.mcp_config.env
        )

        async with stdio_client(server=server_parameters) as stdio_transport:
            reader, writer = stdio_transport
            async with ClientSession(read_stream=reader, write_stream=writer) as mcp_session:   
                async with asyncio.timeout(mcp_settings.startup_timeout):
                    await mcp_session.initialize()
                
                retrieved_tools = await mcp_session.list_tools()
                tool_data_list = []
                keep_all_tools =  len(mcp_settings.allowed_tools) == 1 and mcp_settings.allowed_tools[0] == "*"
                for tool in retrieved_tools.tools:
                    if tool.name in mcp_settings.allowed_tools or keep_all_tools:
                        tool_data = {
                            "name":tool.name,
                            "description":tool.description,
                            "input_schema":tool.inputSchema
                        }
                        tool_data_list.append(tool_data)
                        print(json.dumps(tool_data, indent=2))
                        logger.info(f"Tool {tool.name} added to the list of tools")
                
                if len(tool_data_list) == 0:
                    logger.warning(f"No tools found for {mcp_settings.name}")
                    return 
                
                coroutines = [self.describe_tool(tool_data) for tool_data in tool_data_list]
                tools_utterances = await asyncio.gather(*coroutines, return_exceptions=True)
                service_description = await self.describe_service(tools_utterances)
                logger.info(f"Service {mcp_settings.name} description done")
                
        async with self._build_socket(ctx=self.ctx, socket_type=zmq.DEALER, method="connect", addr=MQEmbedding.OUTER_ROUTER_ADDR) as dealer:
            stringified_service_description = yaml.dump(service_description, sort_keys=False)
            await dealer.send_multipart([b""], flags=zmq.SNDMORE)
            await dealer.send_pyobj([stringified_service_description])
            _, encoded_service_description_embedding = await dealer.recv_multipart()
            service_description_embedding = pickle.loads(encoded_service_description_embedding)
            service_description_embedding = service_description_embedding[0]

            for tool_utterance, tool_data in zip(tools_utterances, tool_data_list):
                await dealer.send_multipart([b""], flags=zmq.SNDMORE)
                await dealer.send_pyobj(tool_utterance)
                _, encoded_embeddings = await dealer.recv_multipart()
                embeddings = pickle.loads(encoded_embeddings)
                tool_embedding = np.mean(embeddings, axis=0)
                tool_embedding = 0.8 * tool_embedding + 0.2 * service_description_embedding
                tool_id = sha256(f"{mcp_settings.name}_{tool_data['name']}".encode()).hexdigest()
                tool_id = str(uuid5(NAMESPACE_DNS, tool_id))
                await self.qdrant_client.upsert(
                    collection_name=self.vector_store_config.qdrant_tools_collection_name,
                    points=[
                        PointStruct(
                            id=tool_id,
                            payload={
                                "name": tool_data["name"],
                                "parent_service": mcp_settings.name,
                                "description": tool_data["description"],
                                "input_schema": tool_data["input_schema"],
                            },
                            vector=tool_embedding.tolist()
                        )
                    ]
                )
            
            service_id = str(uuid5(NAMESPACE_DNS, sha256(f"{mcp_settings.name}".encode()).hexdigest()))

            await self.qdrant_client.upsert(
                collection_name=self.vector_store_config.qdrant_services_collection_name,
                points=[
                    PointStruct(
                        id=service_id,
                        payload={
                            "name": mcp_settings.name, 
                            **service_description,
                            "tools": [tool_data["name"] for tool_data in tool_data_list]
                        },
                        vector=service_description_embedding.tolist()
                    )
                ]
            )
                

    async def init_mcp_servers(self):
        for mcp_settings in self.mcp_servers.mcpServers:
            if mcp_settings.ignore:
                continue 
            await self.index_mcp_server(mcp_settings)
    
    async def start_mcp_server(self, mcp_settings:McpSettings):
        if mcp_settings.ignore:
            return 
        
        server_parameters = StdioServerParameters(
            command=mcp_settings.mcp_config.command,
            args=mcp_settings.mcp_config.args,
            env=mcp_settings.mcp_config.env
        )
        
        stdio_transport = await self.resources_manager.enter_async_context(stdio_client(server=server_parameters))
        reader, writer = stdio_transport
        mcp_session = await self.resources_manager.enter_async_context(ClientSession(read_stream=reader, write_stream=writer))
        await mcp_session.initialize()
        async with self.mcp_session_mutex:
            self.service_name2mcp_session[mcp_settings.name] = mcp_session
        logger.info(f"MCP server {mcp_settings.name} started")

    async def start_mcp_servers(self):
        for mcp_settings in self.mcp_servers.mcpServers:
            if mcp_settings.ignore:
                continue 
            await self.start_mcp_server(mcp_settings)
    
    async def embedding(self, texts:List[str]) -> NDArray:
        async with self._build_socket(ctx=self.ctx, socket_type=zmq.DEALER, method="connect", addr=MQEmbedding.OUTER_ROUTER_ADDR) as dealer:
            await dealer.send_multipart([b""], flags=zmq.SNDMORE)
            await dealer.send_pyobj(texts)
            _, encoded_embeddings = await dealer.recv_multipart()
            embeddings = pickle.loads(encoded_embeddings)
            return embeddings
    
    async def enhance_query(self, query:str, context:str) -> List[str]:
        class EnhancedQuery(BaseModel):
            enhanced_queries:List[str] = Field(description="List of 10/15 enhanced queries")

        response:ParsedChatCompletion = await self.openai_client.beta.chat.completions.parse(
            model=OpenaiModel.GPT_41_MINI,
            messages=[
                ChatMessage(role=Role.SYSTEM, content=SystemInstructions.ENHANCE_QUERY),
                ChatMessage(role=Role.USER, content=f"Query: {query}\nContext: {context}")
            ],
            max_tokens=1024,
            response_format=EnhancedQuery
        )

        parsed_data:EnhancedQuery = response.choices[0].message.parsed
        return parsed_data.enhanced_queries
    
    async def statistics(self) -> str:
        services_count = await self.qdrant_client.count(
            collection_name=self.vector_store_config.qdrant_services_collection_name,
            exact=True
        )
        tools_count = await self.qdrant_client.count(
            collection_name=self.vector_store_config.qdrant_tools_collection_name,
            exact=True
        )
        return yaml.dump({
            "services_count": services_count.count,
            "tools_count": tools_count.count
        }, sort_keys=False)
    
    async def scroll_over_services(self, limit:int, offset:Optional[str]=None) -> str:
        scroll_response = await self.qdrant_client.scroll(
            collection_name=self.vector_store_config.qdrant_services_collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            scroll_filter=None
        )
        records, next_offset = scroll_response
        services = []
        for record in records:
            services.append(yaml.dump({
                "name": record.payload["name"],
                "title": record.payload["title"],
           }, sort_keys=False))
        
        return json.dumps({
            "services": services,
            "next_offset": next_offset
        }, indent=3)

    async def search_services(self, query:str, top_n:int=3) -> str:
        context = "finding services (mcp services), each service is a collection of tools that can be used to perform an action"
        enhanced_queries = await self.enhance_query(query, context)
        queries_embeddings = await self.embedding(enhanced_queries)
        
        query_response = await self.qdrant_client.query_points(
            collection_name=self.vector_store_config.qdrant_services_collection_name,
            query=RecommendQuery(
                recommend=RecommendInput(
                    positive=queries_embeddings,
                    negative=None,
                    strategy=RecommendStrategy.AVERAGE_VECTOR
                )
            ),
            limit=top_n,
            with_payload=True,
            with_vectors=False,
            query_filter=None
        )
        
        services = []
        for record in query_response.points:
            services.append(
                yaml.dump(record.payload, sort_keys=False)
            )
        return "\n###\n".join(services)

    async def search_tools(self, query:str, top_n:int=3, services:Optional[List[str]]=None) -> str:
        context = "finding tools (mcp tools), each tool is a function that can be used to perform an action"
        enhanced_queries = await self.enhance_query(query, context)
        queries_embeddings = await self.embedding(enhanced_queries)
        
        query_response = await self.qdrant_client.query_points(
            collection_name=self.vector_store_config.qdrant_tools_collection_name,
            query=RecommendQuery(
                recommend=RecommendInput(
                    positive=queries_embeddings,
                    negative=None,
                    strategy=RecommendStrategy.AVERAGE_VECTOR
                )
            ),
            limit=top_n,
            with_payload=True,
            with_vectors=False,
            query_filter=None if services is None else Filter(
                should=[
                    FieldCondition(
                        key="parent_service",
                        match=MatchAny(any=services)
                    )
                ]
            )
        )
        
        tools = []
        for record in query_response.points:
            tools.append(
                yaml.dump(record.payload, sort_keys=False)
            )
        return "\n###\n".join(tools)
    
    async def read_service_information(self, service_name:str) -> str:
        service_id = sha256(f"{service_name}".encode()).hexdigest()
        service_id = str(uuid5(NAMESPACE_DNS, service_id))
        records = await self.qdrant_client.retrieve(
            collection_name=self.vector_store_config.qdrant_services_collection_name,
            ids=[service_id], 
            with_payload=True,
            with_vectors=False
        )
        if len(records) == 0:
            return f"Service {service_name} not found. Please check the service name"
        return yaml.dump(records[0].payload, sort_keys=False)
    
    async def read_tool_information(self, service_name:str, tool_name:str) -> str:
        tool_id = sha256(f"{service_name}_{tool_name}".encode()).hexdigest()
        tool_id = str(uuid5(NAMESPACE_DNS, tool_id))
        records = await self.qdrant_client.retrieve(
            collection_name=self.vector_store_config.qdrant_tools_collection_name,
            ids=[tool_id],
            with_payload=True,
            with_vectors=False
        )
        if len(records) == 0:
            return f"Tool {tool_name} not found. Please check the tool name"
        return yaml.dump(records[0].payload, sort_keys=False)
    
    async def execute_tool(self, tool_call_name:str, tool_call_arguments:Dict[str, Any], service_name:str, execution_mode:str="synchronous") -> str:
        async def _synchronous_execution() -> str:
            async with self.mcp_session_mutex:
                session_reference = self.service_name2mcp_session.get(service_name, None)
                if session_reference is None:
                    return f"MCP server {service_name} not found"
            result = await session_reference.call_tool(name=tool_call_name, arguments=tool_call_arguments)
            response = '\n'.join([ res.model_dump_json(indent=2) for res in result.content ])
            
            return response
        
        match execution_mode:
            case "synchronous":
                response = await _synchronous_execution()
                return response
            case "asynchronous":
                task_id = str(uuid4())
                async with self.background_tasks_mutex:
                    self.background_tasks[task_id] = asyncio.create_task(_synchronous_execution())
                return json.dumps({"task_id": task_id})
            case _:
                raise ValueError(f"Invalid mode: {execution_mode}")
    
    async def get_background_task_status(self, task_id:str) -> str:
        async with self.background_tasks_mutex:
            task = self.background_tasks.get(task_id, None)
            if task is None:
                return yaml.dump({
                    "status": "not_found",
                    "message": f"Task {task_id} not found",
                    "result": None
                }, sort_keys=False)
            
            if task.done():
                try:
                    result = task.result()
                    return yaml.dump({
                        "status": "completed",
                        "message": f"Task {task_id} is completed",
                        "result": result
                    }, sort_keys=False)
                except Exception as e:
                    return yaml.dump({
                        "status": "failed",
                        "message": f"Task {task_id} failed with error: {e}",
                        "result": None
                    }, sort_keys=False)
                finally:
                    del self.background_tasks[task_id]
                    logger.info(f"Task {task_id} done and removed from the background tasks")

            return yaml.dump({
                "status": "running",
                "message": f"Task {task_id} is running",
                "result": None
            }, sort_keys=False)
            
    async def get_system_instructions(self) -> str:
        return SystemInstructions.ACTOR.value
    
        
    async def main_runner(self, tool_call:Dict[str, Any]) -> str:
        function_name = tool_call['function']['name']
        arguments = json.loads(tool_call['function']['arguments'])
        action = arguments['action']
        
        print(function_name)
        print(json.dumps(arguments, indent=2))
        
        target_endpoint = attrgetter(action)(self)
        match action:
            case 'get_system_instructions':
                tool_call_result_content = await target_endpoint()
            case 'statistics':
                tool_call_result_content = await target_endpoint()
            case 'scroll_over_services':
                tool_call_result_content = await target_endpoint(
                    limit=arguments['limit'],
                    offset=arguments['offset']
                )
            case 'read_service_information':
                tool_call_result_content = await target_endpoint(
                    service_name=arguments['service_name']
                )
            case 'read_tool_information':
                tool_call_result_content = await target_endpoint(
                    service_name=arguments['service_name'],
                    tool_name=arguments['tool_name']
                )
            case 'search_services':
                tool_call_result_content = await target_endpoint(
                    query=arguments['query'],
                    top_n=arguments.get('top_n', 3)
                )
            case 'search_tools':
                tool_call_result_content = await target_endpoint(
                    query=arguments['query'],
                    top_n=arguments.get('top_n', 3),
                    services=arguments.get('services', None)
                )
            case 'execute_tool':
                tool_call_result_content = await target_endpoint(
                    tool_call_name=arguments['tool_name'],
                    tool_call_arguments=arguments['tool_arguments'],
                    service_name=arguments['service_name'],
                    execution_mode=arguments.get('execution_mode', 'synchronous')
                )
            case 'get_background_task_status':
                tool_call_result_content = await target_endpoint(
                    task_id=arguments['task_id']
                )
            case _:
                raise ValueError(f"Invalid action: {action}")
         
        assert isinstance(tool_call_result_content, str), "tool_call_result_content must be a string"
        return tool_call_result_content

    async def export_tools(self, file_path:Optional[str]=None, services:Optional[List[str]]=None) -> str:
        total_number_of_tools = await self.qdrant_client.count(collection_name=self.vector_store_config.qdrant_tools_collection_name, exact=True)
        if total_number_of_tools.count == 0:
            raise ValueError("No tools found")
        
        scroll_response = await self.qdrant_client.scroll(
            collection_name=self.vector_store_config.qdrant_tools_collection_name,
            limit=total_number_of_tools.count,
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(
                should=[
                    FieldCondition(
                        key="parent_service",
                        match=MatchAny(any=services)
                    )
                ]
            ) if services is not None else None
        )

        tools = []
        for record in scroll_response[0]:
            tools.append(record.payload)
        if len(tools) == 0:
            raise ValueError("No tools found. Please check the services list")
        sorted_tools = sorted(tools, key=itemgetter("parent_service"))
        sorted_tools = [yaml.dump(tool, sort_keys=False) for tool in sorted_tools]
        
        if file_path is not None:
            with open(file_path, "w") as f:
                f.write("\n###\n".join(sorted_tools))
        return "\n###\n".join(sorted_tools)
    