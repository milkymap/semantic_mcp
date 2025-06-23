# Semantic MCP

A semantic Model Context Protocol (MCP) router that provides intelligent service discovery and tool management for AI agents. This project enables dynamic tool routing based on vector similarity search, allowing AI systems to efficiently discover and use relevant tools from multiple MCP servers.

## Features

- **Semantic Tool Discovery**: Uses vector embeddings to find the most relevant tools for a given task
- **Multi-Server MCP Routing**: Connects to multiple MCP servers and provides unified access to their tools
- **Vector Store Integration**: Powered by Qdrant for efficient similarity search of tools and services  
- **Multiple LLM Support**: Compatible with OpenAI, Anthropic Claude, and Google Gemini models
- **Message Queue Architecture**: Scalable embedding service using ZeroMQ
- **FastAPI Integration**: Exposes MCP functionality via REST API
- **Interactive Agent Loop**: Built-in conversational AI agent for testing

## Architecture

The project consists of several key components:

- **MCPRouter**: Core router that manages multiple MCP server connections
- **MQEmbedding**: Message queue-based embedding service for semantic search
- **MCPEngine**: FastAPI server exposing MCP functionality 
- **AgentLoop**: Interactive AI agent client for testing and demonstrations

## Installation

### Prerequisites

- Python 3.12+
- Qdrant vector database
- API keys for desired LLM providers (OpenAI, Anthropic, Google)

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd semantic_mcp

# Install using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Environment Configuration

Create a `.env` file with the required configuration:

```env
# API Keys
OPENAI_API_KEY=your_openai_key

# Vector Store Configuration
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384
HF_HOME=./cache
DEVICE=cpu
EMBEDDING_NB_WORKERS=4

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_key
QDRANT_SERVICES_COLLECTION_NAME=mcp_services
QDRANT_TOOLS_COLLECTION_NAME=mcp_tools
```

## Usage

### 1. Start the Embedding Service

```bash
python -m semantic_mcp launch-embedding-service
```

### 2. Configure MCP Servers

Create a `mcp-servers.json` file defining your MCP servers:

```json
{
  "mcpServers": [
    {
      "name": "filesystem",
      "mcp_config": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"],
        "env": {}
      },
      "allowed_tools": [],
      "ignore": false,
      "startup_timeout": 30,
      "index": true
    }
  ]
}
```

### 3. Start the MCP Router

```bash
python -m semantic_mcp launch-mcp-router -j mcp-servers.json
```

### 4. Launch the API Server

```bash
python -m semantic_mcp launch-api-server -j mcp-servers.json
```

### 5. Run the Interactive Agent

```bash
python -m semantic_mcp launch-agent-loop
```

## Configuration

### MCP Server Configuration

Each MCP server in `mcp-servers.json` supports:

- `name`: Unique identifier for the server
- `mcp_config`: Command and arguments to start the server
- `allowed_tools`: Whitelist of tools (empty = all allowed)
- `ignore`: Skip this server if true
- `startup_timeout`: Seconds to wait for server startup
- `index`: Whether to include tools in vector search index

### Vector Store Settings

Configure embedding and search behavior:

- `EMBEDDING_MODEL_NAME`: HuggingFace model for generating embeddings
- `EMBEDDING_DIMENSIONS`: Vector dimension size
- `DEVICE`: CPU or CUDA device for embeddings
- `EMBEDDING_NB_WORKERS`: Parallel workers for embedding generation

## API Endpoints

When running the API server, the following endpoints are available:

- `GET /mcp`: MCP protocol endpoint
- `POST /mcp`: Execute MCP tool calls
- Additional FastAPI endpoints for service management

## Development

### Project Structure

```
src/semantic_mcp/
├── __init__.py          # Main entry point
├── __main__.py          # CLI commands
├── backend/             # Core MCP routing logic
│   ├── mcp_router.py    # Main router implementation
│   ├── mcp_api_server.py # FastAPI server
│   ├── mq_embedding.py  # Embedding service
│   └── mcp_types.py     # Type definitions
├── client/              # Agent client implementation
│   └── agent_loop.py    # Interactive agent
└── settings/            # Configuration management
    ├── credentials.py   # API key settings
    └── vector_store.py  # Vector store settings
```

### Running Tests

```bash
# Add test commands here when available
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add license information]

## Support

For issues and questions, please open an issue on the GitHub repository.