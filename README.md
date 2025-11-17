=# Semantic MCP Discovery API

**Intelligent search and management system for Model Context Protocol (MCP) servers using vector embeddings and AI-powered analysis**

## Overview

Semantic MCP Discovery API transforms how you discover and manage MCP servers by providing intelligent semantic search capabilities. Instead of manually browsing through server lists, simply describe what you're looking for in natural language and find the most relevant servers and tools.

## Key Features

- **Semantic Search**: Natural language queries to find relevant MCP servers and tools
- **AI-Powered Analysis**: Automatic server description and capability extraction
- **Bulk Management**: Upload multiple server configurations at once
- **Parallel Processing**: Fast concurrent server analysis and indexing
- **Vector Similarity**: Advanced embedding-based similarity matching
- **Standard Format**: Compatible with MCP server configuration formats

## Quick Start

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation
```bash
git clone https://github.com/milkymap/semantic_mcp.git
cd semantic_mcp
uv sync
```

### Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Configure your settings in `.env`:
```bash
# API Configuration
API_KEY=your-secret-api-key
HOST=0.0.0.0
PORT=8000

# Vector Store (Qdrant)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-qdrant-key

# Embedding Service
OPENAI_API_KEY=your-openai-api-key

# Encryption (for secure storage of server configurations)
ENCRYPTION_KEY=your-32-character-encryption-key
```

### Running the Server
```bash
uv run python -m semantic_mcp
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Server Management
- `POST /api/mcp/servers` - Add single MCP server
- `POST /api/mcp/servers/bulk` - Bulk upload servers from JSON file
- `GET /api/mcp/servers` - List all servers
- `GET /api/mcp/servers/{server_name}` - Get server details
- `DELETE /api/mcp/servers/{server_name}` - Remove server

### Semantic Search
- `POST /api/mcp/servers/search` - Search servers by description
- `POST /api/mcp/tools/search` - Search tools by functionality

### Tools & Statistics
- `GET /api/mcp/servers/{server_name}/tools` - List server tools
- `GET /api/mcp/statistics` - Get system statistics

## Usage Examples

### Adding Servers

**Single Server:**
```bash
curl -X POST "http://localhost:8000/api/mcp/servers" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "server_name": "weather-server",
    "command": "python",
    "args": ["-m", "weather_mcp"],
    "env": {"API_KEY": "your-key"}
  }'
```

**Bulk Upload:**
```bash
curl -X POST "http://localhost:8000/api/mcp/servers/bulk" \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@servers_config.json"
```

### Semantic Search
```bash
curl -X POST "http://localhost:8000/api/mcp/servers/search" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "weather data and forecasting tools",
    "limit": 10
  }'
```

## Configuration Format

Bulk server configuration uses this JSON structure:
```json
{
  "mcpServers": {
    "weather-server": {
      "command": "python",
      "args": ["-m", "weather_mcp"],
      "env": {
        "API_KEY": "your-weather-api-key"
      }
    },
    "file-manager": {
      "command": "node",
      "args": ["./dist/index.js"],
      "env": {}
    }
  }
}
```

## Architecture

- **FastAPI**: High-performance async web framework
- **Qdrant**: Vector database for similarity search
- **OpenAI Embeddings**: Text-to-vector conversion
- **Pydantic**: Data validation and serialization
- **Asyncio**: Concurrent processing for performance

## Response Format

Search and bulk operations return detailed results:
```json
{
  "results": [
    {
      "server_name": "weather-server",
      "success": true,
      "message": "Successfully analyzed and indexed",
      "server_info": {
        "title": "Weather Data Provider",
        "summary": "Provides real-time weather information...",
        "capabilities": ["weather-query", "forecasting"],
        "nb_tools": 5
      }
    }
  ],
  "successful_count": 1,
  "total_count": 1,
  "processing_time_ms": 1250.5
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links

- **Issues**: [Report bugs or request features](https://github.com/milkymap/semantic_mcp/issues)
- **MCP Protocol**: [Learn about Model Context Protocol](https://modelcontextprotocol.io/)