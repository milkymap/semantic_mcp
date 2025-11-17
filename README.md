# Semantic MCP Discovery API

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

- Python 3.12+ or Docker
- [uv](https://github.com/astral-sh/uv) package manager (for local development)

## Installation & Running

### Option 1: Docker (Recommended)

#### Build the Docker image:
```bash
git clone https://github.com/milkymap/semantic_mcp.git
cd semantic_mcp
docker build -t semantic-mcp .
```

#### Create your environment file:
```bash
cp .env .env.secret
# Edit .env.secret with your actual API keys (see Environment Variables section below)
```

#### Run with Docker:
```bash
docker run -d \
  --name semantic-mcp \
  --env-file .env.secret \
  -p 8000:8000 \
  -v $(pwd)/data:/home/solver/data \
  semantic-mcp
```

#### Alternative Docker Compose (optional):
Create a `docker-compose.yml`:
```yaml
version: '3.8'
services:
  semantic-mcp:
    build: .
    env_file: .env.secret
    ports:
      - "8000:8000"
    volumes:
      - ./data:/home/solver/data
    restart: unless-stopped
```

Then run:
```bash
docker compose up -d
```

### Option 2: Local Development

#### Installation:
```bash
git clone https://github.com/milkymap/semantic_mcp.git
cd semantic_mcp
uv sync
```

#### Running the Server:
```bash
# Make sure your .env file is configured (see Environment Variables section)
uv run python -m semantic_mcp
```

The API will be available at `http://localhost:8000`

## Environment Variables

Create a `.env.secret` file (or `.env` for local development) with the following required variables:

### 🔑 Required API Keys

```bash
# OpenAI API (for AI-powered server analysis)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-proj-your-openai-api-key-here

# Cohere API (for text embeddings)
# Get from: https://dashboard.cohere.com/api-keys
COHERE_API_KEY=your-cohere-api-key-here

# API Authentication Key (for securing your API endpoints)
# Generate a secure random string, e.g., using: openssl rand -base64 32
API_KEY=your-secure-api-authentication-key

# Encryption Key (for encrypting MCP server configurations)
# Generate using: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
ENCRYPTION_KEY=your-32-byte-base64-encryption-key
```

### 🗄️ Storage Configuration

```bash
# Local path for Qdrant vector database storage
# Docker: Use /home/solver/data (mapped to host volume)
# Local: Use any local directory path
QDRANT_STORAGE_PATH=/home/solver/data/qdrant_storage
```

### ⚙️ Optional Configuration

```bash
# Server Configuration (defaults shown)
API_SERVER_HOST=0.0.0.0
API_SERVER_PORT=8000
API_SERVER_WORKERS=1

# AI Model Configuration
MCP_DESCRIPTOR_MODEL_NAME=gpt-4.1-mini
EMBEDDING_MODEL_NAME=embed-v4.0
DIMENSIONS=1024
INDEX_NAME=semantic-mcp-index

# Performance
THREAD_POOL_MAX_WORKERS=32
```

### 🔐 Security Notes

- **Never commit `.env.secret` or `.env` files to version control**
- Keep your API keys secure and rotate them regularly
- The `API_KEY` is used to authenticate requests to your API
- The `ENCRYPTION_KEY` encrypts sensitive MCP server configurations before storage

### 💡 Generating Secure Keys

```bash
# Generate API authentication key
openssl rand -base64 32

# Generate encryption key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

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