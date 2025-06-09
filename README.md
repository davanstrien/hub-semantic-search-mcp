# Hugging Face Hub Semantic Search MCP Server

An MCP (Model Context Protocol) server that provides semantic search capabilities for Hugging Face models and datasets. This server enables Claude and other MCP-compatible clients to search, discover, and explore the Hugging Face ecosystem using natural language queries.

## Features

- **Semantic Search**: AI-powered similarity search (not just keyword matching)
- **Dataset Search**: Find datasets based on natural language descriptions
- **Model Search**: Find models with optional parameter count filtering
- **Similarity Search**: Find similar models/datasets to a given one
- **Trending Content**: Get currently trending models and datasets
- **Detailed Metadata**: Access comprehensive technical information via HuggingFace API
- **Model/Dataset Cards**: Download README cards for detailed information

## Tools Available

### Dataset Tools
- `search_datasets`: Search datasets using natural language queries
- `find_similar_datasets`: Find datasets similar to a specified one
- `get_trending_datasets`: Get currently trending datasets
- `get_dataset_info`: Get detailed metadata for a specific dataset
- `download_dataset_card`: Download README card for a dataset

### Model Tools  
- `search_models`: Search models using natural language queries with parameter filtering
- `find_similar_models`: Find models similar to a specified one
- `get_trending_models`: Get currently trending models with parameter filtering
- `get_model_info`: Get detailed metadata for a specific model
- `get_model_safetensors_metadata`: Get model architecture details and parameter count from safetensors
- `download_model_card`: Download README card for a model

## Installation

### Prerequisites
- [UV](https://docs.astral.sh/uv/) - Fast Python package installer
- Claude Desktop or another MCP-compatible client

### Quick Start
No installation needed! UV will automatically fetch and run the server.

## Configuration

### Claude Desktop Setup

Add the following to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "huggingface-hub-search": {
      "command": "uvx",
      "args": [
        "git+https://github.com/davanstrien/hub-semantic-search-mcp.git"
      ],
      "env": {
        "HF_SEARCH_API_URL": "https://davanstrien-huggingface-datasets-search-v2.hf.space"
      }
    }
  }
}
```

### Alternative: Local Development Setup

If you want to contribute or modify the code:

```bash
# Clone the repository
git clone https://github.com/davanstrien/hub-semantic-search-mcp.git
cd hub-semantic-search-mcp

# Install dependencies with UV
uv sync
```

Then configure Claude Desktop to use the local version:
```json
{
  "mcpServers": {
    "huggingface-hub-search": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/hub-semantic-search-mcp",
        "run",
        "python",
        "app.py"
      ],
      "env": {
        "HF_SEARCH_API_URL": "https://davanstrien-huggingface-datasets-search-v2.hf.space"
      }
    }
  }
}
```

## Usage Examples

Once configured, you can use the tools in Claude Desktop:

### Search for Datasets
> "Find datasets about climate change and weather patterns"

### Search for Models
> "Find small language models under 1B parameters for text generation"

### Find Similar Content
> "Find datasets similar to 'squad' for question answering"

### Get Trending Content
> "Show me the top 10 trending AI models this week"

### Get Detailed Metadata
> "Get detailed information about the 'stanford-nlp/imdb' dataset"
> "Show me technical details and configuration for 'microsoft/DialoGPT-medium'"
> "What's the parameter count and architecture of 'microsoft/DialoGPT-medium'?"

### Download Documentation
> "Download the model card for 'microsoft/DialoGPT-medium'"

## Environment Variables

- `HF_SEARCH_API_URL`: Base URL for the search API (default: https://davanstrien-huggingface-datasets-search-v2.hf.space)

## Search Backend

This MCP server connects to a semantic search API that indexes Hugging Face models and datasets with AI-generated summaries. The search uses embedding-based similarity rather than keyword matching, making it more effective for discovering relevant content based on intent and meaning.

## Development

### Running Locally
```bash
# Run the server directly
uv run python app.py

# Or activate the virtual environment
uv shell
python app.py
```

### Testing with MCP Inspector
```bash
# Test the GitHub version
npx @modelcontextprotocol/inspector uvx git+https://github.com/davanstrien/hub-semantic-search-mcp.git

# Or test locally
npx @modelcontextprotocol/inspector uv run python app.py
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### Development Setup
```bash
git clone https://github.com/davanstrien/hub-semantic-search-mcp.git
cd hub-semantic-search-mcp
uv sync --dev
```

## License

MIT License - see LICENSE file for details.

## Related Projects

- [Model Context Protocol](https://github.com/modelcontextprotocol/python-sdk)
- [Hugging Face Hub](https://huggingface.co/docs/hub/)
- [Claude Desktop](https://claude.ai/download)
- [UV](https://docs.astral.sh/uv/) - Fast Python package installer