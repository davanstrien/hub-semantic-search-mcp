#!/usr/bin/env python3
"""
MCP Server for Hugging Face Dataset and Model Search API
"""

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
    CallToolRequest,
    ListToolsResult,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HFSearchServer:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        await self.client.aclose()

    async def search_datasets(
        self,
        query: str,
        k: int = 5,
        sort_by: str = "similarity",
        min_likes: int = 0,
        min_downloads: int = 0
    ) -> Dict[str, Any]:
        """Search for datasets based on a text query"""
        params = {
            "query": query,
            "k": k,
            "sort_by": sort_by,
            "min_likes": min_likes,
            "min_downloads": min_downloads
        }

        response = await self.client.get(
            f"{self.base_url}/search/datasets",
            params=params
        )
        response.raise_for_status()
        return response.json()

    async def find_similar_datasets(
        self,
        dataset_id: str,
        k: int = 5,
        sort_by: str = "similarity",
        min_likes: int = 0,
        min_downloads: int = 0
    ) -> Dict[str, Any]:
        """Find similar datasets to a specified dataset"""
        params = {
            "dataset_id": dataset_id,
            "k": k,
            "sort_by": sort_by,
            "min_likes": min_likes,
            "min_downloads": min_downloads
        }

        response = await self.client.get(
            f"{self.base_url}/similarity/datasets",
            params=params
        )
        response.raise_for_status()
        return response.json()

    async def search_models(
        self,
        query: str,
        k: int = 5,
        sort_by: str = "similarity",
        min_likes: int = 0,
        min_downloads: int = 0,
        min_param_count: int = 0,
        max_param_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """Search for models based on a text query"""
        params = {
            "query": query,
            "k": k,
            "sort_by": sort_by,
            "min_likes": min_likes,
            "min_downloads": min_downloads,
            "min_param_count": min_param_count
        }
        if max_param_count is not None:
            params["max_param_count"] = max_param_count

        response = await self.client.get(
            f"{self.base_url}/search/models",
            params=params
        )
        response.raise_for_status()
        return response.json()

    async def find_similar_models(
        self,
        model_id: str,
        k: int = 5,
        sort_by: str = "similarity",
        min_likes: int = 0,
        min_downloads: int = 0,
        min_param_count: int = 0,
        max_param_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """Find similar models to a specified model"""
        params = {
            "model_id": model_id,
            "k": k,
            "sort_by": sort_by,
            "min_likes": min_likes,
            "min_downloads": min_downloads,
            "min_param_count": min_param_count
        }
        if max_param_count is not None:
            params["max_param_count"] = max_param_count

        response = await self.client.get(
            f"{self.base_url}/similarity/models",
            params=params
        )
        response.raise_for_status()
        return response.json()

    async def get_trending_models(
        self,
        limit: int = 10,
        min_likes: int = 0,
        min_downloads: int = 0,
        min_param_count: int = 0,
        max_param_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get trending models with their summaries"""
        params = {
            "limit": limit,
            "min_likes": min_likes,
            "min_downloads": min_downloads,
            "min_param_count": min_param_count
        }
        if max_param_count is not None:
            params["max_param_count"] = max_param_count

        response = await self.client.get(
            f"{self.base_url}/trending/models",
            params=params
        )
        response.raise_for_status()
        return response.json()

    async def get_trending_datasets(
        self,
        limit: int = 10,
        min_likes: int = 0,
        min_downloads: int = 0
    ) -> Dict[str, Any]:
        """Get trending datasets with their summaries"""
        params = {
            "limit": limit,
            "min_likes": min_likes,
            "min_downloads": min_downloads
        }

        response = await self.client.get(
            f"{self.base_url}/trending/datasets",
            params=params
        )
        response.raise_for_status()
        return response.json()

    async def download_model_card(self, model_id: str) -> str:
        """
        Download the README card for a HuggingFace model.
        
        Args:
            model_id (str): The model ID (e.g., 'username/model-name')
        
        Returns:
            str: The content of the model card (README.md)
        """
        url = f"https://huggingface.co/{model_id}/raw/main/README.md"
        response = await self.client.get(url)
        response.raise_for_status()
        return response.text

    async def download_dataset_card(self, dataset_id: str) -> str:
        """
        Download the README card for a HuggingFace dataset.
        
        Args:
            dataset_id (str): The dataset ID (e.g., 'username/dataset-name')
        
        Returns:
            str: The content of the dataset card (README.md)
        """
        url = f"https://huggingface.co/datasets/{dataset_id}/raw/main/README.md"
        response = await self.client.get(url)
        response.raise_for_status()
        return response.text

# Initialize server and API client
server = Server("hf-search")
api_client: Optional[HFSearchServer] = None

@server.list_tools()
async def list_tools() -> ListToolsResult:
    """List available tools"""
    return ListToolsResult(
        tools=[
            Tool(
                name="search_datasets",
                description="Search for datasets based on a text query",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return (1-100)",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 5
                        },
                        "sort_by": {
                            "type": "string",
                            "description": "Sort method for results",
                            "enum": ["similarity", "likes", "downloads", "trending"],
                            "default": "similarity"
                        },
                        "min_likes": {
                            "type": "integer",
                            "description": "Minimum likes filter",
                            "minimum": 0,
                            "default": 0
                        },
                        "min_downloads": {
                            "type": "integer",
                            "description": "Minimum downloads filter",
                            "minimum": 0,
                            "default": 0
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="find_similar_datasets",
                description="Find datasets similar to a specified dataset",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "Dataset ID to find similar datasets for"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return (1-100)",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 5
                        },
                        "sort_by": {
                            "type": "string",
                            "description": "Sort method for results",
                            "enum": ["similarity", "likes", "downloads", "trending"],
                            "default": "similarity"
                        },
                        "min_likes": {
                            "type": "integer",
                            "description": "Minimum likes filter",
                            "minimum": 0,
                            "default": 0
                        },
                        "min_downloads": {
                            "type": "integer",
                            "description": "Minimum downloads filter",
                            "minimum": 0,
                            "default": 0
                        }
                    },
                    "required": ["dataset_id"]
                }
            ),
            Tool(
                name="search_models",
                description="Search for models based on a text query with optional parameter count filtering",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return (1-100)",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 5
                        },
                        "sort_by": {
                            "type": "string",
                            "description": "Sort method for results",
                            "enum": ["similarity", "likes", "downloads", "trending"],
                            "default": "similarity"
                        },
                        "min_likes": {
                            "type": "integer",
                            "description": "Minimum likes filter",
                            "minimum": 0,
                            "default": 0
                        },
                        "min_downloads": {
                            "type": "integer",
                            "description": "Minimum downloads filter",
                            "minimum": 0,
                            "default": 0
                        },
                        "min_param_count": {
                            "type": "integer",
                            "description": "Minimum parameter count (excludes models with unknown params)",
                            "minimum": 0,
                            "default": 0
                        },
                        "max_param_count": {
                            "type": ["integer", "null"],
                            "description": "Maximum parameter count (null for no limit)",
                            "minimum": 0,
                            "default": None
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="find_similar_models",
                description="Find models similar to a specified model",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_id": {
                            "type": "string",
                            "description": "Model ID to find similar models for"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return (1-100)",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 5
                        },
                        "sort_by": {
                            "type": "string",
                            "description": "Sort method for results",
                            "enum": ["similarity", "likes", "downloads", "trending"],
                            "default": "similarity"
                        },
                        "min_likes": {
                            "type": "integer",
                            "description": "Minimum likes filter",
                            "minimum": 0,
                            "default": 0
                        },
                        "min_downloads": {
                            "type": "integer",
                            "description": "Minimum downloads filter",
                            "minimum": 0,
                            "default": 0
                        },
                        "min_param_count": {
                            "type": "integer",
                            "description": "Minimum parameter count (excludes models with unknown params)",
                            "minimum": 0,
                            "default": 0
                        },
                        "max_param_count": {
                            "type": ["integer", "null"],
                            "description": "Maximum parameter count (null for no limit)",
                            "minimum": 0,
                            "default": None
                        }
                    },
                    "required": ["model_id"]
                }
            ),
            Tool(
                name="get_trending_models",
                description="Get trending models with their summaries and optional filtering",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of results to return (1-100)",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 10
                        },
                        "min_likes": {
                            "type": "integer",
                            "description": "Minimum likes filter",
                            "minimum": 0,
                            "default": 0
                        },
                        "min_downloads": {
                            "type": "integer",
                            "description": "Minimum downloads filter",
                            "minimum": 0,
                            "default": 0
                        },
                        "min_param_count": {
                            "type": "integer",
                            "description": "Minimum parameter count (excludes models with unknown params)",
                            "minimum": 0,
                            "default": 0
                        },
                        "max_param_count": {
                            "type": ["integer", "null"],
                            "description": "Maximum parameter count (null for no limit)",
                            "minimum": 0,
                            "default": None
                        }
                    }
                }
            ),
            Tool(
                name="get_trending_datasets",
                description="Get trending datasets with their summaries",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of results to return (1-100)",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 10
                        },
                        "min_likes": {
                            "type": "integer",
                            "description": "Minimum likes filter",
                            "minimum": 0,
                            "default": 0
                        },
                        "min_downloads": {
                            "type": "integer",
                            "description": "Minimum downloads filter",
                            "minimum": 0,
                            "default": 0
                        }
                    }
                }
            ),
            Tool(
                name="download_model_card",
                description="Download the README card for a HuggingFace model",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_id": {
                            "type": "string",
                            "description": "The model ID (e.g., 'username/model-name')"
                        }
                    },
                    "required": ["model_id"]
                }
            ),
            Tool(
                name="download_dataset_card",
                description="Download the README card for a HuggingFace dataset",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "The dataset ID (e.g., 'username/dataset-name')"
                        }
                    },
                    "required": ["dataset_id"]
                }
            )
        ]
    )

@server.call_tool()
async def call_tool(request: CallToolRequest) -> CallToolResult:
    """Handle tool calls"""
    global api_client

    if api_client is None:
        # Initialize API client with base URL from environment or default
        import os
        base_url = os.getenv("HF_SEARCH_API_URL", "http://localhost:8000")
        api_client = HFSearchServer(base_url)

    try:
        # Parse arguments
        args = request.params.arguments if hasattr(request.params, 'arguments') else {}

        # Format results helper
        def format_dataset_results(data: Dict[str, Any]) -> str:
            results = data.get("results", [])
            if not results:
                return "No datasets found."

            output = []
            for i, result in enumerate(results, 1):
                output.append(f"{i}. **{result['dataset_id']}**")
                output.append(f"   - Summary: {result['summary']}")
                output.append(f"   - Similarity: {result['similarity']:.3f}")
                output.append(f"   - Likes: {result['likes']:,} | Downloads: {result['downloads']:,}")
                output.append("")

            return "\n".join(output)

        def format_model_results(data: Dict[str, Any]) -> str:
            results = data.get("results", [])
            if not results:
                return "No models found."

            output = []
            for i, result in enumerate(results, 1):
                output.append(f"{i}. **{result['model_id']}**")
                output.append(f"   - Summary: {result['summary']}")
                output.append(f"   - Similarity: {result['similarity']:.3f}")
                output.append(f"   - Likes: {result['likes']:,} | Downloads: {result['downloads']:,}")
                if result.get('param_count') is not None and result['param_count'] > 0:
                    # Format parameter count nicely
                    param_count = result['param_count']
                    if param_count >= 1_000_000_000:
                        param_str = f"{param_count / 1_000_000_000:.1f}B"
                    elif param_count >= 1_000_000:
                        param_str = f"{param_count / 1_000_000:.1f}M"
                    elif param_count >= 1_000:
                        param_str = f"{param_count / 1_000:.1f}K"
                    else:
                        param_str = str(param_count)
                    output.append(f"   - Parameters: {param_str}")
                output.append("")

            return "\n".join(output)

        # Route to appropriate method
        if request.params.name == "search_datasets":
            result = await api_client.search_datasets(**args)
            formatted = format_dataset_results(result)
            return CallToolResult(
                content=[TextContent(text=formatted)],
                isError=False
            )

        elif request.params.name == "find_similar_datasets":
            result = await api_client.find_similar_datasets(**args)
            formatted = format_dataset_results(result)
            return CallToolResult(
                content=[TextContent(text=formatted)],
                isError=False
            )

        elif request.params.name == "search_models":
            result = await api_client.search_models(**args)
            formatted = format_model_results(result)
            return CallToolResult(
                content=[TextContent(text=formatted)],
                isError=False
            )

        elif request.params.name == "find_similar_models":
            result = await api_client.find_similar_models(**args)
            formatted = format_model_results(result)
            return CallToolResult(
                content=[TextContent(text=formatted)],
                isError=False
            )

        elif request.params.name == "get_trending_models":
            result = await api_client.get_trending_models(**args)
            formatted = format_model_results(result)
            return CallToolResult(
                content=[TextContent(text=formatted)],
                isError=False
            )

        elif request.params.name == "get_trending_datasets":
            result = await api_client.get_trending_datasets(**args)
            formatted = format_dataset_results(result)
            return CallToolResult(
                content=[TextContent(text=formatted)],
                isError=False
            )

        elif request.params.name == "download_model_card":
            result = await api_client.download_model_card(**args)
            return CallToolResult(
                content=[TextContent(text=result)],
                isError=False
            )

        elif request.params.name == "download_dataset_card":
            result = await api_client.download_dataset_card(**args)
            return CallToolResult(
                content=[TextContent(text=result)],
                isError=False
            )

        else:
            return CallToolResult(
                content=[TextContent(text=f"Unknown tool: {request.params.name}")],
                isError=True
            )

    except httpx.HTTPStatusError as e:
        error_msg = f"API request failed with status {e.response.status_code}: {e.response.text}"
        logger.error(error_msg)
        return CallToolResult(
            content=[TextContent(text=error_msg)],
            isError=True
        )
    except Exception as e:
        error_msg = f"Error calling tool {request.params.name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return CallToolResult(
            content=[TextContent(text=error_msg)],
            isError=True
        )

async def main():
    """Main entry point"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

        # Cleanup
        if api_client:
            await api_client.close()

if __name__ == "__main__":
    asyncio.run(main())
