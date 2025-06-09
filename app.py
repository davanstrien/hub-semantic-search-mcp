#!/usr/bin/env python3
# /// script
# dependencies = [
#     "mcp",
#     "httpx",
# ]
# ///
"""
MCP Server for Hugging Face Dataset and Model Search API
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import httpx
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("hf-search")

# Global HTTP client
client: Optional[httpx.AsyncClient] = None
base_url = os.getenv("HF_SEARCH_API_URL", "https://davanstrien-huggingface-datasets-search-v2.hf.space")

async def get_client() -> httpx.AsyncClient:
    """Get or create HTTP client"""
    global client
    if client is None:
        client = httpx.AsyncClient(timeout=60.0)
    return client

def format_dataset_results(data: Dict[str, Any]) -> str:
    """Format dataset search results"""
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
    """Format model search results"""
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

@mcp.tool()
async def search_datasets(
    query: str,
    k: int = 5,
    sort_by: str = "similarity",
    min_likes: int = 0,
    min_downloads: int = 0
) -> str:
    """
    Search for datasets using semantic/similarity search based on a text query.
    
    This uses AI-powered semantic search to find datasets whose descriptions 
    are semantically similar to your query, not just keyword matching.
    
    Args:
        query: Search query text (natural language description of what you're looking for)
        k: Number of results to return (1-100)
        sort_by: Sort method for results (similarity, likes, downloads, trending)
        min_likes: Minimum likes filter
        min_downloads: Minimum downloads filter
    """
    client = await get_client()
    params = {
        "query": query,
        "k": k,
        "sort_by": sort_by,
        "min_likes": min_likes,
        "min_downloads": min_downloads
    }
    
    response = await client.get(f"{base_url}/search/datasets", params=params)
    response.raise_for_status()
    data = response.json()
    
    return format_dataset_results(data)

@mcp.tool()
async def find_similar_datasets(
    dataset_id: str,
    k: int = 5,
    sort_by: str = "similarity",
    min_likes: int = 0,
    min_downloads: int = 0
) -> str:
    """
    Find datasets similar to a specified dataset.
    
    Args:
        dataset_id: Dataset ID to find similar datasets for
        k: Number of results to return (1-100)
        sort_by: Sort method for results (similarity, likes, downloads, trending)
        min_likes: Minimum likes filter
        min_downloads: Minimum downloads filter
    """
    client = await get_client()
    params = {
        "dataset_id": dataset_id,
        "k": k,
        "sort_by": sort_by,
        "min_likes": min_likes,
        "min_downloads": min_downloads
    }
    
    response = await client.get(f"{base_url}/similarity/datasets", params=params)
    response.raise_for_status()
    data = response.json()
    
    return format_dataset_results(data)

@mcp.tool()
async def search_models(
    query: str,
    k: int = 5,
    sort_by: str = "similarity",
    min_likes: int = 0,
    min_downloads: int = 0,
    min_param_count: int = 0,
    max_param_count: Optional[int] = None
) -> str:
    """
    Search for models using semantic/similarity search based on a text query with optional parameter count filtering.
    
    This uses AI-powered semantic search to find models whose descriptions
    are semantically similar to your query, not just keyword matching.
    
    Args:
        query: Search query text (natural language description of what you're looking for)
        k: Number of results to return (1-100)
        sort_by: Sort method for results (similarity, likes, downloads, trending)
        min_likes: Minimum likes filter
        min_downloads: Minimum downloads filter
        min_param_count: Minimum parameter count (excludes models with unknown params)
        max_param_count: Maximum parameter count (None for no limit)
    """
    client = await get_client()
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
    
    response = await client.get(f"{base_url}/search/models", params=params)
    response.raise_for_status()
    data = response.json()
    
    return format_model_results(data)

@mcp.tool()
async def find_similar_models(
    model_id: str,
    k: int = 5,
    sort_by: str = "similarity",
    min_likes: int = 0,
    min_downloads: int = 0,
    min_param_count: int = 0,
    max_param_count: Optional[int] = None
) -> str:
    """
    Find models similar to a specified model.
    
    Args:
        model_id: Model ID to find similar models for
        k: Number of results to return (1-100)
        sort_by: Sort method for results (similarity, likes, downloads, trending)
        min_likes: Minimum likes filter
        min_downloads: Minimum downloads filter
        min_param_count: Minimum parameter count (excludes models with unknown params)
        max_param_count: Maximum parameter count (None for no limit)
    """
    client = await get_client()
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
    
    response = await client.get(f"{base_url}/similarity/models", params=params)
    response.raise_for_status()
    data = response.json()
    
    return format_model_results(data)

@mcp.tool()
async def get_trending_models(
    limit: int = 10,
    min_likes: int = 0,
    min_downloads: int = 0,
    min_param_count: int = 0,
    max_param_count: Optional[int] = None
) -> str:
    """
    Get trending models with their summaries and optional filtering.
    
    Args:
        limit: Number of results to return (1-100)
        min_likes: Minimum likes filter
        min_downloads: Minimum downloads filter
        min_param_count: Minimum parameter count (excludes models with unknown params)
        max_param_count: Maximum parameter count (None for no limit)
    """
    client = await get_client()
    params = {
        "limit": limit,
        "min_likes": min_likes,
        "min_downloads": min_downloads,
        "min_param_count": min_param_count
    }
    if max_param_count is not None:
        params["max_param_count"] = max_param_count
    
    response = await client.get(f"{base_url}/trending/models", params=params)
    response.raise_for_status()
    data = response.json()
    
    return format_model_results(data)

@mcp.tool()
async def get_trending_datasets(
    limit: int = 10,
    min_likes: int = 0,
    min_downloads: int = 0
) -> str:
    """
    Get trending datasets with their summaries.
    
    Args:
        limit: Number of results to return (1-100)
        min_likes: Minimum likes filter
        min_downloads: Minimum downloads filter
    """
    client = await get_client()
    params = {
        "limit": limit,
        "min_likes": min_likes,
        "min_downloads": min_downloads
    }
    
    response = await client.get(f"{base_url}/trending/datasets", params=params)
    response.raise_for_status()
    data = response.json()
    
    return format_dataset_results(data)

@mcp.tool()
async def download_model_card(model_id: str) -> str:
    """
    Download the README card for a HuggingFace model.
    
    Args:
        model_id: The model ID (e.g., 'username/model-name')
    
    Returns:
        The content of the model card (README.md)
    """
    client = await get_client()
    url = f"https://huggingface.co/{model_id}/raw/main/README.md"
    response = await client.get(url)
    response.raise_for_status()
    return response.text

@mcp.tool()
async def get_dataset_info(dataset_id: str) -> str:
    """
    Get detailed metadata information for a HuggingFace dataset.
    
    Returns structured information including tags, license, downloads, 
    likes, dataset structure, configuration, and other metadata.
    
    Args:
        dataset_id: The dataset ID (e.g., 'username/dataset-name')
    
    Returns:
        JSON string with comprehensive dataset metadata
    """
    client = await get_client()
    url = f"https://huggingface.co/api/datasets/{dataset_id}"
    response = await client.get(url)
    response.raise_for_status()
    
    # Format the JSON response for better readability
    data = response.json()
    return json.dumps(data, indent=2)

@mcp.tool()
async def get_model_info(model_id: str) -> str:
    """
    Get detailed metadata information for a HuggingFace model.
    
    Returns structured information including tags, license, downloads,
    likes, model configuration, pipeline info, and other metadata.
    
    Args:
        model_id: The model ID (e.g., 'username/model-name')
    
    Returns:
        JSON string with comprehensive model metadata
    """
    client = await get_client()
    url = f"https://huggingface.co/api/models/{model_id}"
    response = await client.get(url)
    response.raise_for_status()
    
    # Format the JSON response for better readability
    import json
    data = response.json()
    return json.dumps(data, indent=2)

@mcp.tool()
async def download_dataset_card(dataset_id: str) -> str:
    """
    Download the README card for a HuggingFace dataset.
    
    Args:
        dataset_id: The dataset ID (e.g., 'username/dataset-name')
    
    Returns:
        The content of the dataset card (README.md)
    """
    client = await get_client()
    url = f"https://huggingface.co/datasets/{dataset_id}/raw/main/README.md"
    response = await client.get(url)
    response.raise_for_status()
    return response.text

def main():
    """Main entry point for the MCP server"""
    mcp.run()

if __name__ == "__main__":
    main()