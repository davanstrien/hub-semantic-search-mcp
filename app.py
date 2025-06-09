#!/usr/bin/env python3
"""
MCP Server for Hugging Face Dataset and Model Search API using Gradio
"""

import os
import logging
from typing import Optional

import gradio as gr
import httpx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize HTTP client with longer timeout for MCP usage
client = httpx.Client(timeout=60.0)  # Increased timeout
base_url = os.getenv("HF_SEARCH_API_URL", "https://davanstrien-huggingface-datasets-search-v2.hf.space")


def search_datasets(
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
    
    Returns:
        Formatted search results with dataset IDs, summaries, and metadata
    """
    try:
        logger.info(f"Searching datasets: query='{query}', k={k}, sort_by='{sort_by}'")
        
        params = {
            "query": query,
            "k": k,
            "sort_by": sort_by,
            "min_likes": min_likes,
            "min_downloads": min_downloads
        }
        
        logger.info(f"Making request to: {base_url}/search/datasets")
        response = client.get(f"{base_url}/search/datasets", params=params)
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"Successfully retrieved {len(data.get('results', []))} results")
    except httpx.TimeoutException:
        logger.error(f"Request timed out for query: {query}")
        return "Request timed out. The search service may be slow or unavailable."
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
        return f"Search failed with HTTP error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return f"Search failed: {str(e)}"
    
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


def find_similar_datasets(
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
    
    Returns:
        Formatted list of similar datasets with metadata
    """
    params = {
        "dataset_id": dataset_id,
        "k": k,
        "sort_by": sort_by,
        "min_likes": min_likes,
        "min_downloads": min_downloads
    }
    
    response = client.get(f"{base_url}/similarity/datasets", params=params)
    response.raise_for_status()
    data = response.json()
    
    results = data.get("results", [])
    if not results:
        return "No similar datasets found."
    
    output = []
    for i, result in enumerate(results, 1):
        output.append(f"{i}. **{result['dataset_id']}**")
        output.append(f"   - Summary: {result['summary']}")
        output.append(f"   - Similarity: {result['similarity']:.3f}")
        output.append(f"   - Likes: {result['likes']:,} | Downloads: {result['downloads']:,}")
        output.append("")
    
    return "\n".join(output)


def search_models(
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
    
    Returns:
        Formatted search results with model IDs, summaries, and metadata
    """
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
    
    response = client.get(f"{base_url}/search/models", params=params)
    response.raise_for_status()
    data = response.json()
    
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


def find_similar_models(
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
    
    Returns:
        Formatted list of similar models with metadata
    """
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
    
    response = client.get(f"{base_url}/similarity/models", params=params)
    response.raise_for_status()
    data = response.json()
    
    results = data.get("results", [])
    if not results:
        return "No similar models found."
    
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


def get_trending_models(
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
    
    Returns:
        Formatted list of trending models with metadata
    """
    params = {
        "limit": limit,
        "min_likes": min_likes,
        "min_downloads": min_downloads,
        "min_param_count": min_param_count
    }
    if max_param_count is not None:
        params["max_param_count"] = max_param_count
    
    response = client.get(f"{base_url}/trending/models", params=params)
    response.raise_for_status()
    data = response.json()
    
    results = data.get("results", [])
    if not results:
        return "No trending models found."
    
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


def get_trending_datasets(
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
    
    Returns:
        Formatted list of trending datasets with metadata
    """
    params = {
        "limit": limit,
        "min_likes": min_likes,
        "min_downloads": min_downloads
    }
    
    response = client.get(f"{base_url}/trending/datasets", params=params)
    response.raise_for_status()
    data = response.json()
    
    results = data.get("results", [])
    if not results:
        return "No trending datasets found."
    
    output = []
    for i, result in enumerate(results, 1):
        output.append(f"{i}. **{result['dataset_id']}**")
        output.append(f"   - Summary: {result['summary']}")
        output.append(f"   - Similarity: {result['similarity']:.3f}")
        output.append(f"   - Likes: {result['likes']:,} | Downloads: {result['downloads']:,}")
        output.append("")
    
    return "\n".join(output)


def download_model_card(model_id: str) -> str:
    """
    Download the README card for a HuggingFace model.
    
    Args:
        model_id: The model ID (e.g., 'username/model-name')
    
    Returns:
        The content of the model card (README.md)
    """
    url = f"https://huggingface.co/{model_id}/raw/main/README.md"
    response = client.get(url)
    response.raise_for_status()
    return response.text


def download_dataset_card(dataset_id: str) -> str:
    """
    Download the README card for a HuggingFace dataset.
    
    Args:
        dataset_id: The dataset ID (e.g., 'username/dataset-name')
    
    Returns:
        The content of the dataset card (README.md)
    """
    url = f"https://huggingface.co/datasets/{dataset_id}/raw/main/README.md"
    response = client.get(url)
    response.raise_for_status()
    return response.text


# Create Gradio interface
with gr.Blocks(title="HuggingFace Search MCP Server") as demo:
    gr.Markdown("# HuggingFace Search MCP Server")
    gr.Markdown("This server provides semantic search capabilities for HuggingFace models and datasets.")
    gr.Markdown(f"**Backend API:** {base_url}")
    
    with gr.Tab("Search Datasets"):
        gr.Interface(
            fn=search_datasets,
            inputs=[
                gr.Textbox(label="Query", placeholder="Enter search query"),
                gr.Slider(1, 100, value=5, step=1, label="Number of results"),
                gr.Dropdown(["similarity", "likes", "downloads", "trending"], value="similarity", label="Sort by"),
                gr.Number(value=0, label="Minimum likes"),
                gr.Number(value=0, label="Minimum downloads")
            ],
            outputs=gr.Markdown(label="Results"),
            title="Search Datasets",
            description="Search for datasets based on a text query"
        )
    
    with gr.Tab("Find Similar Datasets"):
        gr.Interface(
            fn=find_similar_datasets,
            inputs=[
                gr.Textbox(label="Dataset ID", placeholder="username/dataset-name"),
                gr.Slider(1, 100, value=5, step=1, label="Number of results"),
                gr.Dropdown(["similarity", "likes", "downloads", "trending"], value="similarity", label="Sort by"),
                gr.Number(value=0, label="Minimum likes"),
                gr.Number(value=0, label="Minimum downloads")
            ],
            outputs=gr.Markdown(label="Results"),
            title="Find Similar Datasets",
            description="Find datasets similar to a specified dataset"
        )
    
    with gr.Tab("Search Models"):
        gr.Interface(
            fn=search_models,
            inputs=[
                gr.Textbox(label="Query", placeholder="Enter search query"),
                gr.Slider(1, 100, value=5, step=1, label="Number of results"),
                gr.Dropdown(["similarity", "likes", "downloads", "trending"], value="similarity", label="Sort by"),
                gr.Number(value=0, label="Minimum likes"),
                gr.Number(value=0, label="Minimum downloads"),
                gr.Number(value=0, label="Minimum parameter count"),
                gr.Number(value=None, label="Maximum parameter count (leave empty for no limit)")
            ],
            outputs=gr.Markdown(label="Results"),
            title="Search Models",
            description="Search for models based on a text query with optional parameter count filtering"
        )
    
    with gr.Tab("Find Similar Models"):
        gr.Interface(
            fn=find_similar_models,
            inputs=[
                gr.Textbox(label="Model ID", placeholder="username/model-name"),
                gr.Slider(1, 100, value=5, step=1, label="Number of results"),
                gr.Dropdown(["similarity", "likes", "downloads", "trending"], value="similarity", label="Sort by"),
                gr.Number(value=0, label="Minimum likes"),
                gr.Number(value=0, label="Minimum downloads"),
                gr.Number(value=0, label="Minimum parameter count"),
                gr.Number(value=None, label="Maximum parameter count (leave empty for no limit)")
            ],
            outputs=gr.Markdown(label="Results"),
            title="Find Similar Models",
            description="Find models similar to a specified model"
        )
    
    with gr.Tab("Trending Models"):
        gr.Interface(
            fn=get_trending_models,
            inputs=[
                gr.Slider(1, 100, value=10, step=1, label="Number of results"),
                gr.Number(value=0, label="Minimum likes"),
                gr.Number(value=0, label="Minimum downloads"),
                gr.Number(value=0, label="Minimum parameter count"),
                gr.Number(value=None, label="Maximum parameter count (leave empty for no limit)")
            ],
            outputs=gr.Markdown(label="Results"),
            title="Get Trending Models",
            description="Get trending models with their summaries and optional filtering"
        )
    
    with gr.Tab("Trending Datasets"):
        gr.Interface(
            fn=get_trending_datasets,
            inputs=[
                gr.Slider(1, 100, value=10, step=1, label="Number of results"),
                gr.Number(value=0, label="Minimum likes"),
                gr.Number(value=0, label="Minimum downloads")
            ],
            outputs=gr.Markdown(label="Results"),
            title="Get Trending Datasets",
            description="Get trending datasets with their summaries"
        )
    
    with gr.Tab("Download Model Card"):
        gr.Interface(
            fn=download_model_card,
            inputs=gr.Textbox(label="Model ID", placeholder="username/model-name"),
            outputs=gr.Textbox(label="Model Card Content", lines=20),
            title="Download Model Card",
            description="Download the README card for a HuggingFace model"
        )
    
    with gr.Tab("Download Dataset Card"):
        gr.Interface(
            fn=download_dataset_card,
            inputs=gr.Textbox(label="Dataset ID", placeholder="username/dataset-name"),
            outputs=gr.Textbox(label="Dataset Card Content", lines=20),
            title="Download Dataset Card",
            description="Download the README card for a HuggingFace dataset"
        )


if __name__ == "__main__":
    # Launch with MCP server enabled
    demo.launch(mcp_server=True)