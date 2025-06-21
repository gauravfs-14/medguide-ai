# test_vectorize_mcp_setup.py - Extended Tests

import os
import chromadb
from chromadb.config import Settings
import pytest
import httpx
from typing import List

def test_chroma_setup():
    # Test path creation
    CHROMA_PATH = "data/vectorstore.db"
    assert os.path.exists(os.path.dirname(CHROMA_PATH)), "Data directory not created"
    
    # Test ChromaDB connection
    try:
        client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(
                allow_reset=True,
                is_persistent=True
            )
        )
    except Exception as e:
        pytest.fail(f"ChromaDB connection failed: {str(e)}")
    
    # Test collection creation
    try:
        collection = client.get_or_create_collection(
            name="text_vectors",
            metadata={"hnsw:space": "cosine"}
        )
        assert collection.name == "text_vectors", "Collection name mismatch"
    except Exception as e:
        pytest.fail(f"Collection creation failed: {str(e)}")

    # Test collection metadata
    assert collection.metadata["hnsw:space"] == "cosine", "Incorrect metadata"

def test_server_startup():
    # Test server import and initialization
    try:
        from vectorize_mcp import app
        assert app.name == "vector_store_mcp", "Server name mismatch"
    except ImportError as e:
        pytest.fail(f"Server import failed: {str(e)}")

def get_embeddings(text: str) -> List[float]:
    """Helper to get real embeddings from Ollama"""
    response = httpx.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text:v1.5",
            "prompt": text
        }
    )
    return response.json()["embeddings"]

def test_data_persistence():
    CHROMA_PATH = "data/vectorstore.db"
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection("text_vectors")
    
    # Get real embeddings from Ollama
    test_text = "Test persistence"
    try:
        test_embedding = get_embeddings(test_text)
    except Exception as e:
        pytest.skip(f"Skipping test: Could not get embeddings: {str(e)}")
    
    # Test vector insertion
    collection.add(
        embeddings=[test_embedding],
        documents=[test_text],
        ids=["test_id"],
        metadatas=[{"source": "test"}]
    )
    
    # Verify retrieval
    result = collection.get(ids=["test_id"])
    assert result["documents"][0] == test_text
    assert len(result["embeddings"][0]) > 0

def test_ollama_connection():
    try:
        response = httpx.get("http://localhost:11434/api/version")
        assert response.status_code == 200
    except:
        pytest.fail("Ollama server not responding")

def test_mcp_server_api():
    from vectorize_mcp import app
    
    # Verify server instance
    assert hasattr(app, 'router'), "MCP server missing router"
    
    # Get all registered routes
    routes = [route for route in app.router.routes]
    
    # Check for store_vector endpoint
    store_vector_routes = [
        route for route in routes 
        if getattr(route, "endpoint", None) and 
        route.endpoint.__name__ == "store_vector"
    ]
    
    assert len(store_vector_routes) > 0, "store_vector endpoint not found"
    
    # Verify tool registration
    tools = getattr(app, "tools", [])
    assert any(
        tool.__name__ == "store_vector" 
        for tool in tools
    ), "store_vector not registered as tool"